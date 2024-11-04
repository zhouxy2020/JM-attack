"""
CNN1D
"""
import argparse
import os
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from raw_model_training import RRR,  mcldnn,  VTCNN2
from torch.autograd import Variable
import pickle
import random

torch.manual_seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(10)
random.seed(10)

def adjusted_cosine_similarity(v1, v2, eps=1e-8):
    x1 = v1.reshape(v1.shape[0], -1)
    x2 = v2.reshape(v2.shape[0], -1)
    # 计算x1和x2的L2范数
    norm_x1 = torch.norm(x1, dim=1, keepdim=True)
    norm_x2 = torch.norm(x2, dim=1, keepdim=True)
    # 计算x1和x2的平均长度
    avg_norm = (norm_x1 + norm_x2) / 2
    # 计算余弦相似度
    cos_sim = torch.mm(x1, x2.t()) / (norm_x1 * norm_x2.t())
    # 对余弦相似度进行调整
    # 这里我们使用平均长度的乘积来调整余弦相似度
    adjusted_cos_sim = cos_sim * avg_norm / torch.mean(avg_norm)
    # 限制结果范围在[-1, 1]之间
    # adjusted_cos_sim = torch.clamp(adjusted_cos_sim, min=-1 + eps, max=1 - eps)
    return adjusted_cos_sim.cuda()

def pearson_correlation_coefficient(x1, x2):
    x1 = x1.reshape(x1.shape[0], -1)
    x2 = x2.reshape(x2.shape[0], -1)
    mean1 = torch.mean(x1, dim=1, keepdim=True)
    mean2 = torch.mean(x2, dim=1, keepdim=True)
    # 调整张量，使其中心化（即均值为0）
    x1_centered = x1 - mean1
    x2_centered = x2 - mean2
    # 计算标准差
    std1 = torch.std(x1_centered, dim=1, keepdim=True)
    std2 = torch.std(x2_centered, dim=1, keepdim=True)
    # 避免除以零
    eps = 1e-8
    std1 = std1.clamp(min=eps)
    std2 = std2.clamp(min=eps)
    # 计算协方差
    covariance = torch.sum(x1_centered * x2_centered, dim=1) / (x1.size(1) - 1)
    # 计算Pearson相关系数
    pearson_corr = covariance / (std1 * std2)
    return pearson_corr.cuda()

def cosine_similarity(v1, v2):
    x1 = v1.reshape(v1.shape[0], -1)
    x2 = v2.reshape(v2.shape[0], -1)
    R_num = torch.matmul(x1, x2.T).diag()
    X1_norm = torch.norm(x1, p=2, dim=1)
    X2_norm = torch.norm(x2, p=2, dim=1)
    R_den = X1_norm * X2_norm
    R = R_num /  (R_den+0.000001)
    return R

def train_model(x, Epsilon_uni, args, batch_signal,  model , optimizer,  scheduler, num_epochs=50):

    best_loss = 100.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}------------------------'.format(epoch, num_epochs - 1))
        running_loss = 0.0
        running_corrects = 0
        signal_num = 0
        pbar = tqdm(batch_signal['train'])
        for inputs, labels in pbar:
            inputs = inputs.cuda()  # (batch_size, 2, 128)
            labels = labels.cuda()  # (batch_size, )
            # zero the parameter gradients
            optimizer.zero_grad()
            p_n = Epsilon_uni * torch.div(x, torch.norm(x))
            out_ori = model(inputs)
            _, preds = torch.max(out_ori[-1], 1)
            feature_ori = out_ori[-2]
            out_adv = model(inputs+p_n.repeat(args.batch_size, 1,1))
            _, preds_adv = torch.max(out_adv[-1], 1)
            feature_adv = out_adv[-2]
            cos_loss = cosine_similarity(feature_ori, feature_adv)
            loss = cos_loss.mean()
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds_adv == labels.data)
            signal_num += inputs.size(0)
        scheduler.step()
        epoch_loss = running_loss / signal_num
        epoch_acc = running_corrects.double() / signal_num
        print('epoch:{}, epoch_acc:{}'.format(epoch, epoch_acc))
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            best_x = x
            print('epoch:{}, FG best loss:{}'.format(epoch, best_loss))
    # 保存训练精度最高时的扰动
    with open(args.model+'_FG_UAP.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(best_x.data.cpu().numpy(), f)
    ###########  test
    running_loss = 0.0
    running_corrects = 0
    signal_num = 0
    p_n = Epsilon_uni * torch.div(best_x, torch.norm(best_x) + 0.00001)
    pbar = tqdm(batch_signal['test'])
    for inputs, labels in pbar:
        # if inputs.shape[0]== args.batch_size :
        inputs = inputs.cuda()  # (batch_size, 2, 128)
        labels = labels.cuda()  # (batch_size, )
        out_ori = model(inputs)
        _, preds = torch.max(out_ori[-1], 1)
        out_adv = model(inputs + p_n.repeat(args.batch_size, 1, 1))
        _, preds_adv = torch.max(out_adv[-1], 1)
        running_corrects += torch.sum(preds_adv == labels.data)
        signal_num += inputs.size(0)
        epoch_loss = running_loss / signal_num
        epoch_acc = running_corrects.double() / signal_num
    print('{}  Acc: {:.4f}'.format('test', epoch_acc))
    return x

def ModCls_loaddata():
    import numpy as np
    import pickle as cPickle
    with open('/home/data/RML2016.10a/RML2016.10a_dict.pkl', 'rb') as ff:
        u = cPickle._Unpickler(ff)
        u.encoding = 'latin1'
        Xd = u.load()
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
    X = np.vstack(X)
    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.5)
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))
    X_train = X[train_idx]
    X_test = X[test_idx]
    def to_onehot(yin):
        yy = list(yin)  # This is a workaround as the map output for python3 is not a list
        yy1 = np.zeros([len(list(yy)), max(yy) + 1])
        yy1[np.arange(len(list(yy))), yy] = 1
        return yy1
    Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
    classes = mods
    return (X, lbl, X_train, X_test, classes, snrs, mods, Y_train, Y_test, train_idx, test_idx)

def prepare_data(args):
    # 导入数据集
    X_loaded, lbl, X_train, X_test, classes, snrs, mods, Y_train, Y_test, train_idx, test_idx = ModCls_loaddata()
    SNR = 10
    # TRAINING
    train_SNRs = list(map(lambda x: lbl[x][1], train_idx))
    X_train = X_train[np.where(np.array(train_SNRs) == SNR)]#.reshape(-1, 2, 128, 1)
    Y_train = Y_train[np.where(np.array(train_SNRs) == SNR)]
    X_200 = list(map(lambda i: (X_train[np.where(np.argmax(Y_train, axis=1) == i)][:200]), np.arange(11)))
    Y_200 = list(map(lambda i: (Y_train[np.where(np.argmax(Y_train, axis=1) == i)][:200]), np.arange(11)))
    X_train = np.concatenate(X_200, axis=0)
    Y_train = np.concatenate(Y_200, axis=0)
    # TEST
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    X_test = X_test[np.where(np.array(test_SNRs) == SNR)]#.reshape(-1, 2, 128, 1)
    Y_test = Y_test[np.where(np.array(test_SNRs) == SNR)]
    SigPow = (np.sum(np.linalg.norm(X_test.reshape([-1, 256]), axis=1)) / X_test.shape[0]) ** 2
    PNR = 0
    aid = 10 ** ((PNR - SNR) / 10)
    Epsilon_uni = np.sqrt(SigPow * aid)
    train_2 = torch.from_numpy(X_train).type(torch.FloatTensor)
    test_2 = torch.from_numpy(X_test).type(torch.FloatTensor)
    train_label = torch.argmax(torch.from_numpy(Y_train).type(torch.LongTensor), dim=1)
    test_label = torch.argmax(torch.from_numpy(Y_test).type(torch.LongTensor), dim=1)
    data_sizes = {'train': len(train_label), 'test': len(test_label)}
    # 把数据放在数据库中
    train_signal = torch.utils.data.TensorDataset(train_2, train_label)
    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    # 将训练集和测试集分批
    batch_signal = {'train': torch.utils.data.DataLoader(dataset=train_signal, batch_size=args.batch_size,
                                                         shuffle=True, num_workers=args.num_workers),
                    'test': torch.utils.data.DataLoader(dataset=test_signal, batch_size=args.batch_size,
                                                        shuffle=False, num_workers=args.num_workers)}
    return batch_signal, data_sizes, Epsilon_uni

def load_model(model):
    if model == 'r8conv1':
        model_ft = RRR.resnet(depth=8).cuda()
    elif model == 'VTCNN2':
        model_ft = VTCNN2.VTCNN2().cuda()
    else:
        print('Error! There is no model of your choice')
    print(model_ft)
    # 导入预训练模型
    model_ft.load_state_dict(
        torch.load("../raw_model_training/result/model/"+model+"_best_lr=0.001.pth"))
    model_ft.eval()
    return model_ft

def arg_parse():
    parser = argparse.ArgumentParser(description='Signal_diagonal_matrix_CNN arguments.')
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--num_workers', dest='num_workers', type=int, help='Number of workers to load data.')
    parser.add_argument('--model', dest='model', type=str, help='Model: Resnet8, VTCNN2')
    parser.set_defaults(lr=0.0002, batch_size=10, num_epochs=20, num_workers=2, model='VTCNN2')
    return parser.parse_args()

def main():
    prog_args = arg_parse()
    batch_signal, data_sizes, Epsilon_uni = prepare_data(prog_args)  # 跳转到prepare_data函数，得到批训练集和批测试集
    model_ft = load_model(prog_args.model)
    xx = Variable(torch.distributions.uniform.Uniform(low=-0.001, high=0.001).sample(sample_shape=([1, 2, 128]))).cuda()
    x = Variable(Epsilon_uni * torch.div(xx, torch.norm(xx) + 0.00001)).cuda()
    x.requires_grad = True
    # 优化器
    optimizer_ft = optim.Adam([x], lr=prog_args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.9)
    # 训练模型
    train_model(x, Epsilon_uni, prog_args, batch_signal, model_ft, optimizer_ft , exp_lr_scheduler, num_epochs=prog_args.num_epochs)


if __name__ == '__main__':
    main()

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # # ax.plot(np.arange(128), UAP_deepfool_all[1,0,:,0], 'c-', label='Deepfool-UAP')
    # ax.plot(np.arange(128), p_n.data.cpu().numpy()[0, 0, :], 'r-', label='I channal')
    # # ax.plot(np.arange(128),UAP_fg[0,0,:]*0.7,'y-', label='FG-UAP')
    # ax.plot(np.arange(128), p_n.data.cpu().numpy()[0, 1, :], 'b-', label='Q channal')
    # # ax.plot(PSR_vec, 100 * np.mean(acc_pearacos,axis=0), 'b^-', label='JM-UAP attack')
    # plt.legend(loc='lower left')
    # ax.set_ylabel('Amplitude', fontdict={'size': 10})
    # ax.set_xlabel('Time', fontdict={'size': 10})
    # plt.xticks(size=10)
    # plt.yticks(size=10)
    # # plt.ylim(-0.006,0.006)
    # # ax.grid(True)
    # # plt.savefig("white_attack_acc.png",bbox_inches='tight',dpi=300)
    # plt.show()
    #
    #

