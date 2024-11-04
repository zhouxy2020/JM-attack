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

torch.manual_seed(0)
# 为了确保在多线程情况下的可复现性，还需要设置这个
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 设置NumPy的随机数种子
np.random.seed(0)
# 设置Python内置随机模块的随机数种子
random.seed(0)

def pca_gen(x, Epsilon_uni, args, batch_signal,  model , optimizer,  num_epochs=50):

    num_class=11
    model.eval()  # Set model to evaluate mode
    N = 50
    grad_matrix_n = np.zeros([N, 256])
    ctr_index = 0
    pbar = tqdm(batch_signal['train'])
    for inputs, labels in pbar:
        if ctr_index<N:
            inputs = inputs.cuda()  # (batch_size, 2, 128)
            labels = labels.cuda()  # (batch_size, )
            _, predicted_clean = torch.max(model(inputs)[-1], 1)
            var_samples = Variable(inputs, requires_grad=True).cuda()
            logit_preds = model(var_samples)[-1]
            # y_target = np.eye(num_class)[labels.data.cpu().numpy(), :].reshape(1, num_class)
            # var_ys = Variable(torch.tensor(y_target)).cuda()
            loss = torch.nn.CrossEntropyLoss()(logit_preds, labels)
            loss.backward(retain_graph=True)
            gradient_sign = var_samples.grad
            temp = torch.reshape(gradient_sign, (1, 256)).data.cpu().numpy()
            grad_matrix_n[ctr_index,:] = temp / (np.linalg.norm(temp) + 0.00000001)
            ctr_index+=1
    _,_,v_n_T = np.linalg.svd(grad_matrix_n)
    universal_per = Epsilon_uni * (1 / np.linalg.norm(v_n_T.T[:,0])) * v_n_T.T[:,0]
    universal_per = universal_per.reshape([1,2,128]).astype(np.float32)
    with open(args.model + '_PCA_UAP.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(universal_per, f)
    return model

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
    X_200 = list(map(lambda i: (X_train[np.where(np.argmax(Y_train, axis=1) == i)][:50]), np.arange(11)))
    Y_200 = list(map(lambda i: (Y_train[np.where(np.argmax(Y_train, axis=1) == i)][:50]), np.arange(11)))
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
    if  model == 'r8conv1':
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
    parser.set_defaults(lr=0.0002, batch_size=1, num_epochs=10, num_workers=4, model='VTCNN2')
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
    # 训练模型
    pca_gen(x, Epsilon_uni, prog_args, batch_signal, model_ft, optimizer_ft , num_epochs=prog_args.num_epochs)

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

