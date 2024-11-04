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

def deepfool_gen(x, Epsilon_uni, args, batch_signal, model, num_epochs=50):
    # best_model_wts = copy.deepcopy(model.state_dict())
    def fgm_ModCls(in_img, in_label, num_class):
        # model.train()
        var_samples = Variable(in_img, requires_grad=True).cuda()
        logit_preds = model(var_samples)[-1]
        eps_acc = 0.00001 * torch.norm(in_img)
        epsilon_vector = torch.zeros([num_class])
        for cls in range(num_class):
            y_target = np.eye(num_class)[cls, :].reshape(1, num_class)
            var_ys = Variable(torch.tensor(y_target)).cuda()
            loss = torch.nn.CrossEntropyLoss()(logit_preds, var_ys)
            loss.backward(retain_graph=True)
            gradient_sign = var_samples.grad
            adv_per_needtoreshape = -1 * gradient_sign
            norm_adv_per = adv_per_needtoreshape / (torch.norm(adv_per_needtoreshape) + 0.000000000001)
            epsilon_max = torch.norm(in_img)
            epsilon_min = 0
            num_iter = 0
            wcount = 0
            while (epsilon_max - epsilon_min > eps_acc) and (num_iter < 30):
                wcount = wcount + 1
                num_iter = num_iter + 1
                epsilon = (epsilon_max + epsilon_min) / 2
                adv_img_givencls = in_img + (epsilon * norm_adv_per)

                predicted_probabilities = model(adv_img_givencls)[-1]

                compare = torch.equal(torch.argmax(predicted_probabilities), torch.argmax(torch.tensor(in_label).cuda()))
                if compare:
                    epsilon_min = epsilon
                else:
                    epsilon_max = epsilon
            # print('epsilon:', epsilon)
            epsilon_vector[cls] = epsilon + eps_acc
            # print('the adversarial prediction is:', predicted_probabilities)
            # print(adv_img_givencls[0,0,10:14])
        false_cls = torch.argmin(epsilon_vector)
        minimum_epsilon = torch.min(epsilon_vector)

        y_target = np.eye(num_class)[false_cls, :].reshape(1, num_class)
        var_ys = Variable(torch.tensor(y_target)).cuda()
        loss = torch.nn.CrossEntropyLoss()(logit_preds, var_ys)
        loss.backward(retain_graph=True)
        adv_dirc = var_samples.grad

        norm_adv_dirc = adv_dirc / (torch.norm(adv_dirc) + 0.000000000001)
        adv_perturbation = minimum_epsilon * norm_adv_dirc
        adv_image = in_img + adv_perturbation
        # print('Tue label is ', np.argmax(in_label), 'and the adversay label', np.argmax(sess.run(predictions, feed_dict={X: adv_image, is_training: False})))
        # print(minimum_epsilon)
        return adv_image, adv_perturbation, false_cls, minimum_epsilon

    model.eval()  # Set model to evaluate mode
    best_acc = 100.
    for epoch in range(num_epochs):
        print('Epoch {}/{}----------------'.format(epoch, num_epochs - 1))
        # Each epoch has a training and validation phase
        # for phase in ['train','test']:
        universal_per = torch.distributions.uniform.Uniform(low=-0.001, high=0.001).sample(sample_shape=([1, 2, 128])).cuda()
        running_corrects = 0
        signal_num = 0
        true_rate = 0
        pbar = tqdm(batch_signal['train'])
        for inputs, labels in pbar:
            inputs = inputs.cuda()  # (batch_size, 2, 128)
            labels = labels.cuda()  # (batch_size, )
            _, predicted_clean = torch.max(model(inputs)[-1], 1)
            _, predicted_adv = torch.max(model(inputs+universal_per)[-1], 1)
            if predicted_adv == predicted_clean:  # predicted_label == np.argmax(train_Y_0[cnr_index]):
                true_rate = true_rate + 1
                # First we need to find adverssarial direction for this instant  by solving (1), or using fgm or deepfool
                fgm_start_time = time.perf_counter()
                _, adv_perturbation, _, _ = fgm_ModCls((inputs + universal_per),np.eye(11)[predicted_clean, :], 11)
                fgm_time = time.perf_counter() - fgm_start_time
                # Second we need to revise the universal perturbation
                if torch.norm(universal_per + adv_perturbation) < Epsilon_uni:
                    universal_per = universal_per + adv_perturbation
                else:
                    universal_per = Epsilon_uni * torch.div((universal_per + adv_perturbation), torch.norm((universal_per + adv_perturbation)) + 0.00001)
            # statistics
            running_corrects += torch.sum(predicted_adv == labels.data)
            signal_num += inputs.size(0)
            # 在进度条的右边实时显示数据集类型、loss值和精度
            epoch_acc = running_corrects.double() / signal_num
        universal_per = Epsilon_uni * torch.div(universal_per, torch.norm(universal_per) + 0.00001)
        if epoch_acc < best_acc:
            best_acc = epoch_acc
            best_x = universal_per
    with open(args.model + '_Deepfool_UAP.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(best_x.data.cpu().numpy(), f)
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
    parser.set_defaults(lr=0.0002, batch_size=1, num_epochs=10, num_workers=4, model='VTCNN2')
    return parser.parse_args()

def main():
    prog_args = arg_parse()
    batch_signal, data_sizes, Epsilon_uni = prepare_data(prog_args)  # 跳转到prepare_data函数，得到批训练集和批测试集
    model_ft = load_model(prog_args.model)
    criterion = nn.CrossEntropyLoss()
    # 将tensor封装成Variable
    xx = Variable(torch.distributions.uniform.Uniform(low=-0.001, high=0.001).sample(sample_shape=([1, 2, 128]))).cuda()
    x = Variable(Epsilon_uni * torch.div(xx, torch.norm(xx) + 0.00001)).cuda()
    # 优化器
    optimizer_ft = optim.Adam([x], lr=prog_args.lr)
    # optimizer_ft = optim.SGD([x], lr=prog_args.lr,weight_decay=0.1)
    # 学习率衰减
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.8)
    # 训练模型
    deepfool_gen(x, Epsilon_uni, prog_args, batch_signal, model_ft, num_epochs=prog_args.num_epochs)


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

