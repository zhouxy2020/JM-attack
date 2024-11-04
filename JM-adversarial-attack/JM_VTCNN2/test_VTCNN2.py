"""
CNN1D
"""
import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from raw_model_training import RRR, mcldnn, VTCNN2
import pickle
import matplotlib.pyplot as plt

def train_model(SigPow, args, batch_signal, model, noise):
    model.eval()  # Set model to evaluate mode
    PSR_vec = np.arange(-20, -9, 1)
    # PSR_vec = [-10]
    SNR = 10
    with open('VTCNN2_JM_UAP.pkl', 'rb') as f:
        UAP_jm = pickle.load(f)
    with open('VTCNN2_FG_UAP.pkl', 'rb') as f:
        UAP_fg = pickle.load(f)
    with open('VTCNN2_Deepfool_UAP.pkl', 'rb') as f:
        UAP_deepfool = pickle.load(f)
    with open('VTCNN2_PCA_UAP.pkl', 'rb') as f:
        UAP_pca = pickle.load(f)
    acc_fg = np.zeros(len(PSR_vec))
    acc_jm = np.zeros(len(PSR_vec))
    acc_deepfool = np.zeros(len(PSR_vec))
    acc_pca = np.zeros(len(PSR_vec))
    acc_jam = np.zeros(len(PSR_vec))
    acc_clean = np.zeros(len(PSR_vec))
    for icn in range(len(PSR_vec)):
        PSR = PSR_vec[icn]
        PNR = PSR + SNR
        print('PSR = ', PNR - SNR)
        aid = 10 ** ((PNR - SNR) / 10)
        Epsilon_uni = np.sqrt(SigPow * aid)
        print('Epsilon_uni', Epsilon_uni)
        signal_num = 0
        UAP_fg = Epsilon_uni * (1 / np.linalg.norm(UAP_fg)) * UAP_fg
        UAP_jm = Epsilon_uni * (1 / np.linalg.norm(UAP_jm)) * UAP_jm
        UAP_deepfool = Epsilon_uni * (1 / np.linalg.norm(UAP_deepfool)) * UAP_deepfool
        UAP_pca = Epsilon_uni * (1 / np.linalg.norm(UAP_pca)) * UAP_pca
        jam = Epsilon_uni * (1 / np.linalg.norm(noise)) * noise
        corrects_fg,corrects_jm,corrects_deepfool=0,0,0
        corrects_pca, corrects_jam, corrects_clean = 0, 0, 0
        pbar = tqdm(batch_signal['test'])
        for inputs, labels in pbar:
            if inputs.shape[0]== args.batch_size :
                inputs = inputs.cuda()  # (batch_size, 2, 128)
                labels = labels.cuda()  # (batch_size, )
                _, preds_fg = torch.max(model(inputs + torch.FloatTensor(UAP_fg).repeat(args.batch_size, 1, 1).cuda())[-1], 1)
                corrects_fg += torch.sum(preds_fg == labels.data)
                _, preds_jm = torch.max(model(inputs + torch.FloatTensor(UAP_jm).repeat(args.batch_size, 1, 1).cuda())[-1], 1)
                corrects_jm += torch.sum(preds_jm == labels.data)
                _, preds_deepfool = torch.max(model(inputs + torch.FloatTensor(UAP_deepfool).repeat(args.batch_size, 1, 1).cuda())[-1], 1)
                corrects_deepfool += torch.sum(preds_deepfool == labels.data)
                _, preds_pca = torch.max(model(inputs + torch.FloatTensor(UAP_pca).repeat(args.batch_size, 1, 1).cuda())[-1], 1)
                corrects_pca += torch.sum(preds_pca == labels.data)
                _, preds_jam = torch.max(model(inputs + torch.FloatTensor(jam).repeat(args.batch_size, 1, 1).cuda())[-1], 1)
                corrects_jam += torch.sum(preds_jam == labels.data)
                _, preds_clean = torch.max(model(inputs)[-1], 1)
                corrects_clean += torch.sum(preds_clean == labels.data)
                signal_num += inputs.size(0)
        acc_fg[icn] = corrects_fg / signal_num
        acc_jm[icn] = corrects_jm / signal_num
        acc_deepfool[icn] = corrects_deepfool / signal_num
        acc_pca[icn] = corrects_pca / signal_num
        acc_jam[icn] = corrects_jam / signal_num
        acc_clean[icn] = corrects_clean / signal_num
    a=1

    fig, ax = plt.subplots()
    ax.plot(PSR_vec,100 * acc_clean*np.ones(11),'k>-', label='No attack')
    ax.plot(PSR_vec, 100 * acc_jam, color='purple',marker='h',linestyle='-', label='Jamming attack')
    ax.plot(PSR_vec, 100 * acc_deepfool, 'cs-', label='Deepfool-UAP attack')
    ax.plot(PSR_vec, 100 * acc_pca,'ro-', label='PCA-UAP attack')
    # ax.plot(PSR_vec, 100 * np.mean(acc_acos,axis=0), color='darkgreen',marker='+',linestyle='-', label='ACos-UAP attack - SNR=10 dB')
    ax.plot(PSR_vec, 100 * acc_fg, 'y*-', label='FG-UAP attack')
    ax.plot(PSR_vec, 100 * acc_jm, 'b^-', label='JM-UAP attack')
    plt.legend(loc='lower left')
    y_major_locator = plt.MultipleLocator(10)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xlabel('PSR [dB]', fontdict={'size': 10})
    ax.set_ylabel('Accuracy %', fontdict={'size': 10})
    plt.xticks(size=10)
    plt.yticks(size=10)
    ax.grid(True)
    plt.savefig("whith_vtcnn2.png", bbox_inches='tight', dpi=300)
    plt.show()

    info = {
        'acc_deepfool': acc_deepfool,
        'acc_pca': acc_pca,
        'acc_jam': acc_jam,
        'acc_fg_all': acc_fg,
        'acc_clean': acc_clean,
        'acc_jm': acc_jm,
        # 'acc_all':acc_all
    }
    with open('VTCNN2_white_attack_acc.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(info, f)

    with open('VTCNN2_white_attack_acc.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
        temp = pickle.load(f)
        acc_deepfool = temp['acc_deepfool']
        acc_pca = temp['acc_deepfool']
        acc_jam = temp['acc_jam']
        acc_fg = temp['acc_fg_all']
        acc_clean = temp['acc_clean']
        acc_jm = temp['acc_jm']
    return a

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
    noise_mean = np.sum(np.mean(X_test.reshape(-1,256),axis=1)) / X_test.shape[0]
    noise_std = np.sum(np.std(X_test.reshape(-1,256),axis=1)) / X_test.shape[0]
    noise = np.random.normal(noise_mean,noise_std,[1,2,128])
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
    return batch_signal, data_sizes, Epsilon_uni, noise

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
    parser.set_defaults(lr=0.0002, batch_size=10, num_epochs=10, num_workers=4, model='VTCNN2')
    return parser.parse_args()


def main():
    prog_args = arg_parse()
    batch_signal, data_sizes, SigPow, noise = prepare_data(prog_args)  # 跳转到prepare_data函数，得到批训练集和批测试集
    model_ft = load_model(prog_args.model)
    train_model(SigPow, prog_args, batch_signal, model_ft, noise)

if __name__ == '__main__':
    main()
