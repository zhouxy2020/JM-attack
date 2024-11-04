"""
CNN1D
"""
import argparse
import os
import numpy as np
import time
import csv
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from raw_model_training import RRR, CNN1D, CNN2D, lstm, gru, mcldnn, Alexnet, LeNet, vgg16, VTCNN2
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
from audtorch.metrics.functional import pearsonr
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hypertools as hyp

def train_model(args, SigPow, batch_signal, classes, model):
    model.eval()  # Set model to evaluate mode
    with open('CNN2D_Deepfool_UAP.pkl', 'rb') as f:
        UAP_deepfool = pickle.load(f)
    PSR = -10
    aid = 10 ** (PSR / 10)
    Epsilon_uni = np.sqrt(SigPow * aid)
    print('Epsilon_uni', Epsilon_uni)
    UAP_deepfool = Epsilon_uni * (1 / np.linalg.norm(UAP_deepfool)) * UAP_deepfool
    embs, embs_adv = [], []
    labels = []
    pbar = tqdm(batch_signal['test'])
    for inputs, target in pbar:
        inputs = inputs.cuda()  # (batch_size, 2, 128)
        target = target.cuda()  # (batch_size, )
        out_ori = model(inputs)
        _, preds = torch.max(out_ori[-1], 1)
        feature_ori = out_ori[-2]
        embs.append(feature_ori.data.cpu().numpy())
        labels.append(target.data.cpu().numpy())

        out_adv = model(inputs + torch.FloatTensor(UAP_deepfool).repeat(args.batch_size, 1, 1).cuda())
        _, preds_adv = torch.max(out_adv[-1], 1)
        feature_adv = out_adv[-2]
        embs_adv.append(feature_adv.data.cpu().numpy())

    embs = np.concatenate(embs)
    embs_adv = np.concatenate(embs_adv)
    labels = np.concatenate(labels)

    feat_cent, feat_adv_cent = [], []
    for lbi in range(11):
        feat_cent.append(np.mean(embs[labels == lbi], axis=0)[np.newaxis, :])
        feat_adv_cent.append(np.mean(embs_adv[labels == lbi], axis=0)[np.newaxis, :])
    feat_cent = np.concatenate(feat_cent)
    feat_adv_cent = np.concatenate(feat_adv_cent)

    # 我选择了一些较清楚的颜色，更多的类时也能画清晰
    css4 = list(mcolors.CSS4_COLORS.keys())
    color_ind = [2, 7, 9, 10, 11, 13, 14, 16, 17, 19, 20, 21, 25, 28, 30, 31, 32, 37, 38, 40, 47, 51,
                 55, 60, 65, 82, 85, 88, 106, 110, 115, 118, 120, 125, 131, 135, 139, 142, 146, 147]
    css4 = [css4[v] for v in color_ind]

    tsne = TSNE(n_components=2, learning_rate=100, metric='cosine', n_jobs=-1)
    tsne.fit_transform(embs)
    outs_2d = np.array(tsne.embedding_)
    for lbi in range(11):
        temp = outs_2d[labels == lbi]
        plt.plot(temp[:, 0], temp[:, 1], '.', color=css4[lbi], label=classes[lbi])
    # plt.title('feats dimensionality reduction visualization by tSNE,test data')
    plt.legend(loc='lower left')
    plt.savefig("feats_visualization.png", bbox_inches='tight', dpi=300)
    plt.show()

    embs_cent = []
    for lbi in range(11):
        temp = embs[labels == lbi]
        embs_cent.append(temp - np.mean(temp, axis=0))
    embs_cent = np.concatenate(embs_cent)
    embs_adv_cent = []
    for lbi in range(11):
        temp = embs_adv[labels == lbi]
        embs_adv_cent.append(temp - np.mean(temp, axis=0))
    embs_adv_cent = np.concatenate(embs_adv_cent)

    # tsne_cent = TSNE(n_components=2, learning_rate=100, metric='cosine', n_jobs=-1)
    # tsne_cent.fit_transform(embs_cent)
    # outs_2d_cent = np.array(tsne_cent.embedding_)
    # for lbi in range(11):
    #     temp = outs_2d_cent[labels == lbi]
    #     plt.plot(temp[:, 0], temp[:, 1], '.', color=css4[lbi], label=classes[lbi])
    # # plt.title('feats dimensionality reduction visualization by tSNE,test data')
    # plt.legend(loc='lower left')
    # plt.savefig("feats_visualization_cent.png", bbox_inches='tight', dpi=300)
    # plt.show()

    #==================
    # tsne_adv = TSNE(n_components=2, learning_rate=100, metric='cosine', n_jobs=-1)
    # tsne_adv.fit_transform(embs_adv)
    # outs_2d_adv = np.array(tsne_adv.embedding_)
    # for lbi in range(11):
    #     temp = outs_2d_adv[labels == lbi]
    #     plt.plot(temp[:, 0], temp[:, 1], '.', color=css4[lbi], label=classes[lbi])
    # # plt.title('feats dimensionality reduction visualization by tSNE,test data')
    # plt.legend(loc='lower left')
    # plt.savefig("feats_adv_visualization.png", bbox_inches='tight', dpi=300)
    # plt.show()
    #
    # # ==================

    # tsne_adv_cent = TSNE(n_components=2, learning_rate=100, metric='cosine', n_jobs=-1)
    # tsne_adv_cent.fit_transform(embs_adv_cent)
    # outs_2d_adv_cent = np.array(tsne_adv_cent.embedding_)
    # for lbi in range(11):
    #     temp = outs_2d_adv_cent[labels == lbi]
    #     plt.plot(temp[:, 0], temp[:, 1], '.', color=css4[lbi], label=classes[lbi])
    # # plt.title('feats dimensionality reduction visualization by tSNE,test data')
    # plt.legend(loc='lower left')
    # plt.savefig("feats_adv_visualization_cent.png", bbox_inches='tight', dpi=300)
    # plt.show()

    a=1
    #=================
    from scipy.spatial.distance import cosine
    cosine_distance, cosine_distance_cent = np.zeros(embs_adv.shape[0]), np.zeros(embs_adv.shape[0])
    for i in range(embs_adv.shape[0]):
        cosine_distance[i] = cosine(embs_adv[i], embs[i])
        cosine_distance_cent[i] = cosine(embs_adv_cent[i], embs_cent[i])
    cos_dis_cout = np.zeros(11)
    cos_dis_cout_cent = np.zeros(11)
    for lbi in range(11):
        cos_dis_cout[lbi] = np.mean(cosine_distance[labels == lbi])
        cos_dis_cout_cent[lbi] = np.mean(cosine_distance_cent[labels == lbi])

    plt.figure(figsize=(10, 6))
    width = np.array([0.35, 0.35])  # 可以根据需要调整宽度
    # 计算每个条形的位置
    x1 = np.arange(len(classes))
    x2 = x1 + width[0]  # 第二组条形的位置
    # 创建条形图
    plt.bar(x1, cos_dis_cout, width[0], label='Group 1', color='blue')
    plt.bar(x2, cos_dis_cout_cent, width[0], label='Group 2', color='red')
    # 添加标题和标签
    classes[7] = '16QAM'
    classes[8] = '64QAM'
    # plt.xlabel('')
    plt.ylabel('Cos similarity', size=14)
    plt.xticks(x1 + width[0] / 2, classes, rotation=45, size=14)
    plt.legend()
    plt.yticks(size=14)
    plt.tight_layout()
    plt.savefig("feature_distance.png", bbox_inches='tight', dpi=300)
    plt.show()
    #===================

    # cosine_distance,cosine_distance_cent = 0,0
    # for i in range(embs_adv.shape[0]):
    #     cosine_distance += cosine(embs_adv[i], embs[i])
    #     cosine_distance_cent += np.corrcoef(embs_adv_cent[i], embs_cent[i])[0, 1]
    #
    # pear_distance = 0
    # for i in range(embs_adv.shape[0]):
    #     pear_distance += np.corrcoef(embs_adv_cent[i], embs_cent[i])[0, 1]

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
    return batch_signal, data_sizes, Epsilon_uni, classes

def load_model(model, dataset):
    if model == 'CNN1D':
        model_ft = CNN1D.ResNet1D(dataset).cuda()
    elif model == 'CNN2D':
        model_ft = CNN2D.CNN2D(dataset).cuda()
    elif model == 'LSTM':
        model_ft = lstm.lstm2(dataset).cuda()
    elif model == 'GRU':
        model_ft = gru.gru2(dataset).cuda()
    elif model == 'MCLDNN':
        model_ft = mcldnn.MCLDNN(dataset).cuda()
    elif model == 'Lenet':
        model_ft = LeNet.LeNet_or(dataset).cuda()
    elif model == 'Vgg16':
        model_ft = vgg16.VGG16_or(dataset).cuda()
    elif model == 'Alexnet':
        model_ft = Alexnet.AlexNet_or(dataset).cuda()
    elif model == 'r8conv1':
        model_ft = RRR.resnet(depth=8).cuda()
    elif model == 'VTCNN2':
        model_ft = VTCNN2.VTCNN2().cuda()
    else:
        print('Error! There is no model of your choice')
    print(model_ft)
    # 导入预训练模型
    model_ft.load_state_dict(
        torch.load("../raw_model_training/result/model/128_"+model+"_best_lr=0.0001.pth"))
    model_ft.eval()
    return model_ft

def arg_parse():
    parser = argparse.ArgumentParser(description='Signal_diagonal_matrix_CNN arguments.')
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--num_workers', dest='num_workers', type=int, help='Number of workers to load data.')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Dataset: 128, 512, 1024, 3040')
    parser.add_argument('--model', dest='model', type=str, help='Model: CNN1D, CNN2D, LSTM, GRU, MCLDNN, Lenet, Vgg16, Alexnet')
    parser.set_defaults(lr=0.0002, batch_size=10, num_epochs=10, num_workers=4, dataset='128', model='CNN2D')
    return parser.parse_args()


def main():
    prog_args = arg_parse()
    batch_signal, data_sizes, SigPow, classes = prepare_data(prog_args)  # 跳转到prepare_data函数，得到批训练集和批测试集
    model_ft = load_model(prog_args.model, prog_args.dataset)
    train_model(prog_args,  SigPow, batch_signal, classes, model_ft)

if __name__ == '__main__':
    main()
