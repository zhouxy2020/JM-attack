"""
CNN1D
"""
import argparse
import os
import numpy as np
import sys
import time
import csv
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from raw_model_training import  mcldnn,  RRR,VTCNN2
import torch.nn.functional as F
import pickle
# sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
global device
# device = 'cuda:%d' % '1'
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model(args, batch_signal, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=100):
    since = time.time()  # 开始训练的时间
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    traindata_list = list(enumerate(batch_signal['train']))
    testdata_list = list(enumerate(batch_signal['test']))
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            signal_num = 0
            # Iterate over data.
            pbar = tqdm(batch_signal[phase])
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs[-1], 1)
                    loss = criterion(outputs[-1], labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                signal_num += inputs.size(0)
            if phase == 'train':
                scheduler.step()
            # 显示该轮的loss和精度
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            # 保存当前的训练集精度、测试集精度和最高测试集精度
            with open("./result/result_{}_{}.csv".format(args.model), 'a', newline='') as t1:
                writer_train1 = csv.writer(t1)
                writer_train1.writerow([epoch, phase, epoch_loss, epoch_acc, best_acc])
        # 保存测试精度最高时的模型参数
        torch.save(best_model_wts, "./result/model/{}_best_lr={}.pth".format(args.model, args.lr))
        print('Best test Acc: {:4f}'.format(best_acc))
        print()
    print('Best test Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

def ModCls_loaddata():
    import numpy as np
    import pickle as cPickle
    import sys
    # sys.path.insert(0, '/home/meysam/Work/ModulationClassification/codes_letter')

    # There is a Pickle incompatibility of numpy arrays between Python 2 and 3
    # which generates ascii encoding error, to work around that we use the following instead of
    # Xd = cPickle.load(open("RML2016.10a_dict.dat",'rb'))
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

    # Partition the data
    #  into training and test sets of the form we can train/test on
    #  while keeping SNR and Mod labels handy for each
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

    in_shp = list(X_train.shape[1:])
    classes = mods
    return (X, lbl, X_train, X_test, classes, snrs, mods, Y_train, Y_test, train_idx, test_idx)

def prepare_data(args):
    X_loaded, lbl, X_train, X_test, classes, snrs, mods, Y_train, Y_test, train_idx, test_idx = ModCls_loaddata()
    train_2 = torch.from_numpy(X_train).type(torch.FloatTensor)  # [312000, 2, 128] 统一转为[N, Channel, Length]形式
    test_2 = torch.from_numpy(X_test).type(torch.FloatTensor)  # [156000, 2, 128]
    train_label = torch.from_numpy(Y_train).type(torch.LongTensor)
    train_label = torch.argmax(train_label, dim=1)
    test_label = torch.from_numpy(Y_test).type(torch.LongTensor)
    test_label = torch.argmax(test_label, dim=1)
    data_sizes = {'train': len(train_label), 'test': len(test_label)}
    #####################
    # 把数据放在数据库中
    train_signal = torch.utils.data.TensorDataset(train_2, train_label)
    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    # 将训练集和测试集分批
    batch_signal = {'train': torch.utils.data.DataLoader(dataset=train_signal, batch_size=args.batch_size,
                                                         shuffle=True, num_workers=args.num_workers),
                    'test': torch.utils.data.DataLoader(dataset=test_signal, batch_size=args.batch_size,
                                                        shuffle=False, num_workers=args.num_workers)}
    return batch_signal, data_sizes

def load_model(model):
    if model == 'Resnet8':
        model_ft = RRR.resnet(depth=8).cuda()
    elif model == 'VTCNN2':
        model_ft = VTCNN2.VTCNN2().cuda()
    else:
        print('Error! There is no model of your choice')
    print(model_ft)
    # 导入预训练模型
    # model_ft.load_state_dict(
    #     torch.load("../raw_model_training/result/model/"+model+"_best_lr=0.001.pth"))
    model_ft.train()
    return model_ft

def arg_parse():
    parser = argparse.ArgumentParser(description='Signal_diagonal_matrix_CNN arguments.')
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate.')
    parser.add_argument('--batch-size', dest='batch_size', default=64, type=int, help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--num_workers', dest='num_workers', type=int, help='Number of workers to load data.')
    parser.add_argument('--model', dest='model', type=str, help='Model: VTCNN2, Resnet8')
    parser.set_defaults(lr=0.0001, batch_size=128, num_epochs=200, num_workers=0, model='VTCNN2')
    return parser.parse_args()

def main():
    prog_args = arg_parse()
    batch_signal, data_sizes = prepare_data(prog_args)  # 跳转到prepare_data函数，得到批训练集和批测试集
    model_ft = load_model(prog_args.model)
    criterion = nn.CrossEntropyLoss()
    # 优化器
    # optimizer_ft = optim.SGD([{"params": model_ft.parameters()}, {"params": filter_all}, {"params": bias_all}],
    #                          lr=prog_args.lr, momentum=0.9)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=prog_args.lr)
    # 学习率衰减
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.8)
    # 训练模型
    train_model(prog_args, batch_signal, data_sizes, model_ft, criterion, optimizer_ft,
                exp_lr_scheduler, num_epochs=prog_args.num_epochs)

if __name__ == '__main__':
    main()
