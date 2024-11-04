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
from raw_model_training import CNN1D, RRR, CNN2D, lstm, gru, mcldnn, Alexnet, LeNet, vgg16
from torch.autograd import Variable
import torch.nn.functional as F
import pickle


def train_model(args, batch_signal, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()  # 开始训练的时间
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['test']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            signal_num = 0
            # Iterate over data.
            pbar = tqdm(batch_signal[phase])
            for inputs, labels in pbar:
                inputs = inputs.cuda()  # (batch_size, 2, 128)
                labels = labels.cuda()  # (batch_size, )
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)[-1]
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                signal_num += inputs.size(0)
                # 在进度条的右边实时显示数据集类型、loss值和精度
                epoch_loss = running_loss / signal_num
                epoch_acc = running_corrects.double() / signal_num
                pbar.set_postfix({'Set': '{}'.format(phase),
                                  'Loss': '{:.4f}'.format(epoch_loss),
                                  'Acc': '{:.4f}'.format(epoch_acc)})
                # print('\r{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), end=' ')
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
            # with open("../result/final_result.csv", 'a', newline='') as t:
            #     writer_train = csv.writer(t)
            #     writer_train.writerow(['dataset={}, model={}, num_epoch={}, lr={}, batch_size={}'.format(
            #         args.dataset, args.model, args.num_epochs, args.lr, args.batch_size)])
            #     writer_train.writerow(['epoch', 'phase', 'epoch_loss', 'epoch_acc', 'best_acc'])
            #     writer_train.writerow([epoch, phase, epoch_loss, epoch_acc, best_acc])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
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
    # 导入数据集
    X_loaded, lbl, X_train, X_test, classes, snrs, mods, Y_train, Y_test, train_idx, test_idx = ModCls_loaddata()
    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'Vgg16t':
        X_train = np.expand_dims(np.transpose(X_train,(0,2,1)),axis=1)
        X_test = np.expand_dims(np.transpose(X_test,(0,2,1)),axis=1)

    SNR = 10
    # TRAINING
    train_SNRs = list(map(lambda x: lbl[x][1], train_idx))
    X_train = X_train[np.where(np.array(train_SNRs) == SNR)]#.reshape(-1, 2, 128, 1)
    Y_train = Y_train[np.where(np.array(train_SNRs) == SNR)]
    # TEST
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    X_test = X_test[np.where(np.array(test_SNRs) == SNR)]#.reshape(-1, 2, 128, 1)
    Y_test = Y_test[np.where(np.array(test_SNRs) == SNR)]
    SigPow = (np.sum(np.linalg.norm(X_test.reshape([-1, 256]), axis=1)) / X_test.shape[
        0]) ** 2  # I used to use 0.01 since np.linalg.norm(test_X_0[i]) = 0.09339~ 0.1
    PNR = 0
    aid = 10 ** ((PNR - SNR) / 10)
    Epsilon_uni = np.sqrt(SigPow * aid)

    train_2 = torch.from_numpy(X_train)
    train_2 = train_2.type(torch.FloatTensor)
    test_2 = torch.from_numpy(X_test)
    test_2 = test_2.type(torch.FloatTensor)
    train_label = torch.from_numpy(Y_train)
    train_label = train_label.type(torch.LongTensor)
    train_label = torch.argmax(train_label, dim=1)
    test_label = torch.from_numpy(Y_test)
    test_label = test_label.type(torch.LongTensor)
    test_label = torch.argmax(test_label, dim=1)
    data_sizes = {'train': len(train_label), 'test': len(test_label)}

    # 把数据放在数据库中
    train_signal = torch.utils.data.TensorDataset(train_2, train_label)
    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    # 将训练集和测试集分批
    batch_signal = {'train': torch.utils.data.DataLoader(dataset=train_signal, batch_size=args.batch_size,
                                                         shuffle=True, num_workers=args.num_workers),
                    'test': torch.utils.data.DataLoader(dataset=test_signal, batch_size=args.batch_size,
                                                        shuffle=False, num_workers=args.num_workers)}
    return batch_signal, data_sizes


def arg_parse():
    parser = argparse.ArgumentParser(description='Signal_diagonal_matrix_CNN arguments.')
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--num_workers', dest='num_workers', type=int, help='Number of workers to load data.')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Dataset: 128, 512, 1024, 3040')
    parser.add_argument('--model', dest='model', type=str, help='Model: CNN1D, CNN2D, LSTM, GRU, MCLDNN, Lenet, Vgg16, Alexnet')
    parser.set_defaults(lr=0.001, batch_size=128, num_epochs=1, num_workers=8, dataset='128', model='r8conv1')
    return parser.parse_args()


def main():
    prog_args = arg_parse()

    batch_signal, data_sizes = prepare_data(prog_args)  # 跳转到prepare_data函数，得到批训练集和批测试集
    # 模型放到GPU上
    if prog_args.model == 'CNN1D':
        model_ft = CNN1D.ResNet1D(prog_args.dataset).cuda()
    elif prog_args.model == 'CNN2D':
        model_ft = CNN2D.CNN2D(prog_args.dataset).cuda()
    elif prog_args.model == 'LSTM':
        model_ft = lstm.lstm2(prog_args.dataset).cuda()
    elif prog_args.model == 'GRU':
        model_ft = gru.gru2(prog_args.dataset).cuda()
    elif prog_args.model == 'MCLDNN':
        model_ft = mcldnn.MCLDNN(prog_args.dataset).cuda()
    elif prog_args.model == 'Lenet':
        model_ft = LeNet.LeNet_or(prog_args.dataset).cuda()
    elif prog_args.model == 'Vgg16':
        model_ft = vgg16.VGG16_or(prog_args.dataset).cuda()
    elif prog_args.model == 'Alexnet':
        model_ft = Alexnet.AlexNet_or(prog_args.dataset).cuda()
    elif prog_args.model == 'r8conv1':
        model_ft = RRR.resnet(depth=8).cuda()
    else:
        print('Error! There is no model of your choice')
    print(model_ft)
    # 导入预训练模型
    model_ft.load_state_dict(torch.load("./result/model/{}_{}_best_lr={}.pth".format(prog_args.dataset, prog_args.model, prog_args.lr)))
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    # optimizer_ft = optim.SGD([{"params": model_ft.parameters()}, {"params": filter_all}, {"params": bias_all}],
    #                          lr=prog_args.lr, momentum=0.9)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=prog_args.lr)
    # 学习率衰减
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.8)
    # 训练模型
    train_model(prog_args, batch_signal, data_sizes, model_ft, criterion, optimizer_ft,
                exp_lr_scheduler, num_epochs=prog_args.num_epochs)


if __name__ == '__main__':
    main()
