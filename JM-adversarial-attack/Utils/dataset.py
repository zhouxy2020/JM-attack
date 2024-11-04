import numpy as np
import torch.utils.data as Data
import torch
from args import args

# help functions to get the training/validation/testing data loader

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

def get_signal_train_validate_loader(batch_size, shuffle=True, random_seed=100, num_workers=1):
    """

    :param dir_name:
    :param batch_size:
    :param valid_size:
    :param augment:
    :param shuffle:
    :param random_seed:
    :param num_workers:
    :return:
    """
    # train_2 = np.load('/home/zjut/public/data0/000_Dataset/001_Signal/dataset/radio{}NormTrainX.npy'.format(args.dataset))
    # train_label_path = '/home/zjut/public/data0/000_Dataset/001_Signal/dataset/radio{}NormTrainSnrY.npy'.format(args.dataset)  # 训练集标签
    # test_2 = np.load('/home/zjut/public/data0/000_Dataset/001_Signal/dataset/radio{}NormTestX.npy'.format(args.dataset))
    # test_label_path = '/home/zjut/public/data0/000_Dataset/001_Signal/dataset/radio{}NormTestSnrY.npy'.format(args.dataset)  # 测试集标签

    # if args.dataset == '3040':
    #     train_label = np.load(train_label_path)  # 得到0到11的类标签数据
    #     test_label = np.load(test_label_path)  # 得到0到11的类标签数据
    # else:
    #     train_label = np.load(train_label_path)[:, 0]  # 得到0到11的类标签数据
    #     test_label = np.load(test_label_path)[:, 0]  # 得到0到11的类标签数据

    # # 数组变张量
    # if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'Vgg16t' :
    #     print('Data will be reshaped into [N, 1, 2, Length]')
    #     train_2 = np.reshape(train_2, (train_2.shape[0], 1, train_2.shape[1], 2))
    #     test_2 = np.reshape(test_2, (test_2.shape[0], 1, test_2.shape[1], 2))
    #     # 数组变张量
    #     train_2 = torch.from_numpy(train_2).permute(0, 1, 3, 2)  # [312000, 1, 2, 128]
    #     train_2 = train_2.type(torch.FloatTensor)
    #     test_2 = torch.from_numpy(test_2).permute(0, 1, 3, 2)  # [156000, 1, 2, 128]
    #     test_2 = test_2.type(torch.FloatTensor)
    # else:
    #     train_2 = torch.from_numpy(train_2).permute(0, 2, 1)  # [312000, 2, 128] 统一转为[N, Channel, Length]形式
    #     train_2 = train_2.type(torch.FloatTensor)
    #     test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
    #     test_2 = test_2.type(torch.FloatTensor)
    # train_label = torch.from_numpy(train_label)
    # train_label = train_label.type(torch.LongTensor)
    # test_label = torch.from_numpy(test_label)
    # test_label = test_label.type(torch.LongTensor)
    # print(train_2.shape, train_label.shape, test_2.shape, test_label.shape)
    #############
    X_loaded, lbl, X_train, X_test, classes, snrs, mods, Y_train, Y_test, train_idx, test_idx = ModCls_loaddata()
    # SNR = 10
    # train_SNRs = list(map(lambda x: lbl[x][1], train_idx))
    # train_X_0 = X_train[np.where(np.array(train_SNRs) == SNR)].reshape(-1, 2, 128)
    # train_Y_0 = Y_train[np.where(np.array(train_SNRs) == SNR)]
    # TEST
    # test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    # test_X_0 = X_test[np.where(np.array(test_SNRs) == SNR)].reshape(-1, 2, 128)
    # test_Y_0 = Y_test[np.where(np.array(test_SNRs) == SNR)]
    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'Vgg16t':
        X_train = np.expand_dims(np.transpose(X_train,(0,2,1)),axis=1)
        X_test = np.expand_dims(np.transpose(X_test,(0,2,1)),axis=1)

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

    train_signal = torch.utils.data.TensorDataset(train_2, train_label)
    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    #########################
    # 把数据放在数据库中
    # train_signal = torch.utils.data.TensorDataset(train_2, train_label)
    # test_signal = torch.utils.data.TensorDataset(test_2, test_label)

    train_loader =torch.utils.data.DataLoader(dataset=train_signal, batch_size=batch_size,
                                                         shuffle=True, num_workers=num_workers, drop_last=True)
    validate_loader = torch.utils.data.DataLoader(dataset=train_signal, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers, drop_last=True)

    # load the dataset
    return train_loader, validate_loader

def get_signal_test_loader( batch_size, shuffle=False, num_worker=1):
    """

    :param dir_name:
    :param batch_size:
    :param shuffle:
    :param num_worker:
    :return:
    """

    # test_2 = np.load('/public/wzw/data/512_upper10db/radio512NormTestX.npy')
    # test_label_path = '/public/wzw/data/512_upper10db/radio512NormTestSnrY.npy'
    #
    # if args.dataset == '3040':
    #     test_label = np.load(test_label_path)  # 得到0到11的类标签数据[
    # else:
    #     test_label = np.load(test_label_path)[:, 0]  # 得到0到11的类标签数据
    #
    # if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet':
    #     print('Data will be reshaped into [N, 1, 2, Length]')
    #     test_2 = np.reshape(test_2, (test_2.shape[0], 1, test_2.shape[1], 2))
    #     # 数组变张量
    #     test_2 = torch.from_numpy(test_2).permute(0, 1, 3, 2)  # [156000, 1, 2, 128]
    #     test_2 = test_2.type(torch.FloatTensor)
    # else:
    #     test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
    #     test_2 = test_2.type(torch.FloatTensor)
    #
    # test_label = torch.from_numpy(test_label)
    # test_label = test_label.type(torch.LongTensor)
    # print(test_2.shape, test_label.shape)

    #############
    X_loaded, lbl, X_train, X_test, classes, snrs, mods, Y_train, Y_test, train_idx, test_idx = ModCls_loaddata()
    # SNR = 10
    # # TEST
    # test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    # test_X_0 = X_test[np.where(np.array(test_SNRs) == SNR)].reshape(-1, 2, 128)
    # test_Y_0 = Y_test[np.where(np.array(test_SNRs) == SNR)]
    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'Vgg16t':
        X_test = np.expand_dims(np.transpose(X_test,(0,2,1)),axis=1)

    test_2 = torch.from_numpy(X_test)
    test_2 = test_2.type(torch.FloatTensor)
    test_label = torch.from_numpy(Y_test)
    test_label = test_label.type(torch.LongTensor)
    #########################

    # test_I = np.load(test_path_I)  # [44000, 128]
    # test_Q = np.load(test_path_Q)  # [44000, 128]
    # test_2 = np.stack([test_I, test_Q], axis=-1)  # [44000, 128, 2]
    # test_label = np.load(test_label_path)[1]  # 得到0到11的类标签数据
    # test_2 = torch.from_numpy(test_2).permute(0, 2, 1) # [156000, 2, 128]数组转tensor，021转换索引
    # test_label = torch.from_numpy(test_label)
    # # print(test_2.dtype,"1111111111111111",test_label.dtype)
    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_signal, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_worker, drop_last=True)

    return test_loader


def get_alldb_signal_train_validate_loader(batch_size, shuffle=True, random_seed=100, num_workers=1):
    """

    :param dir_name:
    :param batch_size:
    :param valid_size:
    :param augment:
    :param shuffle:
    :param random_seed:
    :param num_workers:
    :return:
    """
    # train_2 = np.load('/public/wzw/Series_exp/filterData/radio11CNormTrainX.npy')
    # train_label_path = '/public/wzw/Series_exp/filterData/radio11CNormTrainSnrY.npy'  # 训练集标签
    # test_2 = np.load('/public/wzw/Series_exp/filterData/radio11CNormTestX.npy')
    # test_label_path = '/public/wzw/Series_exp/filterData/radio11CNormTestSnrY.npy'  # 测试集标签
    #
    # if args.dataset == '3040':
    #     train_label = np.load(train_label_path)  # 得到0到11的类标签数据
    #     test_label = np.load(test_label_path)  # 得到0到11的类标签数据
    # else:
    #     train_label = np.transpose(np.load(train_label_path),(1,0))[:, 1]
    #     test_label = np.transpose(np.load(test_label_path), (1, 0))[:, 1]
    #     # train_label = np.load(train_label_path)[:, 0]  # 得到0到11的类标签数据
    #     # test_label = np.load(test_label_path)[:, 0]  # 得到0到11的类标签数据
    #
    # # 数组变张量
    # if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'resnet8':
    #     print('Data will be reshaped into [N, 1, 2, Length]')
    #     train_2 = np.reshape(train_2, (train_2.shape[0], 1, train_2.shape[1], 2))
    #     test_2 = np.reshape(test_2, (test_2.shape[0], 1, test_2.shape[1], 2))
    #     # 数组变张量
    #     train_2 = torch.from_numpy(train_2).permute(0, 1, 3, 2)  # [312000, 1, 2, 128]
    #     train_2 = train_2.type(torch.FloatTensor)
    #     test_2 = torch.from_numpy(test_2).permute(0, 1, 3, 2)  # [156000, 1, 2, 128]
    #     test_2 = test_2.type(torch.FloatTensor)
    # else:
    #     train_2 = torch.from_numpy(train_2).permute(0, 2, 1)  # [312000, 2, 128] 统一转为[N, Channel, Length]形式
    #     train_2 = train_2.type(torch.FloatTensor)
    #     test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
    #     test_2 = test_2.type(torch.FloatTensor)
    # train_label = torch.from_numpy(train_label)
    # train_label = train_label.type(torch.LongTensor)
    # test_label = torch.from_numpy(test_label)
    # test_label = test_label.type(torch.LongTensor)
    # print(train_2.shape, train_label.shape, test_2.shape, test_label.shape)
    #
    ###########################
    X_loaded, lbl, X_train, X_test, classes, snrs, mods, Y_train, Y_test, train_idx, test_idx = ModCls_loaddata()
    # SNR = 10
    # train_SNRs = list(map(lambda x: lbl[x][1], train_idx))
    # train_X_0 = X_train[np.where(np.array(train_SNRs) == SNR)].reshape(-1, 2, 128)
    # train_Y_0 = Y_train[np.where(np.array(train_SNRs) == SNR)]
    # # TEST
    # test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    # test_X_0 = X_test[np.where(np.array(test_SNRs) == SNR)].reshape(-1, 2, 128)
    # test_Y_0 = Y_test[np.where(np.array(test_SNRs) == SNR)]
    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'Vgg16t':
        X_train = np.expand_dims(np.transpose(X_train,(0,2,1)),axis=1)
        X_test = np.expand_dims(np.transpose(X_test,(0,2,1)),axis=1)

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
    #########################

    # 把数据放在数据库中
    train_signal = torch.utils.data.TensorDataset(train_2, train_label)
    test_signal = torch.utils.data.TensorDataset(test_2, test_label)

    train_loader =torch.utils.data.DataLoader(dataset=train_signal, batch_size=batch_size,
                                                         shuffle=True, num_workers=num_workers, drop_last=True)
    validate_loader = torch.utils.data.DataLoader(dataset=test_signal, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers, drop_last=True)

    # load the dataset
    return train_loader, validate_loader

def get_alldb_signal_test_loader( batch_size, shuffle=False, num_worker=1):
    """

    :param dir_name:
    :param batch_size:
    :param shuffle:
    :param num_worker:
    :return:
    """
    #
    # test_2 = np.load('/public/wzw/Series_exp/filterData/128_High_radio11CNormTestX.npy')
    # test_2 = np.load('/public/wzw/Series_exp/filterData/128_High_radio11CNormTestX.npy')
    # test_2 = np.load('/public/wzw/Series_exp/filterData/radio11CNormTestX.npy')
    # test_label_path = '/public/wzw/Series_exp/filterData/radio11CNormTestSnrY.npy'  # 训练集标签
    #
    # if args.dataset == '3040':
    #     test_label = np.load(test_label_path)  # 得到0到11的类标签数据
    # else:
    #     test_label = np.transpose(np.load(test_label_path), (1,0))[:, 1]
    #      # 得到0到11的类标签数据
    #
    # if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'resnet8':
    #     print('Data will be reshaped into [N, 1, 2, Length]')
    #     test_2 = np.reshape(test_2, (test_2.shape[0], 1, test_2.shape[1], 2))
    #     # 数组变张量
    #     test_2 = torch.from_numpy(test_2).permute(0, 1, 3, 2)  # [156000, 1, 2, 128]
    #     test_2 = test_2.type(torch.FloatTensor)
    # else:
    #     test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
    #     test_2 = test_2.type(torch.FloatTensor)
    #
    # test_label = torch.from_numpy(test_label)
    # test_label = test_label.type(torch.LongTensor)
    # print(test_2.shape, test_label.shape)

    ###########################
    X_loaded, lbl, X_train, X_test, classes, snrs, mods, Y_train, Y_test, train_idx, test_idx = ModCls_loaddata()
    SNR = 10
    # TEST
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_0 = X_test[np.where(np.array(test_SNRs) == SNR)].reshape(-1, 2, 128)
    test_Y_0 = Y_test[np.where(np.array(test_SNRs) == SNR)]
    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'Vgg16t':
        test_X_0 = np.expand_dims(np.transpose(test_X_0,(0,2,1)),axis=1)

    test_2 = torch.from_numpy(test_X_0)
    test_2 = test_2.type(torch.FloatTensor)
    test_label = torch.from_numpy(test_Y_0)
    test_label = test_label.type(torch.LongTensor)
    test_label = torch.argmax(test_label, dim=1)
    #########################

    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_signal, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_worker, drop_last=True)

    return test_loader

def ttttt_test_loader( batch_size, shuffle=False, num_worker=1):
    """

    :param dir_name:
    :param batch_size:
    :param shuffle:
    :param num_worker:
    :return:
    """
    #
    test_2 = np.load('/home/zjut/public/signal/wzw/SignalAttack/KD/CleanDatasets/CNN1D/128/128_inputs.npy')
    test_label_path = '/home/zjut/public/signal/wzw/SignalAttack/KD/CleanDatasets/CNN1D/128/128_labels.npy'  # 训练集标签
    # test_2 = np.load('/home/wzw/data/128a_singledb_nor/{}db_NormTestX.npy'.format(args.db))
    # test_label_path = '/home/wzw/data/128a_singledb_nor/{}db_NormTestSnrY.npy'.format(args.db)  # 训练集标签

    if args.dataset == '3040':
        test_label = np.load(test_label_path)  # 得到0到11的类标签数据
    else:
        test_label = np.transpose(np.load(test_label_path), (1,0))[:, 1]
         # 得到0到11的类标签数据

    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'resnet8':
        print('Data will be reshaped into [N, 1, 2, Length]')
        test_2 = np.reshape(test_2, (test_2.shape[0], 1, test_2.shape[1], 2))
        # 数组变张量
        test_2 = torch.from_numpy(test_2).permute(0, 1, 3, 2)  # [156000, 1, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)
    else:
        test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)

    test_label = torch.from_numpy(test_label)
    test_label = test_label.type(torch.LongTensor)
    print(test_2.shape, test_label.shape)

    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_signal, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_worker, drop_last=True)

    return test_loader

def get_single_db_signal_test_loader( batch_size, shuffle=False, num_worker=1):
    """

    :param dir_name:
    :param batch_size:
    :param shuffle:
    :param num_worker:
    :return:
    """
    #
    # test_2 = np.load('/home/wzw/data/128a-all-nor/radio11CNormTestX.npy')
    # test_label_path = '/home/wzw/data/128a-all-nor/radio11CNormTestSnrY.npy'  # 训练集标签

    test_2 = np.load('/public/wzw/data/128a_singledb_nor/{}db_NormTestX.npy'.format(args.db))
    test_label_path = '/public/wzw/data/128a_singledb_nor/{}db_NormTestSnrY.npy'.format(args.db)  # 训练集标签

    print('chosen data db is {}'.format(args.db))

    if args.dataset == '3040':
        test_label = np.load(test_label_path)  # 得到0到11的类标签数据
    else:
        test_label = np.transpose(np.load(test_label_path), (1,0))[:, 1]
         # 得到0到11的类标签数据

    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'mobilenet' or args.model == 'resnet8':
        print('Data will be reshaped into [N, 1, 2, Length]')
        test_2 = np.reshape(test_2, (test_2.shape[0], 1, test_2.shape[1], 2))
        # 数组变张量
        test_2 = torch.from_numpy(test_2).permute(0, 1, 3, 2)  # [156000, 1, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)
    else:
        test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)

    test_label = torch.from_numpy(test_label)
    test_label = test_label.type(torch.LongTensor)
    print(test_2.shape, test_label.shape)

    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_signal, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_worker, drop_last=False)

    return test_loader

def get_upper_minus4db_signal_test_loader( batch_size, shuffle=False, num_worker=1):
    """

    :param dir_name:
    :param batch_size:
    :param shuffle:
    :param num_worker:
    :return:
    """
    #
    # test_2 = np.load('/home/wzw/data/128a-all-nor/radio11CNormTestX.npy')
    # test_label_path = '/home/wzw/data/128a-all-nor/radio11CNormTestSnrY.npy'  # 训练集标签

    test_2 = np.load('/public/wzw/data/128upper-4db/TestX.npy')
    test_label_path = '/public/wzw/data/128upper-4db/TestY.npy' # 训练集标签
    print('Using upper minus 4 db signal dataset to test\n')

    if args.dataset == '3040':
        test_label = np.load(test_label_path)  # 得到0到11的类标签数据
    else:
        test_label = np.transpose(np.load(test_label_path), (1,0))[:, 1]
         # 得到0到11的类标签数据

    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'mobilenet' or args.model == 'resnet8':
        print('Data will be reshaped into [N, 1, 2, Length]')
        test_2 = np.reshape(test_2, (test_2.shape[0], 1, test_2.shape[1], 2))
        # 数组变张量
        test_2 = torch.from_numpy(test_2).permute(0, 1, 3, 2)  # [156000, 1, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)
    else:
        test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)

    test_label = torch.from_numpy(test_label)
    test_label = test_label.type(torch.LongTensor)
    print(test_2.shape, test_label.shape)

    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_signal, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_worker, drop_last=False)

    return test_loader