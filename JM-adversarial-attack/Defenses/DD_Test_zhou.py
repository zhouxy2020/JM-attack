import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
# from RawModels.Utils.dataset import get_mnist_train_validate_loader, get_mnist_test_loader
# from RawModels.Utils.dataset import get_cifar10_train_validate_loader, get_cifar10_test_loader
from Utils.dataset import get_signal_train_validate_loader, get_signal_test_loader, get_alldb_signal_train_validate_loader, ModCls_loaddata
from models.network import define_model
from args import args
import pickle
from Defenses.DefenseMethods.DD import DistillationDefense


def main():
    # Device configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the random seed manually for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Get training parameters, set up model frames and then get the train_loader and test_loader
    dataset = args.dataset.upper()
    if dataset == '128' or '512' or '1024' or '3040':
        model_framework = define_model(name=args.model).to(device)
        model_location = '../DefenseEnhancedModels/DD/128_CNN2D_temp30.0/model_best.pth_epoch133.tar'
        model_framework.load_state_dict(
            torch.load(model_location, map_location=f"cuda:{args.gpu_index}")["net"])  # ["net"]
        # model_location = '../raw_model_training/result/model/128_CNN1D_best_lr=0.001.pth'
        # model_framework.load_state_dict(torch.load(model_location, map_location=f"cuda:{args.gpu_index}"))  # ["net"]

    else:
        print("data error")
    with open('/home/Automatic-Modulation-Classification/Adversarial-Attack-on-Deep-Learning-Based-Radio-Signal-Classification/Security-and-Robustness-of-Deep-Learning-in-Wireless-Communication-Systems-master/Adv_Attack_Modulation_Classification/zhou_20230324/GAP_20230809/PearACos-UAP_bs5_nt25_N5_ec200_SNR10_PSR-10.pkl',
            'rb') as f:
        UAP_pearacos_all = pickle.load(f)
    with open('/home/Automatic-Modulation-Classification/Adversarial-Attack-on-Deep-Learning-Based-Radio-Signal-Classification/Security-and-Robustness-of-Deep-Learning-in-Wireless-Communication-Systems-master/Adv_Attack_Modulation_Classification/zhou_20230324/GAP_20230809/FG-UAP_bs5_nt25_N5_ec200_SNR10_PSR-10.pkl', 'rb') as f:
        UAP_fg_all = pickle.load(f)
    with open('/home/Automatic-Modulation-Classification/Adversarial-Attack-on-Deep-Learning-Based-Radio-Signal-Classification/Security-and-Robustness-of-Deep-Learning-in-Wireless-Communication-Systems-master/Adv_Attack_Modulation_Classification/zhou_20230324/GAP_20230708/PCAUAP0714_Pca_Deepfool_Jamming.pkl', 'rb') as f:
        temp = pickle.load(f)
        UAP_deepfool_all = temp['UAP_deepfool_all']
        UAP_pca_all = temp['UAP_pca_all']
        jamming_all = temp['jamming_all']

    PSR_vec = np.arange(-20, -9, 1)
    acc_pearacos = np.zeros([UAP_pearacos_all.shape[0], len(PSR_vec)])
    acc_fg = np.zeros([UAP_pearacos_all.shape[0], len(PSR_vec)])
    acc_deepfool = np.zeros([UAP_pearacos_all.shape[0], len(PSR_vec)])
    acc_pca = np.zeros([UAP_pearacos_all.shape[0], len(PSR_vec)])
    acc_jamming = np.zeros([UAP_pearacos_all.shape[0], len(PSR_vec)])

    SNR = 10
    # Here we load the DATA for modulation classification
    X_loaded, lbl, X_train, X_test, classes, snrs, mods, Y_train, Y_test, train_idx, test_idx = ModCls_loaddata()
    train_SNRs = list(map(lambda x: lbl[x][1], train_idx))
    train_X_0 = X_train[np.where(np.array(train_SNRs) == SNR)].reshape(-1, 2, 128)
    train_Y_0 = Y_train[np.where(np.array(train_SNRs) == SNR)]
    # TEST
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_0 = X_test[np.where(np.array(test_SNRs) == SNR)].reshape(-1, 2, 128)
    test_Y_0 = Y_test[np.where(np.array(test_SNRs) == SNR)]


    SigPow = (np.sum(np.linalg.norm(train_X_0.reshape([-1, 256]), axis=1)) / train_X_0.shape[
        0]) ** 2  # I used to use 0.01 since np.linalg.norm(test_X_0[i]) = 0.09339~ 0.1

    defense_name = 'DD'
    model_framework.eval()
    batchsize = 128
    with torch.no_grad():
        for icn in range(len(PSR_vec)):
            PSR = PSR_vec[icn]
            # ==============================================================================
            PNR = PSR + SNR
            print('PSR = ', PNR - SNR)
            aid = 10 ** ((PNR - SNR) / 10)
            Epsilon_uni = np.sqrt(SigPow * aid)
            print('Epsilon_uni', Epsilon_uni)
            for rnd in range(UAP_pearacos_all.shape[0]):
                UAP_pearacos = UAP_pearacos_all[rnd: rnd + 1]
                UAP_fg = UAP_fg_all[rnd: rnd + 1]
                UAP_deepfool = UAP_deepfool_all[rnd: rnd + 1]
                UAP_pca = UAP_pca_all[rnd: rnd + 1]
                jamming = jamming_all[rnd: rnd + 1]

                UAP_pearacos = Epsilon_uni * (1 / np.linalg.norm(UAP_pearacos)) * UAP_pearacos
                UAP_fg = Epsilon_uni * (1 / np.linalg.norm(UAP_fg)) * UAP_fg
                UAP_deepfool = Epsilon_uni * (1 / np.linalg.norm(UAP_deepfool)) * UAP_deepfool
                UAP_pca = Epsilon_uni * (1 / np.linalg.norm(UAP_pca)) * UAP_pca
                jamming = Epsilon_uni * (1 / np.linalg.norm(jamming)) * jamming
                UAP_pearacos = UAP_pearacos.reshape(-1, 2, 128).repeat(batchsize, axis=0)
                UAP_fg = UAP_fg.reshape(-1, 2, 128).repeat(batchsize, axis=0)
                UAP_deepfool = UAP_deepfool.reshape(-1, 2, 128).repeat(batchsize, axis=0)
                UAP_pca = UAP_pca.reshape(-1, 2, 128).repeat(batchsize, axis=0)
                jamming = jamming.reshape(-1, 2, 128).repeat(batchsize, axis=0)

                sum_accuracy_deepfool, sum_accuracy_pca, sum_accuracy_jamming = 0, 0, 0
                sum_accuracy_pearacos, sum_accuracy_fg = 0, 0
                start = 0

                for batch_counter in range(test_X_0.shape[0] // batchsize):
                    x_batch_test = test_X_0[
                                   start:start + batchsize]  # Remember to put each minibatch within the minibatch counter
                    y_batch_test = test_Y_0[start:start + batchsize]
                    start = start + batchsize
                    test_label = torch.from_numpy(y_batch_test)
                    test_label = test_label.type(torch.LongTensor)
                    test_label = torch.argmax(test_label, dim=1).cuda()

                    test_jamming = torch.from_numpy(x_batch_test+jamming)
                    test_jamming = test_jamming.type(torch.FloatTensor).cuda()
                    outputs_jamming = model_framework(test_jamming)
                    _, predicted_jamming = torch.max(outputs_jamming.data, 1)

                    test_pearacos = torch.from_numpy(x_batch_test+UAP_pearacos)
                    test_pearacos = test_pearacos.type(torch.FloatTensor).cuda()
                    outputs_pearacos = model_framework(test_pearacos)
                    _, predicted_pearacos = torch.max(outputs_pearacos.data, 1)

                    test_fg = torch.from_numpy(x_batch_test+UAP_fg)
                    test_fg = test_fg.type(torch.FloatTensor).cuda()
                    outputs_fg = model_framework(test_fg)
                    _, predicted_fg = torch.max(outputs_fg.data, 1)

                    test_deepfool = torch.from_numpy(x_batch_test+UAP_deepfool)
                    test_deepfool = test_deepfool.type(torch.FloatTensor).cuda()
                    outputs_deepfool = model_framework(test_deepfool)
                    _, predicted_deepfool = torch.max(outputs_deepfool.data, 1)

                    test_pca = torch.from_numpy(x_batch_test+UAP_pca)
                    test_pca = test_pca.type(torch.FloatTensor).cuda()
                    outputs_pca = model_framework(test_pca)
                    _, predicted_pca = torch.max(outputs_pca.data, 1)

                    sum_accuracy_pearacos = sum_accuracy_pearacos + (predicted_pearacos == test_label).sum().item()
                    sum_accuracy_fg = sum_accuracy_fg + (predicted_fg == test_label).sum().item()
                    sum_accuracy_deepfool = sum_accuracy_deepfool + (predicted_deepfool == test_label).sum().item()
                    sum_accuracy_pca = sum_accuracy_pca + (predicted_pca == test_label).sum().item()
                    sum_accuracy_jamming = sum_accuracy_jamming + (predicted_jamming == test_label).sum().item()
                acc_pearacos[rnd, icn] = sum_accuracy_pearacos / start
                acc_fg[rnd, icn] = sum_accuracy_fg / start
                acc_deepfool[rnd, icn] = sum_accuracy_deepfool / start
                acc_pca[rnd, icn] = sum_accuracy_pca / start
                acc_jamming[rnd, icn] = sum_accuracy_jamming / start

        sum_accuracy_clean = 0
        start = 0
        for batch_counter in range(test_X_0.shape[0] // batchsize):
            x_batch_test = test_X_0[
                           start:start + batchsize]  # Remember to put each minibatch within the minibatch counter
            y_batch_test = test_Y_0[start:start + batchsize]
            start = start + batchsize
            test_label = torch.from_numpy(y_batch_test)
            test_label = test_label.type(torch.LongTensor)
            test_label = torch.argmax(test_label, dim=1).cuda()

            test_clean = torch.from_numpy(x_batch_test)
            test_clean = test_clean.type(torch.FloatTensor).cuda()
            outputs_clean = model_framework(test_clean)
            _, predicted_clean = torch.max(outputs_clean.data, 1)
            sum_accuracy_clean = sum_accuracy_clean + (predicted_clean == test_label).sum().item()
        acc_clean = sum_accuracy_clean / start

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(PSR_vec, 100 * acc_clean * np.ones(11), 'k>-', label='No attack')
    ax.plot(PSR_vec, 100 * np.mean(acc_jamming, axis=0), color='purple', marker='h', linestyle='-',
            label='Jamming attack')
    ax.plot(PSR_vec, 100 * np.mean(acc_deepfool, axis=0), 'cs-', label='Deepfool-UAP attack')
    ax.plot(PSR_vec, 100 * np.mean(acc_pca, axis=0), 'ro-', label='PCA-UAP attack')
    # ax.plot(PSR_vec, 100 * np.mean(acc_acos,axis=0), color='darkgreen',marker='+',linestyle='-', label='ACos-UAP attack - SNR=10 dB')
    ax.plot(PSR_vec, 100 * np.mean(acc_fg, axis=0), 'y*-', label='FG-UAP attack')
    ax.plot(PSR_vec, 100 * np.mean(acc_pearacos, axis=0), 'b^-', label='JM-UAP attack')
    # ax.plot(PSR_vec,100 * np.mean(acc_pearacos,axis=0), color='darkorange',marker='1',linestyle='-', label='PearACos-UAP attack - SNR=10 dB')
    # ax.plot(PSR_vec, 100 * np.mean(acc_all,axis=0), 'g^-',label='ALL-UAP attack - SNR=10 dB')
    # ax.plot(PSR_vec, 100 * np.mean(acc_pearcos,axis=0), 'md-', label='PearCos-UAP attack - SNR=10 dB')
    plt.legend(loc='lower left')
    y_major = plt.MultipleLocator(10)
    ax.yaxis.set_major_locator(y_major)
    ax.set_xlabel('PSR [dB]', fontdict={'size': 10})
    ax.set_ylabel('Accuracy %', fontdict={'size': 10})
    plt.xticks(size=10)
    plt.yticks(size=10)
    ax.grid(True)
    plt.savefig("white_attack_acc.png", bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':

    main()
