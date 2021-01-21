import os
import torch
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
import sys
from math import sqrt
import torch.utils.data
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from utils.l0dense import L0Dense
from utils.encoder import encoder
from utils.aggregator import aggregator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

class MGPred(nn.Module):
    def __init__(self, u_one_embedding, u_two_embedding, u_three_embedding, i_one_embedding, i_two_embedding, i_three_embedding, embed_dim, N = 30000, droprate = 0.5, beta_ema = 0.999):
        super(MGPred, self).__init__()

        self.u_one_embed = u_one_embedding
        self.u_two_embed = u_two_embedding
        self.u_three_embed = u_three_embedding
        self.i_one_embed = i_one_embedding
        self.i_two_embed = i_two_embedding
        self.i_three_embed = i_three_embedding

        self.embed_dim = embed_dim
        self.N = N
        self.droprate = droprate
        self.beta_ema = beta_ema

        self.one_layer1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.one_layer2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.two_layer1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.two_layer2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.three_layer1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.three_layer2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.con_layer1 = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.con_layer2 = nn.Linear(self.embed_dim, 1)

        self.one_ui = nn.BatchNorm1d(self.embed_dim, momentum = 0.5)
        self.two_ui = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.three_ui = nn.BatchNorm1d(self.embed_dim, momentum = 0.5)
        self.con = nn.BatchNorm1d(self.embed_dim, momentum = 0.5)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense):
                self.layers.append(m)

        if beta_ema > 0.:
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_i):
        u_one_embed = self.u_one_embed(nodes_u, nodes_i)
        u_two_embed = self.u_two_embed(nodes_u, nodes_i)
        u_three_embed = self.u_three_embed(nodes_u, nodes_i)
        i_one_embed = self.i_one_embed(nodes_u, nodes_i)
        i_two_embed = self.i_two_embed(nodes_u, nodes_i)
        i_three_embed = self.i_three_embed(nodes_u, nodes_i)

        ui_one = torch.cat((u_one_embed, i_one_embed), dim=1)

        ui_two = torch.cat((u_two_embed, i_two_embed), dim=1)

        ui_three = torch.cat((u_three_embed, i_three_embed), dim=1)


        ui_one = F.relu(self.one_ui(self.one_layer1(ui_one)), inplace=True)
        ui_one = F.dropout(ui_one, training=self.training, p=self.droprate)
        ui_one = self.one_layer2(ui_one)

        ui_two = F.relu(self.two_ui(self.two_layer1(ui_two)), inplace=True)
        ui_two = F.dropout(ui_two, training=self.training, p=self.droprate)
        ui_two = self.two_layer2(ui_two)

        ui_three = F.relu(self.three_ui(self.three_layer1(ui_three)), inplace=True)
        ui_three = F.dropout(ui_three, training=self.training, p=self.droprate)
        ui_three = self.three_layer2(ui_three)

        x_ui = torch.cat((ui_one, ui_two, ui_three), dim=1)

        x = F.relu(self.con(self.con_layer1(x_ui)), inplace = True)
        x = F.dropout(x, training = self.training, p = self.droprate)
        scores = self.con_layer2(x)
        return scores.squeeze()

    def regularization(self):
        regularization = 0
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        return regularization

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def loss(self, nodes_u, nodes_i, ratings):
        scores = self.forward(nodes_u, nodes_i)
        loss = self.criterion(scores, ratings)

        total_loss = loss + self.regularization()
        return total_loss


def read_raw_data(rawdata_dir):
    gii = open(rawdata_dir + '/' + 'Text_similarity_five.pkl', 'rb')
    drug_Tfeature_five = pickle.load(gii)
    gii.close()


    gii = open(rawdata_dir + '/' + 'effect_side_semantic.pkl', 'rb')
    effect_side_semantic = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'Drug_word2vec.pkl', 'rb')
    Drug_word2vec = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'glove_wordEmbedding.pkl', 'rb')
    glove_word = pickle.load(gii)
    gii.close()


    return drug_Tfeature_five, Drug_word2vec, effect_side_semantic, glove_word


def fold_files(data_train, data_test, data_neg, args):
    rawdata_dir = args.rawpath

    drug_Tfeature_five, drug_feature_fingerprint, effect_side_semantic, glove_word = read_raw_data(rawdata_dir)

    i_train, u_train, r_train = [], [], []
    for i in range(len(data_train)):
        i_train.append(int(data_train[i][0]))
        u_train.append(int(data_train[i][1]))
        r_train.append(int(float(data_train[i][2])))

    i_test, u_test, r_test = [], [], []
    for i in range(len(data_test)):
        i_test.append(int(data_test[i][0]))
        u_test.append(int(data_test[i][1]))
        r_test.append(int(float(data_test[i][2])))

    i_neg, u_neg, r_neg = [], [], []
    for i in range(len(data_neg)):
        i_neg.append(int(data_neg[i][0]))
        u_neg.append(int(data_neg[i][1]))
        r_neg.append(int(float(data_neg[i][2])))

    u_adj = {}
    i_adj = {}
    u_train = np.array(u_train)
    for i in range(len(u_train)):
        if u_train[i] not in u_adj.keys():
            u_adj[u_train[i]] = []
        if i_train[i] not in i_adj.keys():
            i_adj[i_train[i]] = []
        if r_train[i] != 0:
            u_adj[u_train[i]].extend([(i_train[i], r_train[i])])
            i_adj[i_train[i]].extend([(u_train[i], r_train[i])])



    n_users = 750
    n_items = 994
    ufeature1 = {}
    ufeature2 = {}
    ufeature3 = {}
    for i in range(n_users):
        ufeature1[i] = [0 for _ in range(n_items)]
        ufeature2[i] = [0 for _ in range(n_users)]
        ufeature3[i] = [0 for _ in range(n_users)]

    ifeature1 = {}
    ifeature2 = {}
    ifeature3 = {}
    for i in range(n_items):
        ifeature1[i] = [0 for _ in range(n_users)]
        ifeature2[i] = [0 for _ in range(n_items)]
        ifeature3[i] = [0 for _ in range(n_items)]

    for key in u_adj.keys():
        n = u_adj[key].__len__()
        for i in range(n):
            ufeature1[key][u_adj[key][i][0]] = u_adj[key][i][1]

    for key in u_adj.keys():
        ufeature2[key] = drug_Tfeature_five[key].tolist()
        ufeature3[key] = drug_feature_fingerprint[key].tolist()

    for key in i_adj.keys():
        n = i_adj[key].__len__()
        for i in range(n):
            ifeature1[key][i_adj[key][i][0]] = i_adj[key][i][1]

    for key in i_adj.keys():
        ifeature2[key] = effect_side_semantic[key].tolist()
        ifeature3[key] = glove_word[key].tolist()

    ufeature_size1 = ufeature1[0].__len__()  # 第0个key所对应的列表的长度 （这里每一个key对应的长度都一样）
    ifeature_size1 = ifeature1[0].__len__()
    ufeature_size2 = ufeature2[0].__len__()  # 第0个key所对应的列表的长度 （这里每一个key对应的长度都一样）
    ifeature_size2 = ifeature2[0].__len__()
    ufeature_size3 = ufeature3[0].__len__()  # 第0个key所对应的列表的长度 （这里每一个key对应的长度都一样）
    ifeature_size3 = ifeature3[0].__len__()

    ufea1 = []
    for key in ufeature1.keys():
        ufea1.append(ufeature1[key])
    ufea1 = torch.Tensor(np.array(ufea1, dtype=np.float32))
    u2e1 = nn.Embedding(n_users, ufeature_size1)
    u2e1.weight = torch.nn.Parameter(ufea1)

    ifea1 = []
    for key in ifeature1.keys():
        ifea1.append(ifeature1[key])
    ifea1 = torch.Tensor(np.array(ifea1, dtype=np.float32))
    i2e1 = nn.Embedding(n_items, ifeature_size1)
    i2e1.weight = torch.nn.Parameter(ifea1)

    ufea2 = []
    for key in ufeature2.keys():
        ufea2.append(ufeature2[key])
    ufea2 = torch.Tensor(np.array(ufea2, dtype=np.float32))
    u2e2 = nn.Embedding(n_users, ufeature_size2)
    u2e2.weight = torch.nn.Parameter(ufea2)

    ifea2 = []
    for key in ifeature2.keys():
        ifea2.append(ifeature2[key])
    ifea2 = torch.Tensor(np.array(ifea2, dtype=np.float32))
    i2e2 = nn.Embedding(n_items, ifeature_size2)
    i2e2.weight = torch.nn.Parameter(ifea2)

    ufea3 = []
    for key in ufeature3.keys():
        ufea3.append(ufeature3[key])
    ufea3 = torch.Tensor(np.array(ufea3, dtype=np.float32))
    u2e3 = nn.Embedding(n_users, ufeature_size3)
    u2e3.weight = torch.nn.Parameter(ufea3)

    ifea3 = []
    for key in ifeature3.keys():
        ifea3.append(ifeature3[key])
    ifea3 = torch.Tensor(np.array(ifea3, dtype=np.float32))
    i2e3 = nn.Embedding(n_items, ifeature_size3)
    i2e3.weight = torch.nn.Parameter(ifea3)

    return u2e1, i2e1, u2e2, i2e2, u2e3, i2e3, u_train, i_train, r_train, u_test, i_test, i_neg, u_neg, r_neg, r_test, u_adj, i_adj


def train_test(data_train, data_test, data_neg, fold, args):
    u2e1, i2e1, u2e2, i2e2, u2e3, i2e3, u_train, i_train, r_train, u_test, i_test, i_neg, u_neg, r_neg, r_test, u_adj, i_adj = fold_files(data_train, data_test, data_neg, args)
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(u_train), torch.LongTensor(i_train),
                                              torch.FloatTensor(r_train))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(u_test), torch.LongTensor(i_test),
                                             torch.FloatTensor(r_test))
    negset = torch.utils.data.TensorDataset(torch.LongTensor(u_neg), torch.LongTensor(i_neg),
                                             torch.FloatTensor(r_neg))
    _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=16, pin_memory=True)
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True,
                                        num_workers=16, pin_memory=True)
    _neg = torch.utils.data.DataLoader(negset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=16, pin_memory=True)

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    embed_dim = args.embed_dim
    data_path = './datasets/' + args.dataset

    # drug part
    u_agg_embed_cmp1 = aggregator(u2e1.to(device), i2e1.to(device), u_adj, embed_dim, cuda=device,
                                  weight_decay=args.weight_decay, droprate=args.droprate)
    u_embed_cmp1 = encoder(embed_dim, u_agg_embed_cmp1, cuda=device)

    u_agg_embed_cmp2 = aggregator(u2e2.to(device), i2e2.to(device), u_adj, embed_dim, cuda=device,
                                  weight_decay=args.weight_decay, droprate=args.droprate)

    u_embed_cmp2 = encoder(embed_dim, u_agg_embed_cmp2, cuda=device)

    u_agg_embed_cmp3 = aggregator(u2e3.to(device), i2e3.to(device), u_adj, embed_dim, cuda=device,
                                  weight_decay=args.weight_decay, droprate=args.droprate)

    u_embed_cmp3 = encoder(embed_dim, u_agg_embed_cmp3, cuda=device)


    # side-effect part
    i_agg_embed_cmp1 = aggregator(u2e1.to(device), i2e1.to(device), i_adj, embed_dim, cuda=device,
                                  weight_decay=args.weight_decay, droprate=args.droprate, is_user_part=False)
    i_embed_cmp1 = encoder(embed_dim, i_agg_embed_cmp1, cuda=device, is_user_part=False)

    i_agg_embed_cmp2 = aggregator(u2e2.to(device), i2e2.to(device), i_adj, embed_dim, cuda=device,
                                  weight_decay=args.weight_decay, droprate=args.droprate, is_user_part=False)
    i_embed_cmp2 = encoder(embed_dim, i_agg_embed_cmp2, cuda=device, is_user_part=False)

    i_agg_embed_cmp3 = aggregator(u2e3.to(device), i2e3.to(device), i_adj, embed_dim, cuda=device,
                                  weight_decay=args.weight_decay, droprate=args.droprate, is_user_part=False)
    i_embed_cmp3 = encoder(embed_dim, i_agg_embed_cmp3, cuda=device, is_user_part=False)

    model = MGPred(u_embed_cmp1, u_embed_cmp2, u_embed_cmp3, i_embed_cmp1, i_embed_cmp2, i_embed_cmp3, embed_dim, args.N, droprate=args.droprate).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rmse_mn = np.inf
    mae_mn = np.inf
    endure_count = 0

    for epoch in range(1, args.epochs + 1):
        # ====================   training    ====================
        train(model, _train, optimizer, epoch, rmse_mn, mae_mn, device)
        # ====================     test       ====================
        rmse, mae, ground_i, ground_u, ground_truth, pred = test(model, _test, _test, device)

        if rmse_mn > rmse:
            rmse_mn = rmse
            mae_mn = mae
            endure_count = 0
            lr = 'final_p'
            lrr = lr + str(fold) + '.p'
            with open(lrr, 'wb') as meta:
                pickle.dump((ground_i, ground_u, ground_truth, pred), meta)
        else:
            endure_count += 1

        print("<Test> RMSE: %.5f, MAE: %.5f " % (rmse, mae))
        rmse1, mae1, ground_i, ground_u, ground_truth, pred = test(model, _train, _train, device)
        print("<Train> RMSE: %.5f, MAE: %.5f " % (rmse1, mae1))



        if endure_count > 30:
            break

    print('The best RMSE/MAE: %.5f / %.5f' % (rmse_mn, mae_mn))
    return rmse, mae


def train(model, train_loader, optimizer, epoch, rmse_mn, mae_mn, device):
    model.train()
    avg_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        batch_u, batch_i, batch_ratings = data

        optimizer.zero_grad()
        loss = model.loss(batch_u.to(device), batch_i.to(device), batch_ratings.to(device))
        loss.backward(retain_graph = True)
        optimizer.step()

        avg_loss += loss.item()

        # clamp the parameters
        layers = model.layers
        for k, layer in enumerate(layers):
            layer.constrain_parameters()

        if model.beta_ema > 0.:
            model.update_ema()

        if (i + 1) % 100 == 0:
            print('%s Training: [%d epoch, %3d batch] loss: %.5f, the best RMSE/MAE: %.5f / %.5f' % (
                datetime.now(), epoch, i + 1, avg_loss / 100, rmse_mn, mae_mn))
            avg_loss = 0.0
    return 0


def test(model, test_loader, neg_loader, device):
    model.eval()

    if model.beta_ema > 0:
        old_params = model.get_params()
        model.load_ema_params()

    pred = []
    ground_truth = []
    ground_u = []
    ground_i = []

    for test_u, test_i, test_ratings in test_loader:
        ground_i.append(list(test_i.data.cpu().numpy()))
        ground_u.append(list(test_u.data.cpu().numpy()))
        test_u, test_i, test_ratings = test_u.to(device), test_i.to(device), test_ratings.to(device)
        scores_one = model(test_u, test_i)
        pred.append(list(scores_one.data.cpu().numpy()))
        ground_truth.append(list(test_ratings.data.cpu().numpy()))

    pred = np.array(sum(pred, []), dtype = np.float32)
    ground_truth = np.array(sum(ground_truth, []), dtype = np.float32)


    rmse = sqrt(mean_squared_error(pred, ground_truth))
    mae = mean_absolute_error(pred, ground_truth)

    if model.beta_ema > 0:
        model.load_params(old_params)
    return rmse, mae, ground_i, ground_u, ground_truth, pred


def ten_fold(args):
    rawpath = args.rawpath
    gii = open(rawpath+'/drug_side.pkl', 'rb')
    drug_side = pickle.load(gii)
    gii.close()
    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(drug_side, addition_negative_number='all')
    addition_negative_sample = np.vstack((addition_negative_sample, final_negative_sample))
    final_sample = final_positive_sample
    X = final_sample[:, 0::]
    final_target = final_sample[:, final_sample.shape[1] - 1]
    y = final_target
    data = []
    data_x = []
    data_y = []
    data_neg_x = []
    data_neg_y = []
    data_neg = []
    for i in range(addition_negative_sample.shape[0]):
        data_neg_x.append((addition_negative_sample[i, 1], addition_negative_sample[i, 0]))
        data_neg_y.append((int(float(addition_negative_sample[i, 2]))))
        data_neg.append((addition_negative_sample[i, 1], addition_negative_sample[i, 0], addition_negative_sample[i, 2]))
    for i in range(X.shape[0]):
        data_x.append((X[i, 1], X[i, 0]))
        data_y.append((int(float(X[i, 2]))))
        data.append((X[i, 1], X[i, 0], X[i, 2]))
    fold = 1
    kfold = StratifiedKFold(10, random_state=1, shuffle=True)
    total_rmse, total_mae = [], []
    for k, (train, test) in enumerate(kfold.split(data_x, data_y)):
        print("==================================fold {} start".format(fold))
        data = np.array(data)
        rmse, mae = train_test(data[train].tolist(), data[test].tolist(), data_neg, fold, args)
        total_rmse.append(rmse)
        total_mae.append(mae)
        print("==================================fold {} end".format(fold))
        fold += 1
        print('Total_RMSE:')
        print(np.mean(total_rmse))
        print('Total_MAE:')
        print(np.mean(total_mae))
        sys.stdout.flush()


def Extract_positive_negative_samples(DAL, addition_negative_number='all'):
    k = 0
    interaction_target = np.zeros((DAL.shape[0]*DAL.shape[1], 3)).astype(int)
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k,0] = i
            interaction_target[k,1] = j
            interaction_target[k,2] = DAL[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]  # 按照最后一列对行排序
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]
    negative_sample = data_shuffle[0:interaction_target.shape[0] - number_positive]
    a = np.arange(interaction_target.shape[0] - number_positive)
    a = list(a)
    if addition_negative_number == 'all':
        b = random.sample(a, (interaction_target.shape[0] - number_positive))
    else:
        b = random.sample(a, (1 + addition_negative_number) * number_positive)
    final_negtive_sample = negative_sample[b[0:number_positive], :]
    addition_negative_sample = negative_sample[b[number_positive::], :]
    return addition_negative_sample, final_positive_sample, final_negtive_sample


def main():
    # Training settings
    parser = argparse.ArgumentParser(description = 'MGPred')
    parser.add_argument('--epochs', type = int, default = 100,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.001,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 64,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 0.0005,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--N', type = int, default = 30000,
                        metavar = 'N', help = 'L0 parameter')
    parser.add_argument('--droprate', type = float, default = 0.5,
                        metavar = 'FLOAT', help = 'dropout rate')
    parser.add_argument('--batch_size', type = int, default = 256,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default = 256,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--dataset', type = str, default = 'yelp',
                        metavar = 'STRING', help = 'dataset')
    parser.add_argument('--rawpath', type=str, default='D:/~博士/~收集的文献/需要研究的模型/MCC/My_dataset',
    # parser.add_argument('--rawpath', type=str, default='/home/zhaohc/My_dataset',
                        metavar='STRING', help='rawpath')
    args = parser.parse_args()

    print('Dataset: ' + args.dataset)
    print('-------------------- Hyperparams --------------------')
    print('N: ' + str(args.N))
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))
    ten_fold(args)

if __name__ == "__main__":
    main()