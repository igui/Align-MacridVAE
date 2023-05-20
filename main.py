import math
import os
from pathlib import Path
import sys
import time
import argparse
from typing import List, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from data import load_simple_data, load_cates, load_urls
from model import load_net


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True,
                    help='film-trust, ciao-dvd, etc.')
parser.add_argument('--model', type=str, default='DisenVAE',
                    help='MultiDAE, MultiVAE, DisenVAE, DisenEVAE')
parser.add_argument('--mode', type=str, default='train',
                    help='train, test, visualize')
parser.add_argument('--seed', type=int, default=98765)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=800)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.2)
parser.add_argument('--kfac', type=int, default=7)
parser.add_argument('--dfac', type=int, default=200)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--device', type=str, default='cpu',
                    help='cpu, cuda:n')
parser.add_argument('--log', type=str, default='file')
args = parser.parse_args()


if args.seed < 0:
    args.seed = int(time.time())
info = '%s-%s-%dE-%dB-%gL-%gW-%gD-%gb-%dk-%dd-%gt-%ds' \
    % (args.data, args.model, args.epochs, args.batch_size, args.lr, args.weight_decay,
       args.dropout, args.beta, args.kfac, args.dfac, args.tau, args.seed)
print(info)


np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')


dir = Path('RecomData') / args.data / 'prep'
print('Loading data')
data = load_simple_data(dir)
print('Loading net')
net = load_net(args.model, data.n_users, data.n_items, args.kfac, args.dfac,
               args.tau, args.dropout, data.items_embed)
net.to(device)

def load_model_weights(net: nn.Module):
    weights = torch.load('run/%s/model.pkl' % info, map_location=device)
    net.load_state_dict(weights)

def train_model(
    net: nn.Module,
    train: sparse.csr_matrix,
    validation: sparse.csr_matrix,
    validation_test: sparse.csr_matrix,
    lr: float,
    epochs: int,
    weight_decay: float,
    batch_size: int
):
    # We should test that the validation and test shapes are complementary
    assert validation.shape == validation_test.shape

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = net.loss_fn

    n_train = train.shape[0]
    n_batches = math.ceil(n_train / batch_size)
    update = 0
    anneals = 500 * n_batches

    best_n100 = 0.0
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_idx = np.random.permutation(n_train)

        t = time.time()
        for start_idx in range(0, n_train, batch_size):
            end_idx = min(start_idx + batch_size, n_train)
            X = train[train_idx[start_idx: end_idx]]
            X = torch.Tensor(X.toarray()).to(device)     # users-items matrix
            optimizer.zero_grad()
            X_logits, X_mu, X_logvar, A_logits, A_mu, A_logvar = net(X, A=None)
            anneal = min(args.beta, update / anneals)
            loss = criterion(X, X_logits, X_mu, X_logvar,
                             None, A_logits, A_mu, A_logvar, anneal)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            update += 1

        print('[%3d] loss: %.3f' % (epoch, running_loss / n_batches), end='\t', file=log)
        n100 = test_model(
            net,
            validation,
            validation_test,
            batch_size=args.batch_size
        ).ndcg_100
        if n100 > best_n100:
            best_n100 = n100
            torch.save(net.state_dict(), 'run/%s/model.pkl' % info)
        print('time: %.3f' % (time.time() - t), file=log)

class TestStatsBatch(NamedTuple):
    ndcg_100: torch.Tensor
    recall_20: torch.Tensor
    recall_50: torch.Tensor

    @staticmethod
    def concatenate(batches: List['TestStatsBatch']) -> 'TestStatsBatch':
        return TestStatsBatch(
            ndcg_100=torch.cat([b.ndcg_100 for b in batches]),
            recall_20=torch.cat([b.recall_20 for b in batches]),
            recall_50=torch.cat([b.recall_50 for b in batches])
        )


class TestStats(NamedTuple):
    ndcg_100: float
    recall_20: float
    recall_50: float

    @staticmethod
    def mean_from_batch(batch: TestStatsBatch) -> 'TestStats':
        return TestStats(
            ndcg_100=batch.ndcg_100.mean(),
            recall_20=batch.recall_20.mean(),
            recall_50=batch.recall_50.mean()
        )


def make_chunks(*xs: sparse.csr_matrix, batch_size: int):
    """
    Iterates a group of matrices of the same shape. And goes ahead and
    splits the results into batches of rows with size `batch_size`
    """
    assert len(xs) > 0
    n_rows = xs[0].shape[0]

    for start_idx in range(0, n_rows, batch_size):
        yield (
            x[start_idx:start_idx+batch_size]
            for x in xs
        )

def test_batch(
    net: nn.Module, x: sparse.csr_matrix, y: sparse.csr_matrix
) -> TestStatsBatch:
    x_device = torch.Tensor(x.toarray()).to(device)
    y_tensor = torch.Tensor(y.toarray())

    x_pred, *_ = net(x_device, A=None)

    # exclude x's samples from the data
    x_pred[torch.nonzero(x_device, as_tuple=True)] = float('-inf')
    x_pred = x_pred.cpu()

    return TestStatsBatch(
        ndcg_100=ndcg_kth(x_pred, y_tensor, k=100),
        recall_20=recall_kth(x_pred, y_tensor, k=20),
        recall_50=recall_kth(x_pred, y_tensor, k=50)
    )

def print_attr(name: str, values: torch.Tensor) -> float:
    mean_val = values.mean()
    std_dev = values.std() / np.sqrt(len(values))
    msg = f'{name}: {mean_val:.5f}(±{std_dev:.5f})'
    print(msg, end='\t', file=log)


def check_interactions(
    train_interactions: sparse.csr_matrix,
    validation_interactions: sparse.csr_matrix
):
    assert train_interactions.shape == validation_interactions.shape, \
        "Must be same shape"

    # It can happen
    # assert a.multiply(b).sum() == 0, "Must not have elements in common"

    assert train_interactions.sum(axis=1).min() > 0, \
        "train_interactions must have logits in all users"

    # We should have this uncommented. The test should have logits for all users
    # assert b.sum(axis=1).min() > 0, "Must have logits in all users"

def test_model(
        net: nn.Module,
        interactions: sparse.csr_matrix,
        test_interactions: sparse.csr_matrix,
        batch_size: int
    ) -> TestStats:
    net.eval()
    check_interactions(interactions, test_interactions)

    with torch.no_grad():
        batch_results = TestStatsBatch.concatenate([
            test_batch(net, x, y)
            for x, y in make_chunks(interactions, test_interactions, batch_size=batch_size)
        ])

    print_attr('ndcg@100', batch_results.ndcg_100)
    print_attr('recall@20', batch_results.recall_20)
    print_attr('recall@50', batch_results.recall_50)

    return TestStats.mean_from_batch(batch_results)


def evaluate_batch(net: nn.Module, x: sparse.csr_matrix):
    x_device = torch.Tensor(x.toarray()).to(device)
    x_pred, *_ = net(x_device, A=None)
    return x_pred.cpu()


def evaluate_model(
    net: nn.Module,
    interactions: sparse.csr_matrix,
    batch_size: int
) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        return torch.cat([
            evaluate_batch(net, x)
            for x, in make_chunks(interactions, batch_size=batch_size)
        ])

def ndcg_kth(outputs, labels, k=100):
    _, preds = torch.topk(outputs, k)               # sorted top k index of outputs
    _, facts = torch.topk(labels, k)                # min(k, labels.nnz(dim=1))
    rows = torch.arange(labels.shape[0]).view(-1, 1)

    tp = 1.0 / torch.log2(torch.arange(2, k + 2).float())
    dcg = torch.sum(tp * labels[rows, preds], dim=1)
    idcg = torch.sum(tp * labels[rows, facts], dim=1)
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0
    return ndcg


def recall_kth(outputs, labels, k=50):
    _, preds = torch.topk(outputs, k, sorted=False) # top k index
    rows = torch.arange(labels.shape[0]).view(-1, 1)

    recall = torch.sum(labels[rows, preds], dim=1) \
           / torch.min(torch.Tensor([k]), torch.sum(labels, dim=1))
    recall[torch.isnan(recall)] = 0
    return recall


def visualize(net, train):
    net.eval()
    n_visual = train.shape[0]
    n_items = train.shape[1]

    users = []
    with torch.no_grad():
        for start_idx in range(0, n_visual, args.batch_size):
            X = train[start_idx:start_idx + args.batch_size]
            X = torch.Tensor(X.toarray()).to(device)
            _, X_mu, _, _, _, _ = net(X, A=None)
            users.append(X_mu)

    users = torch.cat(users).detach().cpu()
    items = net.state_dict()['items'].detach().cpu()
    cores = net.state_dict()['cores'].detach().cpu()

    users = F.normalize(users).view(-1, args.kfac, args.dfac)
    items = F.normalize(items)
    cores = F.normalize(cores)

    # align categories with prototypes

    items_cates = load_cates(dir, n_items, args.kfac)
    items_cates = match_cores_cates(items, cores, items_cates)
    # items_item, items_cate = items_cates.nonzero()
    items_cate = np.argmax(items_cates, axis=1)

    # users and the categories they bought
    assert sparse.isspmatrix(train)
    users_cates = train.dot(items_cates)
    users_user, users_cate = users_cates.nonzero()
    users = users[users_user, users_cate, :]

    # nodes (items and users) prediction and ground truth
    nodes = torch.cat((items, users)).numpy()
    nodes_pred = np.argmax(np.dot(nodes, cores.T), axis=1)
    nodes_true = np.concatenate((items_cate, users_cate), axis=0)


    # plot pictures
    palette = np.array(
        [[35 , 126, 181, 80], # _0. Blue
        [255, 129, 190, 80],  # _1. Pink
        [255, 127, 38 , 80],  # _2. Orange
        [59 , 175, 81 , 80],  # _3. Green
        [156, 78 , 161, 80],  # _4. Purple
        [238, 27 , 39 , 80],  # _5. Red
        [153, 153, 153, 80]], # _6. Gray
        dtype=float) / 255.0

    col_pred = palette[nodes_pred]
    col_true = palette[nodes_true]

    try:
        nodes_2d = np.load('run/%s/tsne.npy' % info)
    except:
        print('tsne...')
        nodes_kd = PCA(n_components=args.kfac).fit_transform(nodes) \
                   if args.dfac > args.kfac else nodes
        nodes_2d = TSNE(n_jobs=8).fit_transform(nodes_kd)
        np.save('run/%s/tsne.npy' % info, nodes_2d)
    plot('tsne2d-nodes-pred', nodes_2d, col_pred)
    plot('tsne2d-nodes-true', nodes_2d, col_true)
    plot('tsne2d-items-pred', nodes_2d[:n_items], col_pred[:n_items])
    plot('tsne2d-items-true', nodes_2d[:n_items], col_true[:n_items])
    plot('tsne2d-users-pred', nodes_2d[n_items:], col_pred[n_items:])
    plot('tsne2d-users-true', nodes_2d[n_items:], col_true[n_items:])


def match_cores_cates(items, cores, items_cates):
    '''
    align categories with prototypes

    items = embedding matrix [m, d]
    cores = embedding matrix [k, d]
    items_cates = sparse one-hot matrix [m, k]
    '''
    cates = np.argmax(items_cates, axis=1)
    cates_centers = [torch.sum(items[cates == ki], dim=0, keepdim=True)
                    for ki in range(args.kfac)]
    cates_centers = torch.cat(cates_centers, dim=0)
    cates_centers = F.normalize(cates_centers)
    cores_cates = torch.mm(cores, cates_centers.t())
    cates2cores = torch.argmax(cores_cates, dim=1).numpy()
    cores2cates = torch.argmax(cores_cates, dim=0).numpy()

    print('cates:', cates2cores, file=log)
    print('cores:', cores2cates, file=log)

    if len(set(cates2cores)) == args.kfac:
        print('interpretability: 1\tseed: %d' % args.seed)
        if len(set(cores2cates)) == args.kfac:
            print('interpretability: 2\tseed: %d' % args.seed)
            for ki in range(args.kfac):
                if cores2cates[cates2cores[ki]] != ki:
                    break
            else:
                print('interpretability: 3\tseed: %d' % args.seed)
        return items_cates[:, cates2cores]
    print('Some prototypes do not align well with categories.', file=log)

    # Fix cates2cores
    cores_cates = torch.mm(cores, cates_centers.t()).numpy()
    cates2cores = np.zeros((args.kfac, ), dtype=int)
    for ki in range(args.kfac):
        loc = np.where(cores_cates == np.max(cores_cates))
        core, cate = loc[0][0], loc[1][0]   # (array([r]), array[c])
        cores_cates[core, :] = - 1.0
        cores_cates[:, cate] = - 1.0
        cates2cores[core] = cate

    print(cates2cores, file=log)
    assert len(set(cates2cores)) == args.kfac
    return items_cates[:, cates2cores]


def plot(fname, xy, color, marksz=1.0):
    plt.figure()
    plt.scatter(x=xy[:, 0], y=xy[:, 1], c=color, s=marksz)
    plt.savefig('run/%s/%s.png' % (info, fname))



def uncorrelate(net, train, test):
    net.eval()
    n_disen = train.shape[0]
    users = []
    n5s, r20s, r5s = [], [], []
    with torch.no_grad():
        for start_idx in range(0, n_disen, args.batch_size):
            end_idx = min(start_idx + args.batch_size, n_disen)
            X_tr  = train[start_idx: end_idx]
            X_te  = test[start_idx: end_idx]
            X_tr = torch.Tensor(X_tr.toarray()).to(device)
            X_te = torch.Tensor(X_te.toarray())
            X_tr_logits, X_mu, _, _, _, _ = net(X_tr, A=None)

            users.append(X_mu)
            X_tr_logits[torch.nonzero(X_tr, as_tuple=True)] = float('-inf')
            X_tr_logits = X_tr_logits.cpu()
            n5s.append(ndcg_kth(X_tr_logits, X_te, k=5))
            r20s.append(recall_kth(X_tr_logits, X_te, k=20))
            r5s.append(recall_kth(X_tr_logits, X_te, k=50))
    n5 = torch.cat(n5s).mean().item()
    r20 = torch.cat(r20s).mean().item()
    r5 = torch.cat(r5s).mean().item()

    users = torch.cat(users).detach().cpu()

    users = F.normalize(users).view(-1, args.kfac, args.dfac).numpy()
    items = net.state_dict()['items'].detach().cpu()
    items = F.normalize(items).numpy()

    uncorr_user = []
    for ki in range(args.kfac):
        corr = np.corrcoef(users[:, ki, :], rowvar=False)
        np.fill_diagonal(corr, 0)
        uncorr = 1.0 - 1.0 / (args.dfac * (args.dfac - 1)) * np.sum(np.abs(corr))
        uncorr_user.append(uncorr)
    uncorr_user = np.array(uncorr_user).mean()

    corr = np.corrcoef(items, rowvar=False)
    np.fill_diagonal(corr, 0)
    uncorr_item = 1.0 - 1.0 / (args.dfac * (args.dfac - 1)) * np.sum(np.abs(corr))

    return uncorr_user, uncorr_item, n5, r20, r5



def disentangle(net):
    urls = load_urls(dir)
    items = net.state_dict()['items'].detach().cpu()
    items = F.normalize(items)
    start = - 1.0
    stop = 1.0
    step = (stop - start) / 10
    item_id = 19436


    for did in range(0, 200, 10):
        jpg_dir = 'run/%s-%s-disen-%d-%d' % (args.data, args.model, item_id, did)
        if not os.path.exists(jpg_dir):
            os.mkdir(jpg_dir)
        vect = items[item_id, :].clone()
        items[item_id, :] = torch.zeros_like(vect)
        for i in range(10):
            fac = start + step * i
            vect[did] = fac
            new_vect = F.normalize(vect, dim=-1)
            sim_id = torch.argmax(torch.sum(new_vect * items, dim=1))
            if urls[sim_id]:
                os.system('wget %s -O %s/%d.jpg' % (urls[sim_id], jpg_dir, i))


if args.mode == 'train':
    if not os.path.exists('run'):
        os.mkdir('run')
    if not os.path.exists('run/%s' % info):
        os.mkdir('run/%s' % info)

log = open(f'run/{info}/log.txt', mode='a', buffering=1) if args.log == 'file' else sys.stdout


if args.mode == 'train':
    print('training ...')
    t = time.time()
    try:
        train_model(
            net,
            train=data.train,
            validation=data.train,
            validation_test=data.validation,
            lr=args.lr,
            epochs=args.epochs,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        print('terminate training...')
    print('train time: %.3f' % (time.time() - t), file=log)


if args.mode == 'train' or args.mode == 'test':
    assert os.path.exists('run/%s' % info)
    print('Loading model')
    load_model_weights(net)
    print('evaluating...')
    y_pred = evaluate_model(
        net,
        data.train,
        batch_size=args.batch_size
    )
    print('saving...')
    np.savez(f'run/{info}/eval.npz', y_pred)
    print('testing ...')
    t = time.time()
    test_model(
        net,
        data.train,
        data.test,
        batch_size=args.batch_size
    )
    print('test time: %.3f' % (time.time() - t), file=log)


if args.mode == 'visualize':
    assert os.path.exists('run/%s' % info)
    assert args.model in ['DisenVAE', 'DisenEVAE']
    print('visualizing...')
    t = time.time()
    load_model_weights(net)
    visualize(net, data.train)
    print('visualize time: %.3f' % (time.time() - t))


if args.mode == 'uncorrelate':
    assert os.path.exists('run/%s' % info)
    assert args.model in ['MultiVAE', 'DisenVAE', 'DisenEVAE']
    print('uncorrelating...')
    t = time.time()
    load_model_weights(net)
    disen_user, disen_item, n5, r20, r5 = uncorrelate(net, data.train, data.test)
    with open('run/%s-%s-beta-log.txt' % (args.data, args.model), 'a') as file:
        file.write('%s\t%f\t%f\t%f\t%f\t%f\t%f\n' % (args.data, args.beta, disen_user, disen_item, n5, r20, r5))
    print('uncorrelate time: %.3f' % (time.time() - t))


if args.mode == 'disentangle':
    assert os.path.exists('run/%s' % info)
    assert args.model in ['DisenVAE', 'DisenEVAE']
    print('disentangling...')
    t = time.time()
    load_model_weights(net)
    disentangle(net)
    print('disentangle time: %.3f' % (time.time() - t))


log.close()
