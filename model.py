import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from tqdm import tqdm


def load_net(model, N, M, K, D, tau, dropout, items, items_visual, items_textual):
    if model == 'MultiDAE':
        return MultiDAE(M, D, dropout)
    elif model == 'MultiVAE':
        return MultiVAE(M, D, dropout)
    elif model == 'DisenVAE':
        return DisenVAE(M, K, D, tau, dropout)
    elif model == 'DisenEVAE':
        return DisenEVAE(M, K, D, tau, dropout, items)
    elif model == 'DisenEEVAEMulti':
        return DisenEEVAEMulti(M, K, D, tau, dropout, items_visual, items_textual)
    raise ValueError(f'Unknown model: {model}')


def recon_loss(inputs, logits):
    return torch.mean(torch.sum(-logits * inputs, dim=1))

def kl_loss(mu, logvar):
    return torch.mean(0.5 * torch.sum(
        torch.exp(logvar) + mu ** 2 - 1 - logvar, dim=1))


class MultiDAE(nn.Module):
    def __init__(self, M, D, dropout):
        super(MultiDAE, self).__init__()

        self.M = M
        self.H = D * 3
        self.D = D

        self.encoder = nn.Sequential(
            nn.Linear(self.M, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.D),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.D, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.M)
        )
        self.drop = nn.Dropout(dropout)

    def encode(self, X):
        X = self.drop(X)
        h = self.encoder(X)
        return h

    def decode(self, h):
        logits = self.decoder(h)
        logits = F.log_softmax(logits, dim=1)
        return logits

    def forward(self, X, A):
        h = self.encode(X)
        logits = self.decode(h)
        return logits, None, None, None, None, None

    def loss_fn(self, X, X_logits, X_mu, X_logvar,
                A, A_logits, A_mu, A_logvar, anneal):
        return recon_loss(X, X_logits)


class MultiVAE(nn.Module):
    def __init__(self, M, D, dropout):
        super(MultiVAE, self).__init__()

        self.M = M
        self.H = D * 3
        self.D = D

        self.encoder = nn.Sequential(
            nn.Linear(self.M, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.D * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.D, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.M)
        )
        self.drop = nn.Dropout(dropout)

    def encode(self, X):
        X = self.drop(X)
        h = self.encoder(X)
        mu = h[:, :self.D]
        logvar = h[:, self.D:]
        return mu, logvar

    def decode(self, z):
        logits = self.decoder(z)
        logits = F.log_softmax(logits, dim=1)
        return logits

    def sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def forward(self, X, A):
        mu, logvar = self.encode(X)
        z = self.sample(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar, None, None, None

    def loss_fn(self, X, X_logits, X_mu, X_logvar,
                A, A_logits, A_mu, A_logvar, anneal):
        return recon_loss(X, X_logits) + anneal * kl_loss(X_mu, X_logvar)


class DisenVAE(nn.Module):
    def __init__(self, M, K, D, tau, dropout):
        super(DisenVAE, self).__init__()

        self.M = M
        self.H = D * 3
        self.D = D
        self.K = K
        self.tau = tau

        self.encoder = nn.Sequential(
            nn.Linear(self.M, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.D * 2)
        )
        self.items = Parameter(torch.Tensor(self.M, self.D))
        self.cores = Parameter(torch.Tensor(self.K, self.D))
        self.drop = nn.Dropout(dropout)

        init.xavier_normal_(self.items)
        init.xavier_normal_(self.cores)

    def cluster(self):
        items = F.normalize(self.items, dim=1)      # M * D
        cores = F.normalize(self.cores, dim=1)      # K * D
        cates = torch.mm(items, cores.t()) / self.tau
        cates = F.softmax(cates, dim=1)             # M * K
        return items, cores, cates

    def encode(self, X, cates):
        n = X.shape[0]
        X = self.drop(X)
        X = X.view(n, 1, self.M) *  \
            cates.t().expand(n, self.K, self.M)     # n * K * M
        X = X.reshape(n * self.K, self.M)           # (n * K) * M
        h = self.encoder(X)                         # (n * K) * D * 2
        mu, logvar = h[:, :self.D], h[:, self.D:]   # (n * k) * D
        return mu, logvar

    def decode(self, z, items, cates):
        n = z.shape[0] // self.K
        z = F.normalize(z, dim=1)                   # (n * K) * D
        logits = torch.mm(z, items.t()) / self.tau  # (n * K) * M
        probs = torch.exp(logits)                   # (n * K) * M
        probs = torch.sum(probs.view(n, self.K, self.M) *
                          cates.t().expand(n, self.K, self.M), dim=1)
        logits = torch.log(probs)
        logits = F.log_softmax(logits, dim=1)
        return logits

    def sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def forward(self, X, A):
        items, cores, cates = self.cluster()
        mu, logvar = self.encode(X, cates)
        z = self.sample(mu, logvar)
        logits = self.decode(z, items, cates)
        return logits, mu, logvar, None, None, None

    def loss_fn(self, X, X_logits, X_mu, X_logvar,
                A, A_logits, A_mu, A_logvar, anneal):
        return recon_loss(X, X_logits) + anneal * kl_loss(X_mu, X_logvar)


class DisenEVAE(DisenVAE):
    def __init__(self, M, K, D, tau, dropout, items):
        super(DisenEVAE, self).__init__(M, K, D, tau, dropout)

        # change the feature from X to self.D dimensions
        items = PCA(n_components=self.D).fit_transform(items)
        # fit the xavier_normal distribution i.e. mu = 0, std = sqrt(2 / (fan_in + fan_out))
        items = scale(items, axis=1) * np.sqrt(2 / (M + D))
        # init the feature of cores
        cores = KMeans(n_clusters=self.K, n_init=10).fit(
            items).cluster_centers_

        self.items = Parameter(torch.Tensor(items))
        self.cores = Parameter(torch.Tensor(cores))

# class DisenEEVAEMulti(DisenVAE):
#     def __init__(self, M, K, D, tau, dropout, items_visual, items_textual):
#         super().__init__(2*M, K, D, tau, dropout)

#         # change the feature from X to self.D dimensions
#         items = np.concatenate((items_visual, items_textual), axis=0)
#         items = PCA(n_components=self.D).fit_transform(items)
#         # fit the xavier_normal distribution i.e. mu = 0, std = sqrt(2 / (fan_in + fan_out))
#         items = scale(items, axis=1) * np.sqrt(2 / (2*M + D))
#         # init the feature of cores
#         cores = KMeans(n_clusters=self.K, n_init=10).fit(
#             items).cluster_centers_

#         self.items = Parameter(torch.Tensor(items))
#         self.cores = Parameter(torch.Tensor(cores))

#     def forward(self, X, A):
#         X_concatenated = torch.cat((X, X), axis=1)
#         items, cores, cates = self.cluster()
#         mu, logvar = self.encode(X_concatenated, cates)
#         z = self.sample(mu, logvar)
#         logits = self.decode(z, items, cates)

#         real_items = self.M // 2
#         logits_visual = logits[:, :real_items]
#         logits_textual = logits[:, real_items:]
#         logits = torch(logits_visual + logits_textual) / 2

#         return logits, mu, logvar, None, None, None


class Aligner(nn.Module):
    def __init__(
            self,
            input_dimension_textual: int,
            input_dimension_visual: int,
            embedding_dimension: int,
        ):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(
                input_dimension_visual,
                embedding_dimension,
                bias=True
            ),
            #nn.Tanh()
        )
        self.textual_proj = nn.Sequential(
            nn.Linear(
                input_dimension_textual,
                embedding_dimension,
                bias=True
            ),
            #nn.Tanh()
        )
        self.tau = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x_textual, x_visual):
        similarities = self.similarity(x_textual, x_visual)
        target = torch.arange(similarities.shape[0]).to(similarities.device)
        loss_1 = F.cross_entropy(similarities, target)
        loss_2 = F.cross_entropy(similarities.T, target)
        return (loss_1 + loss_2) / 2

    def similarity(self, x_textual, x_visual):
        assert x_textual.shape[0] == x_visual.shape[0]

        #assert x_textual.shape[1] == self.textual_proj.in_features
        #assert x_visual.shape[1] == self.visual_proj.in_features

        textual_proj = self.textual_proj(x_textual)
        textual_proj_norm = F.normalize(textual_proj)
        visual_proj = self.visual_proj(x_visual)
        visual_proj_norm = F.normalize(visual_proj)

        return (textual_proj_norm @ visual_proj_norm.T) * torch.exp(self.tau)

    def fit(self, items_textual, items_visual, batch_size=128, epochs=10):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_items = items_textual.shape[0]

        dataset = EmbedDataset(
            items_textual=items_textual,
            items_visual=items_visual
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(),
            weight_decay=0.01,
            lr=0.001
        )
        for epoch in range(epochs):
            progress = tqdm(dataloader)
            accum_loss = 0
            for idx, x in enumerate(progress):
                optimizer.zero_grad()
                loss = self(x_textual=x['text'], x_visual=x['visual'])
                accum_loss += loss.item()
                progress.set_postfix_str(f'Loss: {accum_loss / (idx+1):.4f}')
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch+1} loss: {accum_loss / len(progress):.4f}')

    def transform(self, items_textual, items_visual):
       self.eval()
       with torch.no_grad():
        items_visual = self.visual_proj(
            torch.tensor(items_visual, dtype=torch.float32)
            )
        items_textual = self.textual_proj(
            torch.tensor(items_textual, dtype=torch.float32)
            )

        return items_textual.numpy(), items_visual.numpy()


class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self, items_textual, items_visual):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert items_textual.shape[0] == items_visual.shape[0]
        self.items_textual = items_textual.astype(np.float32)
        self.items_visual = items_visual.astype(np.float32)

    def __getitem__(self, index):
        return {
            'text': torch.Tensor(self.items_textual[index]).to(self.device),
            'visual': torch.Tensor(self.items_visual[index]).to(self.device)
        }

    def __len__(self):
        return self.items_textual.shape[0]


class DisenEEVAEMulti(DisenVAE):
    def __init__(self, M, K, D, tau, dropout, items_visual, items_textual):
        super().__init__(2*M, K, D, tau, dropout)

        # change the feature from X to self.D dimensions
        aligner = Aligner(
            input_dimension_textual=items_textual.shape[1],
            input_dimension_visual=items_visual.shape[1],
            embedding_dimension=D
        )
        aligner.to('cuda')
        aligner.fit(items_textual=items_textual,
                    items_visual=items_visual,
                    epochs=10, batch_size=32
                )
        aligner.to('cpu')
        items_textual, items_visual = aligner.transform(
            items_textual=items_textual,
            items_visual=items_visual
        )
        items = np.concatenate((items_visual, items_textual), axis=0)

        # fit the xavier_normal distribution i.e. mu = 0, std = sqrt(2 / (fan_in + fan_out))
        # items = scale(items, axis=1) * np.sqrt(2 / (2*M + D))

        # init the feature of cores
        cores = KMeans(n_clusters=self.K, n_init=10).fit(
            items).cluster_centers_

        self.items = Parameter(torch.Tensor(items))
        self.cores = Parameter(torch.Tensor(cores))

    def forward(self, X, A):
        X_concatenated = torch.cat((X, X), axis=1)
        items, cores, cates = self.cluster()
        mu, logvar = self.encode(X_concatenated, cates)
        z = self.sample(mu, logvar)
        logits = self.decode(z, items, cates)

        real_items = self.M // 2
        logits_visual = logits[:, :real_items]
        logits_textual = logits[:, real_items:]
        logits = (logits_visual + logits_textual) / 2

        return logits, mu, logvar, None, None, None


# class DisenEEVAEMulti(DisenVAE):
#     def __init__(self, M, K, D, tau, dropout, items_visual, items_textual):
#         super().__init__(2*M, K, D, tau, dropout)

#         # change the feature from X to self.D dimensions
#         items = np.concatenate((items_visual, items_textual), axis=0)
#         items = PCA(n_components=self.D).fit_transform(items)
#         # fit the xavier_normal distribution i.e. mu = 0, std = sqrt(2 / (fan_in + fan_out))
#         items = scale(items, axis=1) * np.sqrt(2 / (2*M + D))
#         # init the feature of cores
#         cores = KMeans(n_clusters=self.K, n_init=10).fit(
#             items).cluster_centers_

#         self.items = Parameter(torch.Tensor(items))
#         self.cores = Parameter(torch.Tensor(cores))

#     def forward(self, X, A):
#         X_concatenated = torch.cat((X, X), axis=1)
#         items, cores, cates = self.cluster()
#         mu, logvar = self.encode(X_concatenated, cates)
#         z = self.sample(mu, logvar)
#         logits = self.decode(z, items, cates)

#         real_items = self.M // 2
#         logits_visual = logits[:, :real_items]
#         logits_textual = logits[:, real_items:]
#         logits = torch(logits_visual + logits_textual) / 2

#         return logits, mu, logvar, None, None, None
