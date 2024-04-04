import torch
import torch.nn as nn

class FM(nn.Module):
    def __init__(self, num_feats, emb_dim):
        super(FM, self).__init__()
        self.embedding = nn.Embedding(num_feats, emb_dim)
        self.weight = nn.Parameter(torch.zeros(num_feats))
        self.offset = nn.Parameter(torch.zeros(1))

        torch.nn.init.xavier_uniform_(self.embedding.weight)

    #         torch.nn.init.uniform_(self.embedding.weight)

    def forward(self, X):
        emb = self.embedding(X)
        square_of_sum = torch.sum(emb, dim=1) ** 2
        sum_of_square = torch.sum(emb ** 2, dim=1)
        interaction = (square_of_sum - sum_of_square).sum(dim=1) * 0.5

        linear = torch.sum(self.weight[X], dim=1)

        X_out = self.offset + linear + interaction

        X_out = self.sigmoid_range(X_out)

        return X_out

    # to return
    def sigmoid_range(self, x, low=1, high=5):
        return torch.sigmoid(x) * (high - low) + low