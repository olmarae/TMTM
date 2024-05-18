import torch
from torch import nn
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F


class TMTM(nn.Module):
    def __init__(self, hidden_dimension=128, out_dim=2, relation_num=12, dropout=0.3):
        super(TMTM, self).__init__()
        self.dropout = dropout

        self.linear_relu_des=nn.Sequential(
            nn.Linear(768, int(hidden_dimension/4)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(768, int(hidden_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(34, int(hidden_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(12, int(hidden_dimension/4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.rgcn = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        d = self.linear_relu_des(feature[:, -1536:-768].to(torch.float32))
        t = self.linear_relu_tweet(feature[:, -768:].to(torch.float32))
        num = self.linear_relu_num_prop(feature[:, 12:46].to(torch.float32))
        cat = self.linear_relu_cat_prop(feature[:, 0:12].to(torch.float32))
        
        x = torch.cat((d, t, num, cat), dim=1)
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x