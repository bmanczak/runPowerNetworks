# initial data dump from SMAAC paper repo

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GTrXL(nn.Module):

    def __init__(self, hidden_dim, nheads,
                num_layers = 2, num_in_features = None, dropout=0):
        """
        Stacks GAT layers.

        Args:
            hidden_dim ([int]): Number of features of each node after transformation.
            nheads (int): number of attention heads.
            num_layers (int, optional): Number of GAT layers. Defaults to 2.
            dropout (int, optional): Dropout probability. Defaults to 0.
            num_in_features ([int], optional): Number of input features. Defaults to None.
                If not None then before the first layer a linear layer is added to transfrom 
                the input from num_in_features to hidden_dim.
        """
        super(GTrXL, self).__init__()
        self.num_in_features = num_in_features

        if num_in_features is not None:
            self.linear = nn.Linear(num_in_features, hidden_dim)

        self.gat_layers = nn.ModuleList([
                GATLayer(hidden_dim, nheads, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x, adj):
        if self.num_in_features is not None:
            x = self.linear(x)
        for layer in self.gat_layers:
            node_embeddings = layer(x, adj)

        return node_embeddings 
class GATLayer(nn.Module):
    def __init__(self, output_dim, nheads, dropout=0):
        super(GATLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(nheads, output_dim, output_dim//4, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_in = output_dim, dhid = 2*output_dim, dropout=dropout)
    
    def forward(self, x, adj):
        x = self.slf_attn(x, adj)
        x = self.pos_ffn(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, dropout=0, query_context=False):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k  #d_k
        self.query_context = query_context
        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k)
        self.w_vs = nn.Linear(d_model, n_head * self.d_k)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        
        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_k, 0.5), attn_dropout=dropout)
        self.ln= nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * self.d_k, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.gate = GRUGate(d_model)
            
    def forward(self, x, adj):
        residual = x
        x = self.ln(x)

        q = torch.mean(x, dim = 1, keepdim=True) if self.query_context else x
        k = x
        v = x

        d_k, n_head = self.d_k, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_k)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_k) # (n*b) x lv x dv

        if self.query_context: # let the attention look at all the substations
            adj = torch.ones(sz_b*n_head, q.shape[1], k.shape[1])
        else: 
            adj = adj.unsqueeze(1).repeat(1, n_head, 1, 1).reshape(-1, len_q, len_q)
        output = self.attention(q, k, v, adj)
       
        output = output.view(n_head, sz_b, len_q, d_k)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        # print("output here", output)
        # return x

        output = F.relu(self.dropout(self.fc(output)))
        # print("post relu", output)
        if not self.query_context:
            output = self.gate(residual, output)
        # print("post gate", output)
        return output  

class DecoderAttention(nn.Module):

    def __init__(self,temperature ,clip_constant = 10):
        super().__init__()
        self.temperature = temperature
        self.clip_constant = clip_constant

    def forward(self, q, k, adj):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.clip_constant * F.tanh(attn)
        # attn = attn.masked_fill(adj==0, -np.inf)
        #print("attn", attn.shape)
      
        return attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, dhid, dropout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, dhid)
        self.w_2 = nn.Linear(dhid, d_in)
        self.ln = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.gate = GRUGate(d_in)
            
    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = F.relu(self.w_2(F.relu((self.w_1(x)))))
        return self.gate(residual, x)
        

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, adj):
    
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        # print("INSIDE DOT PRODUCT ATTENTION")
        # print("attn", attn.shape)
        # print("adj", adj.shape)
        # print("attn device", attn.device)
        # print("adj device", adj.device)
        adj = adj.to(attn.device)
        attn = attn.masked_fill(adj==0, -np.inf)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        # print("after matmul", output)
        return output


class GRUGate(nn.Module):    
    def __init__(self, d):
        super(GRUGate,self).__init__()

        self.linear_w_r = nn.Linear(d, d, bias=False)
        self.linear_u_r = nn.Linear(d, d, bias=False)
        self.linear_w_z = nn.Linear(d, d)
        self.linear_u_z = nn.Linear(d, d, bias=False)
        self.linear_w_g = nn.Linear(d, d, bias=False)
        self.linear_u_g = nn.Linear(d, d, bias=False)

        self.init_bias()

    def init_bias(self):
        with torch.no_grad():
            self.linear_w_z.bias.fill_(-2)

    def forward(self, x, y):
        z = torch.sigmoid(self.linear_w_z(y) + self.linear_u_z(x))
        r = torch.sigmoid(self.linear_w_r(y) + self.linear_u_r(x))
        h_hat = torch.tanh(self.linear_w_g(y) + self.linear_u_g(r*x))
        return (1.- z) * x + z * h_hat