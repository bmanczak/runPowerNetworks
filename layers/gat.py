import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn


gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38

class GATModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GAT", dp_rate=0.1, heads = 4,
                    out_heads = None, **kwargs):
        """

        Stacks GAT layers layers.
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        # heads = kwargs.get("heads", 1)
        # print("heads hello", heads)

        layers = []
        # in_channels, out_channels = c_in, c_hidden#*heads
        out_heads = heads if out_heads is None else out_heads
        embed_dim = c_hidden//heads
        in_channels, out_channels = c_in, embed_dim

        assert c_hidden%heads == 0, "Hidden dimension must be divisible by number of heads"
        assert c_out%out_heads == 0, "Output dimension must be divisible by number of heads"

        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          heads = heads,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = embed_dim*heads
        # out_channels = c_out//heads
        layers += [gnn_layer(in_channels=embed_dim*heads,
                             out_channels=c_out//out_heads,
                             heads = out_heads,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for layer_num, l in enumerate(self.layers):
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
            # print(layer_num, x.shape)
        return x

class DecoderAttention(nn.Module):

    def __init__(self, d_q, d_k, n_heads = 8,
                 d_out = None, clip_constant = 10.0):
        """
        Performs one step of the decoding as presented in the paper
        "Attention, learn to solve routing problems" (https://arxiv.org/abs/1803.08475)

        Parameters:
        -----------

        d_q: int
            The dimensionality of the query vector (substation context).
        d_k: int
            The dimensionality of the key vector (node embeddings).
        n_heads: int
            The number of heads to use in the multi-head attention mechanism
            for updating the context vector.
        d_out : int
            The dimensionality of the projection matrix for the query and key.
            By default, this is set to d_k.
        clip_constant: float
            The clipping constant used to clip the attention values after the 
            tanh activation.
        """
        super().__init__()

        

        self.d_q = d_q  #d_k
        self.d_k = d_k  # set default output as the 
        if d_out is None:
            self.d_out = d_k
        else:
            self.d_out = d_out

        # Multi-head attention to produce the new context vector
        self.mhn = nn.MultiheadAttention(embed_dim = self.d_out, num_heads= n_heads, 
                        kdim= self.d_k, vdim= self.d_k, batch_first=True)

        self.w_qs = nn.Linear(d_q, self.d_out)
        self.w_ks = nn.Linear(d_k,self.d_out)

        self.temperature = np.power(self.d_out, 0.5 )
        self.clip_constant = clip_constant

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (self.d_out + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (self.d_out + self.d_k)))
    
            
    def forward(self, q, k, mask = None):
    
        
        if mask is not None:
            assert mask.shape[0] == k.shape[0] and mask.shape[1] == k.shape[1] and mask.ndim == 2, \
                 "Mask should have shape [BATCH, MAX_NUM_ELEMs]"

        new_context, _ = self.mhn(q, k, k, key_padding_mask = mask)

        q = self.w_qs(new_context)
        k = self.w_ks(k)
        attn = torch.bmm(new_context, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.clip_constant * torch.tanh(attn)

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), FLOAT_MIN)
            
        return attn
        
        
        