import torch.nn as nn
import torch.nn.functional as F

from layers.graph_attention_layer import GATLayer
from models.utils import pool_per_substation


class SubstationModel(nn.Module):
    def __init__(self, num_features, hidden_dim, nheads, sub_id_to_elem_id, num_layers = 6, dropout=0):
        """
        Constructs the Encoder.

        Args:
            num_features (int): Number of features of each node.
            hidden_dim ([int]): Number of features of each node after transformation.
            nheads (int): number of attention heads.
            num_layers (int, optional): Number of GAT layers. Defaults to 6.
            dropout (int, optional): Dropout probability. Defaults to 0.
        """
        super(SubstationModel, self).__init__()
        self.linear = nn.Linear(num_features, hidden_dim)
        self.gat_layers = nn.ModuleList([
                GATLayer(hidden_dim, nheads, dropout=dropout) for _ in range(num_layers)])

        self.classify_substations = nn.Linear(hidden_dim, 1) # a binary classifier on each substation
        self.sub_id_to_elem_id = sub_id_to_elem_id
        

    def forward(self, x, adj):

        x = self.linear(x)
        for layer in self.gat_layers:
            node_embeddings = layer(x, adj)
        substation_embeddings = pool_per_substation(node_embeddings, self.sub_id_to_elem_id)
        
        substation_prob_distr = F.softmax(self.classify_substations(substation_embeddings), dim = 1)

        return substation_prob_distr, node_embeddings, substation_embeddings