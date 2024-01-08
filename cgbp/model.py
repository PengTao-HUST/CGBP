import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, SAGEConv
import torch.nn.functional as F


def sliding_window(seq, window_size=2):
    # This function takes in a sequence and a window size and returns a list of windows from the sequence
    return [seq[i:i+window_size] for i in range(len(seq)-window_size+1)]


class GCN_my(nn.Module):
    # This class defines a graph convolutional network
    def __init__(
            self,
            dim_list,
            dropout=0.0,
            dscale=1.,
            random_pick=True,
            spe_idx=0,
            batch_norm=True,
            last_sigmoid=True):
        """
        """
        super(GCN_my, self).__init__()
        # Create a list of modules for the layers of the graph convolutional network
        self.layers = nn.ModuleList()
        # Create a list of modules for the batch normalization layers of the graph convolutional network
        self.bn_layers = nn.ModuleList()
        # Loop through the list of dimensions and create a layer for each dimension pair
        for (in_d, out_d) in sliding_window(dim_list, 2):
            self.layers.append(GraphConv(in_d, out_d))
            # If batch normalization is enabled, create a batch normalization layer for each dimension pair
            if batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(out_d))

        self.batch_norm = batch_norm
        self.dropout_frac = dropout
        self.dscale = dscale
        self.random_pick = random_pick
        self.spe_idx = spe_idx
        self.last_sigmoid = last_sigmoid

    def forward(self, g, inputs):
        """
        """
        if self.random_pick:
            idx = torch.randperm(len(inputs))[self.spe_idx]
        else:
            idx = self.spe_idx

        outs = inputs
        hids = [torch.sigmoid(inputs * self.dscale)]  # for the embedding layer
        for i, layer in enumerate(self.layers):
            # If the layer is not the first, apply dropout
            if i != 0:
                outs = F.dropout(outs, p=self.dropout_frac)
            outs = layer(g, outs)
            hid = torch.sigmoid(outs * self.dscale)[idx]
            hids += [hid, hid]  # for weight and bias
            # If batch normalization is enabled, apply batch normalization to the output
            if self.batch_norm:
                outs = self.bn_layers[i](outs)
            # If the layer is not the last layer, apply a relu activation
            if i != len(self.layers) - 1:
                outs = torch.relu(outs)
            else:
                # If the last sigmoid is enabled, apply a sigmoid activation
                if self.last_sigmoid:
                    outs = torch.sigmoid(outs)
        # do not add chaos to BN layers
        hids += [0.] * len(self.bn_layers) * 2

        return outs, hids


class SAGE_my(nn.Module):
    # This class defines a graph sage network
    def __init__(
        self,
        dim_list,
        agg_type='mean',
        dropout=0.0,
        dscale=1.,
        random_pick=True,
        spe_idx=0,
        batch_norm=True,
        last_sigmoid=True
    ):
        super(SAGE_my, self).__init__()
        # Create a list of modules for the layers of the graph sage network
        self.layers = nn.ModuleList()
        # Create a list of modules for the batch normalization layers of the graph sage network
        self.bn_layers = nn.ModuleList()
        # Loop through the list of dimensions and create a layer for each dimension pair
        for (in_d, out_d) in sliding_window(dim_list, 2):
            self.layers.append(SAGEConv(in_d, out_d, agg_type))
            # If batch normalization is enabled, create a batch normalization layer for each dimension pair
            if batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(out_d))

        self.batch_norm = batch_norm
        self.dropout_frac = dropout
        self.dscale = dscale
        self.random_pick = random_pick
        self.spe_idx = spe_idx
        self.last_sigmoid = last_sigmoid

    def forward(self, g, inputs):
        if self.random_pick:
            idx = torch.randperm(len(inputs))[self.spe_idx]
        else:
            idx = self.spe_idx

        outs = inputs
        hids = [torch.sigmoid(inputs * self.dscale)]  # for the embedding layer
        for i, layer in enumerate(self.layers):
            # If the layer is not the first, apply dropout
            if i != 0:
                outs = F.dropout(outs, p=self.dropout_frac)
            outs = layer(g, outs)
            hid = torch.sigmoid(outs * self.dscale)[idx]
            hid2 = hid.unsqueeze(1)
            hids += [hid2, hid2, hid]  # for weight1, weight2 and bias

            # If batch normalization is enabled, apply batch normalization to the output
            if self.batch_norm:
                outs = self.bn_layers[i](outs)
            # If the layer is not the last layer, apply a relu activation
            if i != len(self.layers) - 1:
                outs = torch.relu(outs)
            else:
                # If the last sigmoid is enabled, apply a sigmoid activation
                if self.last_sigmoid:
                    outs = torch.sigmoid(outs)
        # do not add chaos to BN layers
        hids += [0.] * len(self.bn_layers) * 2

        return outs, hids
