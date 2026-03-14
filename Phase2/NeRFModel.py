import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L, embed_direction_L):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################
        self.embed_pos_L = embed_pos_L
        self.embed_dir_L = embed_direction_L

        # encoded dimension
        self.pos_dim = 3 + 6 * embed_pos_L
        self.dir_dim = 3 + 6 * self.embed_dir_L

        W = 256

        # position MLP
        self.fc1 = nn.Linear(self.pos_dim, W)
        self.fc2 = nn.Linear(W, W)
        self.fc3 = nn.Linear(W, W)
        self.fc4 = nn.Linear(W, W)

        # skip connection layer
        self.fc5 = nn.Linear(W + self.pos_dim, W)

        self.fc6 = nn.Linear(W, W)
        self.fc7 = nn.Linear(W, W)
        self.fc8 = nn.Linear(W, W)

        # density
        self.sigma = nn.Linear(W, 1)

        # feature vector
        self.feature = nn.Linear(W, W)

        # direction
        self.fc_dir = nn.Linear(W + self.dir_dim, 128)

        # RGB output
        self.rgb = nn.Linear(128, 3)

    def position_encoding(self, x, L):
        #############################
        # Implement position encoding here
        #############################
        out = [x]

        for i in range(L):
            freq = 2.0**i
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))

        y = torch.cat(out, dim=-1)

        return y

    def forward(self, pos, direction):
        #############################
        # network structure
        #############################

        # normalize viewing directions
        direction = direction / torch.norm(direction, dim=-1, keepdim=True)

        # encode inputs
        pos_enc = self.position_encoding(pos, self.embed_pos_L)
        dir_enc = self.position_encoding(direction, self.embed_dir_L)

        x = pos_enc

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # skip connection
        x = torch.cat([x, pos_enc], dim=-1)

        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))

        # density
        sigma = self.sigma(x)

        # feature
        feat = self.feature(x)

        # combine with direction
        h = torch.cat([feat, dir_enc], dim=-1)

        h = F.relu(self.fc_dir(h))

        rgb = torch.sigmoid(self.rgb(h))

        output = torch.cat([rgb, sigma], dim=-1)

        return output
