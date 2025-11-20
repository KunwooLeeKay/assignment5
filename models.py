import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()

        # Shared MLP over points: 3 → 64 → 128 → 1024
        # Input will be (B, 3, N) when passed to these convs
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.bn1   = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2   = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn3   = nn.BatchNorm1d(1024)

        # Classification head: 1024 → 512 → 256 → num_classes
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, num_classes)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        B, N, _ = points.shape

        # Convert to (B, 3, N) for Conv1d
        x = points.permute(0, 2, 1)

        # Per point feature extraction
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))   # (B, 1024, N)

        # Max pool over points to get global feature (B, 1024)
        x = torch.max(x, dim=2)[0]

        # Classification head
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)

        x = self.fc3(x)  # logits (B, num_classes)

        return x



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        
        # Encoder: shared MLP over points
        # Input to convs will be (B, 3, N)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.bn1   = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2   = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn3   = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)
        self.bn4   = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv1d(512, 1024, kernel_size=1)
        self.bn5   = nn.BatchNorm1d(1024)

        # Decoder: per point segmentation head
        # We will concatenate local point features (128) with global (1024)
        in_channels = 128 + 1024

        self.conv6 = nn.Conv1d(in_channels, 512, kernel_size=1)
        self.bn6   = nn.BatchNorm1d(512)

        self.conv7 = nn.Conv1d(512, 256, kernel_size=1)
        self.bn7   = nn.BatchNorm1d(256)

        self.conv8 = nn.Conv1d(256, 128, kernel_size=1)
        self.bn8   = nn.BatchNorm1d(128)

        self.conv9 = nn.Conv1d(128, num_seg_classes, kernel_size=1)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        B, N, _ = points.shape

        # (B, 3, N)
        x = points.permute(0, 2, 1)

        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))      # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))      # (B, 128, N)
        pointfeat = x                            # save local features (B, 128, N)

        x = F.relu(self.bn3(self.conv3(x)))      # (B, 256, N)
        x = F.relu(self.bn4(self.conv4(x)))      # (B, 512, N)
        x = F.relu(self.bn5(self.conv5(x)))      # (B, 1024, N)

        # Global feature: max pool over points
        global_feat = torch.max(x, dim=2, keepdim=True)[0]    # (B, 1024, 1)

        # Tile global feature to per point
        global_feat = global_feat.repeat(1, 1, N)             # (B, 1024, N)

        # Concatenate local and global features
        x = torch.cat([pointfeat, global_feat], dim=1)        # (B, 1152, N)

        # Decoder head
        x = F.relu(self.bn6(self.conv6(x)))                   # (B, 512, N)
        x = self.dropout(x)
        x = F.relu(self.bn7(self.conv7(x)))                   # (B, 256, N)
        x = F.relu(self.bn8(self.conv8(x)))                   # (B, 128, N)

        x = self.conv9(x)                                     # (B, num_seg_classes, N)

        # Return as (B, N, num_seg_classes)
        x = x.permute(0, 2, 1)

        return x


# %%


def knn(x, k):
    """
    x: (B, N, C) features
    return: idx (B, N, k) indices of k nearest neighbors for each point
    """
    # Pairwise distances in feature space
    dist = torch.cdist(x, x)                 # (B, N, N)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]  # (B, N, k)
    return idx


def index_points(points, idx):
    """
    points: (B, N, C)
    idx:    (B, N, k)
    return: (B, N, k, C)
    """
    B, N, C = points.shape
    device = points.device
    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand_as(idx)
    # Advanced indexing, gathers neighbors for each point
    neighbors = points[batch_indices, idx, :]   # (B, N, k, C)
    return neighbors


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: (B, N, C_in)
        returns: (B, C_out, N)
        """
        B, N, C = x.shape

        # kNN in feature space
        idx = knn(x, self.k)                     # (B, N, k)
        neighbors = index_points(x, idx)         # (B, N, k, C)

        # Central point features
        x_i = x.unsqueeze(2)                     # (B, N, 1, C)

        # Edge features: [x_i, x_j - x_i]
        edge_feat = torch.cat(
            [x_i.repeat(1, 1, self.k, 1), neighbors - x_i],
            dim=-1
        )                                        # (B, N, k, 2C)

        # For Conv2d: (B, 2C, N, k)
        edge_feat = edge_feat.permute(0, 3, 1, 2)

        out = self.mlp(edge_feat)                # (B, C_out, N, k)
        out = torch.max(out, dim=3)[0]           # max over neighbors -> (B, C_out, N)

        return out


# ------ DGCNN CLASSIFICATION MODEL ------
class cls_model_DGCNN(nn.Module):
    def __init__(self, num_classes=3, k=20):
        super(cls_model_DGCNN, self).__init__()
        self.k = k

        # EdgeConv layers
        self.ec1 = EdgeConvBlock(3,   64, k=self.k)
        self.ec2 = EdgeConvBlock(64,  64, k=self.k)
        self.ec3 = EdgeConvBlock(64, 128, k=self.k)
        self.ec4 = EdgeConvBlock(128, 256, k=self.k)

        # After concatenation: 64 + 64 + 128 + 256 = 512
        self.lin1 = nn.Linear(512, 256)
        self.bn1  = nn.BatchNorm1d(256)
        self.lin2 = nn.Linear(256, 128)
        self.bn2  = nn.BatchNorm1d(128)
        self.lin3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, points):
        """
        points: (B, N, 3)
        output: (B, num_classes)
        """
        B, N, _ = points.shape

        # First EdgeConv takes coordinates as features
        x = points

        x1 = self.ec1(x)                         # (B, 64, N)
        x2 = self.ec2(x1.permute(0, 2, 1))       # (B, 64, N)
        x3 = self.ec3(x2.permute(0, 2, 1))       # (B, 128, N)
        x4 = self.ec4(x3.permute(0, 2, 1))       # (B, 256, N)

        # Concatenate along channels
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)  # (B, 512, N)

        # Global max pooling over points
        x_global = torch.max(x_cat, dim=2)[0]       # (B, 512)

        # MLP head
        x = F.relu(self.bn1(self.lin1(x_global)))   # (B, 256)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.lin2(x)))          # (B, 128)
        x = self.lin3(x)                            # (B, num_classes)

        return x


# ------ DGCNN SEGMENTATION MODEL ------
class seg_model_DGCNN(nn.Module):
    def __init__(self, num_seg_classes=6, k=20):
        super(seg_model_DGCNN, self).__init__()
        self.k = k

        # Same backbone
        self.ec1 = EdgeConvBlock(3,   64, k=self.k)
        self.ec2 = EdgeConvBlock(64,  64, k=self.k)
        self.ec3 = EdgeConvBlock(64, 128, k=self.k)
        self.ec4 = EdgeConvBlock(128, 256, k=self.k)

        # After concat: 64 + 64 + 128 + 256 = 512
        # We will also concatenate global feature (512) -> 1024 total channels
        self.conv1 = nn.Conv1d(1024, 512, kernel_size=1)
        self.bn1   = nn.BatchNorm1d(512)

        self.conv2 = nn.Conv1d(512, 256, kernel_size=1)
        self.bn2   = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, num_seg_classes, kernel_size=1)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, points):
        """
        points: (B, N, 3)
        output: (B, N, num_seg_classes)
        """
        B, N, _ = points.shape

        x = points

        x1 = self.ec1(x)                         # (B, 64, N)
        x2 = self.ec2(x1.permute(0, 2, 1))       # (B, 64, N)
        x3 = self.ec3(x2.permute(0, 2, 1))       # (B, 128, N)
        x4 = self.ec4(x3.permute(0, 2, 1))       # (B, 256, N)

        x_cat = torch.cat([x1, x2, x3, x4], dim=1)   # (B, 512, N)

        # Global feature
        x_global = torch.max(x_cat, dim=2, keepdim=True)[0]   # (B, 512, 1)
        x_global = x_global.repeat(1, 1, N)                   # (B, 512, N)

        # Concatenate local and global features
        x_full = torch.cat([x_cat, x_global], dim=1)          # (B, 1024, N)

        # Segmentation head
        x = F.relu(self.bn1(self.conv1(x_full)))              # (B, 512, N)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))                   # (B, 256, N)
        x = self.conv3(x)                                     # (B, num_seg_classes, N)

        # Return (B, N, num_seg_classes)
        x = x.permute(0, 2, 1)

        return x
