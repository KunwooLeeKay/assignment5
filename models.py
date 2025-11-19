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



