import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # KxM

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

"""
Attention-based Deep Multiple Instance Learning (ABMIL) for Camelyon16.

Args:
    in_features (int): Input feature vector dimension (1024 for UNI, 2048 for ResNet50).
    M (int): Latent projection dimension for bag representation.
    L (int): Dimension of attention sub-space.
    attention_branches (int): Number of attention heads (default: 1).
"""
class Attention(nn.Module):
    def __init__(self, in_features=1024, M=500, L=128, attention_branches=1):
        super(Attention, self).__init__()
        self.in_features = in_features
        self.M = M # Patch embedding size
        self.L = L # Attention hidden size
        self.ATTENTION_BRANCHES = attention_branches

        # Feature Projection (Replaces CNN Part 1 & 2 since features are pre-extracted)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.in_features, self.M),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # Attention Mechanism V and w
        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),                       # Matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES)       # Matrix/Vector w
        )

        # Slide-Level Binary Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Unpack DataLoader batch dimension if present: (1, N, 1024) -> (N, 1024)
        if x.dim() == 3:
            x = x.squeeze(0)

        # 1. Project N patch features into M-dimensional latent space
        H = self.feature_extractor(x)  # Shape: (N, M)

        # 2. Compute Attention Weights over N patches
        A = self.attention(H)           # Shape: (N, ATTENTION_BRANCHES)
        A = torch.transpose(A, 1, 0)     # Shape: (ATTENTION_BRANCHES, N)
        A = F.softmax(A, dim=1)         # Softmax normalization over N patches

        # 3. Aggregate Patch Embeddings into Single Slide Embedding Z
        Z = torch.mm(A, H)              # Shape: (ATTENTION_BRANCHES, M)

        # 4. Slide Classification
        Y_prob = self.classifier(Z)     # Shape: (1, 1)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        """Calculates classification error for a given slide bag."""
        Y = Y.float().view(-1, 1)
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        """Calculates negative log likelihood (Binary Cross Entropy) loss."""
        Y = Y.float().view(-1, 1)
        Y_prob, _, A = self.forward(X)
        
        # Clamp probability for numerical stability
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5)
        
        # Negative log bernoulli loss
        neg_log_likelihood = -1.0 * (Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob))
        
        return neg_log_likelihood, A

"""
Gated attention-based MIL model for pre-extracted Camelyon16 patches.
Each input bag contains N patch feature vectors of size ``in_features``.
The gated attention score is ``w^T(tanh(Vh) * sigmoid(Uh))`` for every
patch, followed by a softmax across the N patches in that slide.
"""
class GatedAttention(nn.Module):
    def __init__(self, in_features=1024, M=500, L=128, attention_branches=1):
        super(GatedAttention, self).__init__()
        self.in_features = in_features
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = attention_branches

        # Camelyon16 supplies pre-extracted patch features, so no CNN is used.
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.in_features, self.M),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # Gated-attention branches from Ilse et al. (2018).
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # DataLoader adds batch dimension: (1, N, in_features) -> (N, in_features).
        if x.dim() == 3:
            x = x.squeeze(0)

        H = self.feature_extractor(x)              # (N, M)
        A_V = self.attention_V(H)                  # (N, L)
        A_U = self.attention_U(H)                  # (N, L)
        A = self.attention_w(A_V * A_U)            # (N, attention_branches)
        A = torch.transpose(A, 1, 0)               # (attention_branches, N)
        A = F.softmax(A, dim=1)                    # normalize over patches

        Z = torch.mm(A, H)                         # (attention_branches, M)
        Y_prob = self.classifier(Z)                # (1, 1) for one attention branch
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float().view(-1, 1)
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float().view(-1, 1)
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (
            Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob)
        )
        return neg_log_likelihood, A
