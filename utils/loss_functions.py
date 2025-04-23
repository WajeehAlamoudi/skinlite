import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsuleLoss(nn.Module):
    def __init__(self, pos_class=0.9,
                 neg_class=0.1, penalty=0.5,
                 reduction='sum', class_weights=None):
        super(CapsuleLoss, self).__init__()
        self.pos_class = pos_class
        self.neg_class = neg_class
        self.penalty = penalty
        self.reduction = reduction
        self.class_weights = class_weights
        self.reconstruction_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, x, labels, images, reconstructions):
        batch_size = x.size(0)

        # Capsule vector lengths = class prediction confidences
        v_c = torch.sqrt((x ** 2).sum(dim=-1)).squeeze(-1)

        # Margin loss
        left = F.relu(self.pos_class - v_c)
        # .view(batch_size, -1))
        right = F.relu(v_c - self.neg_class)
        # .view(batch_size, -1)
        margin_loss = labels * left + self.penalty * (1.0 - labels) * right
        margin_loss = margin_loss.sum()

        # Reconstruction loss
        images = images.view(reconstructions.size(0), -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        if self.reduction == 'sum':
            total_loss = (margin_loss + 0.0005 * reconstruction_loss) / batch_size
        else:  # assume mean
            total_loss = margin_loss / batch_size + 0.0005 * reconstruction_loss

        return total_loss


"""
# margin_loss: [B, num_classes]
margin_loss = labels * left + self.penalty * (1.0 - labels) * right

if self.class_weights is not None:
    # Apply class weights: [B, C] * [C] = [B, C]
    margin_loss = margin_loss * self.class_weights.to(margin_loss.device)

margin_loss = margin_loss.sum()

"""