import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, loss_name, class_weights, alpha, gamma, loss_reduction):
        super(CustomLoss, self).__init__()
        self.loss_name = loss_name.lower()
        self.class_weights = class_weights  # should be tensor [num_classes]
        self.alpha = alpha
        self.gamma = gamma
        self.loss_reduction = loss_reduction.lower

    def forward(self, outputs, targets):
        """
        outputs: logits [B, num_classes]
        targets: class indices [B]
        """
        # === Step 1: Base Cross-Entropy Loss ===
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')  # shape [B]

        # === Step 2: Focal Loss Modulation (optional) ===
        if self.loss_name == 'focal':
            pt = torch.exp(-ce_loss)

            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(targets.device)
                alpha_t = alpha[targets]  # Get alpha value for each sample
            else:
                alpha_t = self.alpha  # scalar fallback

            ce_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss

        # === Step 3: Class-Weight-Based Adjustment (optional) ===
        if self.loss_name == 'class_weight' and self.class_weights is not None:
            probs = F.softmax(outputs, dim=1)  # shape [B, num_classes]
            weight_per_sample = torch.sum(probs * self.class_weights, dim=1)  # [B]
            ce_loss = ce_loss * weight_per_sample

        # === Step 4: Final reduction ===
        if self.reduction == 'mean':
            return ce_loss.mean()
        elif self.reduction == 'sum':
            return ce_loss.sum()
        return ce_loss  # no reduction
