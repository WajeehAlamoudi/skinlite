import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, loss_name, class_weights, alpha, gamma, loss_reduction):
        super(CustomLoss, self).__init__()
        self.loss_name = loss_name.lower()
        self.class_weights = class_weights  # tensor of shape [num_classes] or None
        self.alpha = alpha
        self.gamma = gamma
        self.loss_reduction = loss_reduction.lower()

    def forward(self, outputs, targets):
        """
        outputs: logits [B, num_classes]
        targets: class indices [B]
        """
        # === Step 1: Base Cross-Entropy Loss (no weight at first) ===
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')  # [B]

        # === Step 2: Focal Loss ===
        if self.loss_name == 'focal':
            pt = torch.exp(-ce_loss)
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(targets.device)
                alpha_t = alpha[targets]
            else:
                alpha_t = self.alpha  # scalar fallback
            ce_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss

        # === Step 3: Class-weighted Loss ===
        elif self.loss_name == 'class_weight' and self.class_weights is not None:
            probs = F.softmax(outputs, dim=1)  # [B, C]
            weight_per_sample = torch.sum(probs * self.class_weights.to(outputs.device), dim=1)  # [B]
            ce_loss = ce_loss * weight_per_sample

        # === Step 4: Regular CrossEntropy (no extra modulations) ===
        elif self.loss_name in ['ce', 'cross_entropy']:
            pass  # just use raw ce_loss

        # === Step 5: Final reduction ===
        if self.loss_reduction == 'mean':
            return ce_loss.mean()
        elif self.loss_reduction == 'sum':
            return ce_loss.sum()
        return ce_loss  # no reduction
