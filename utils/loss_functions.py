import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, loss_name, class_weights, alpha, gamma, loss_reduction):
        super(CustomLoss, self).__init__()
        self.loss_name = loss_name.lower()
        self.class_weights = class_weights  # not used for capsule loss
        self.alpha = alpha
        self.gamma = gamma
        self.loss_reduction = loss_reduction.lower()
        print("🔹 Initializing Loss Fun:")
        print(f"   • Type         : {self.loss_name.upper()}")
        print(f"   • Class weight: {self.class_weights}")
        print(f"   • Reduction     : {self.loss_reduction}")

    def forward(self, outputs, targets):
        """
        outputs: [B, num_classes, capsule_dim] for 'capsule_margin'
                 OR [B, num_classes] for others
        targets: class indices [B]
        """

        if self.loss_name == "capsule_margin":
            return self.capsule_margin_loss(outputs, targets)

        # Convert class indices to logits-compatible flow
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')  # [B]

        if self.loss_name == 'focal':
            pt = torch.exp(-ce_loss)
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(targets.device)
                alpha_t = alpha[targets]
            else:
                alpha_t = self.alpha
            ce_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss

        elif self.loss_name == 'class_weight' and self.class_weights is not None:
            probs = F.softmax(outputs, dim=1)
            weight_per_sample = torch.sum(probs * self.class_weights.to(outputs.device), dim=1)
            ce_loss = ce_loss * weight_per_sample


        if self.loss_reduction == 'mean':
            return ce_loss.mean()
        elif self.loss_reduction == 'sum':
            return ce_loss.sum()

        return ce_loss

    def capsule_margin_loss(self, capsule_output, targets, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        """
        capsule_output: [B, num_classes, capsule_dim]
        targets: [B] - class indices
        """
        v_lengths = torch.norm(capsule_output, dim=-1)  # [B, num_classes]
        y_true = F.one_hot(targets, num_classes=v_lengths.size(1)).float()

        left = F.relu(m_plus - v_lengths) ** 2
        right = F.relu(v_lengths - m_minus) ** 2

        loss = y_true * left + lambda_ * (1.0 - y_true) * right
        loss = loss.sum(dim=1)  # per-sample total loss

        if self.loss_reduction == 'mean':
            return loss.mean()
        elif self.loss_reduction == 'sum':
            return loss.sum()
        return loss  # [B]
