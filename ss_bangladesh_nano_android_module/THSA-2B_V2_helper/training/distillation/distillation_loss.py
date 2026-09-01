"""
Teacher-Student Knowledge Distillation Loss Formulation (Section 23.3).
Combines Cross-Entropy with KL Divergence on Soft Teacher Logits:
L_total = (1 - alpha) * L_CE + alpha * tau^2 * D_KL(Softmax(z_student/tau) || Softmax(z_teacher/tau))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.65, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, targets):
        """
        student_logits: [B * S, Vocab_Size]
        teacher_logits: [B * S, Vocab_Size]
        targets: [B * S] ground-truth token IDs
        """
        # Hard label Cross-Entropy loss
        loss_ce = self.ce_loss(student_logits, targets)
        
        if teacher_logits is not None:
            # Soft label KL Divergence loss at temperature tau
            s_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            t_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            loss_kl = self.kl_div(s_log_probs, t_probs) * (self.temperature ** 2)
            loss_total = (1.0 - self.alpha) * loss_ce + self.alpha * loss_kl
        else:
            loss_total = loss_ce
            
        return loss_total
