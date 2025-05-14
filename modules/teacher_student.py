import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherStudentLoss(nn.Module):
    """
    Teacher-Student 对齐损失
    -------------------------
    用于将 Student（Raw Encoder）特征和注意力分布对齐到 Teacher（GT Encoder）的清晰表征。

    公式:
      L_feat_i = ||F_raw_i - stopgrad(F_gt_i)||_1
      L_attn_i = KL(softmax(A_raw_i) || stopgrad(softmax(A_gt_i)))

    Args:
      feat_weight (float): 每级特征对齐的权重
      attn_weight (float): 每级注意力对齐的权重
    """
    def __init__(self, feat_weight: float = 1.0, attn_weight: float = 1.0):
        super().__init__()
        self.feat_weight = feat_weight
        self.attn_weight = attn_weight
        self.l1_loss = nn.L1Loss()
        # KLDivLoss expects input as log-probs, target as probs
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self,
                student_feats: list,
                teacher_feats: list,
                student_attns: list = None,
                teacher_attns: list = None) -> torch.Tensor:
        """
        Args:
          student_feats: list of Tensors [B,C,H_i,W_i]
          teacher_feats: same shapes, teacher_outputs.detach()
          student_attns: list of attention maps [B, heads, N, N] or None
          teacher_attns: same shapes, teacher maps detached or None

        Returns:
          loss (Tensor): scalar alignment loss
        """
        loss = 0.0
        # 特征对齐
        for sf, tf in zip(student_feats, teacher_feats):
            # 检查特征大小是否匹配，如果不匹配则进行调整
            if sf.shape[2:] != tf.shape[2:]:
                # 选择较小的尺寸作为目标尺寸
                target_size = (
                    min(sf.shape[2], tf.shape[2]),
                    min(sf.shape[3], tf.shape[3])
                )
                # 如果学生特征较大，则对其下采样
                if sf.shape[2] > target_size[0] or sf.shape[3] > target_size[1]:
                    sf = F.interpolate(sf, size=target_size, mode='bilinear', align_corners=False)
                # 如果教师特征较大，则对其下采样
                if tf.shape[2] > target_size[0] or tf.shape[3] > target_size[1]:
                    tf = F.interpolate(tf, size=target_size, mode='bilinear', align_corners=False)
            
            loss += self.feat_weight * self.l1_loss(sf, tf.detach())
        
        # 注意力对齐 - 只有当两个注意力映射都不为None时才进行
        if student_attns is not None and teacher_attns is not None:
            for sa, ta in zip(student_attns, teacher_attns):
                # 将注意力图展平为概率分布
                B, H, N, _ = sa.shape
                sa_flat = sa.view(B*H, N, N)
                ta_flat = ta.detach().view(B*H, N, N)
                # 计算 log-probs 和 probs
                sa_log = F.log_softmax(sa_flat, dim=-1)
                ta_prob = F.softmax(ta_flat, dim=-1)
                # KLDivLoss: input=log_prob, target=prob
                loss += self.attn_weight * self.kl_loss(sa_log, ta_prob)
        
        return loss

# 示例用法:
# tsloss = TeacherStudentLoss(feat_weight=0.5, attn_weight=0.2)
# loss_align = tsloss(student_feats, teacher_feats, student_attns, teacher_attns)



