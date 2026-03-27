# Copyright (c) ModelScope Contributors. All rights reserved.
# Implementation of SDFT (Self-Distillation Fine-Tuning) from:
# "Self-Distillation Enables Continual Learning" (arXiv:2601.19897)

import torch
import torch.nn.functional as F
from typing import Optional, Union

import torch.nn as nn
from transformers import PreTrainedModel

from swift.utils import get_logger
from .gkd_trainer import GKDTrainer

logger = get_logger()


class SDFTTrainer(GKDTrainer):
    """Self-Distillation Fine-Tuning (SDFT) Trainer.

    SDFT uses the same model as both teacher and student:
    - Student: model conditioned on the query only, π(·|x)
    - Teacher: model conditioned on query + demonstration context, π(·|x, c)

    The training minimizes the KL divergence between student and teacher on
    on-policy (student-generated) completions. The `sdft_alpha` parameter
    controls the KL variant:
    - sdft_alpha=0: Forward KL, KL(Teacher || Student)
    - sdft_alpha=1: Reverse KL, KL(Student || Teacher) [paper default]
    - 0 < sdft_alpha < 1: Generalized Jensen-Shannon divergence

    Data format: Each example should have a `teacher_prompt` field containing
    the demonstration-augmented prompt. The student sees the original prompt,
    while the teacher sees the teacher_prompt (which includes demonstrations).

    Reference: https://arxiv.org/abs/2601.19897
    """

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        args = kwargs.get('args')

        # SDFT always uses on-policy generation (lambda=1.0)
        if args is not None:
            args.lmbda = 1.0

        super().__init__(model, *_args, **kwargs)

        self.sdft_alpha = getattr(args, 'sdft_alpha', 1.0)

        logger.info(f'SDFT initialized with sdft_alpha={self.sdft_alpha} '
                     f'({"reverse KL" if self.sdft_alpha == 1 else "forward KL" if self.sdft_alpha == 0 else "GJS"}), '
                     f'beta(temperature)={self.beta}, temperature={self.temperature}')

    def _compute_jsd_loss(self, outputs_student, outputs_teacher, inputs, opsd_teacher_inputs):
        """Override GKD's JSD loss to use SDFT's KL divergence.

        Uses sdft_alpha to control the KL variant instead of GKD's beta for JSD.
        The beta parameter from GKDConfig is used as temperature here.
        """
        if opsd_teacher_inputs is not None:
            student_shifted = torch.roll(inputs['labels'], shifts=-1, dims=1)
            teacher_shifted = torch.roll(opsd_teacher_inputs['labels'], shifts=-1, dims=1)
            student_mask = student_shifted != -100
            teacher_mask = teacher_shifted != -100
            assert student_mask.sum() == teacher_mask.sum(), (
                f'SDFT label count mismatch: student={student_mask.sum().item()}, '
                f'teacher={teacher_mask.sum().item()}. '
                'Student and teacher must share the same response tokens.')
            student_logits = outputs_student.logits[student_mask]
            teacher_logits = outputs_teacher.logits[teacher_mask]
        else:
            shifted_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)
            mask = shifted_labels != -100
            student_logits = outputs_student.logits[mask]
            teacher_logits = outputs_teacher.logits[mask]

        return self._sdft_kl_loss(student_logits, teacher_logits)

    def _sdft_kl_loss(self, student_logits, teacher_logits):
        """Compute SDFT loss based on sdft_alpha.

        Args:
            student_logits: [num_tokens, vocab_size] logits from student model
            teacher_logits: [num_tokens, vocab_size] logits from teacher model
        """
        num_valid = student_logits.size(0)
        if num_valid == 0:
            return student_logits.new_zeros(())

        # Apply temperature
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature

        # Handle vocab size mismatch (student may have different vocab than teacher)
        stu_dim = student_logits.shape[-1]
        tea_dim = teacher_logits.shape[-1]
        if stu_dim != tea_dim:
            max_dim = max(stu_dim, tea_dim)
            if stu_dim < max_dim:
                student_logits = F.pad(student_logits, (0, max_dim - stu_dim), value=float('-inf'))
            if tea_dim < max_dim:
                teacher_logits = F.pad(teacher_logits, (0, max_dim - tea_dim), value=float('-inf'))

        s_log_probs = F.log_softmax(student_logits, dim=-1)
        t_log_probs = F.log_softmax(teacher_logits, dim=-1)

        alpha = self.sdft_alpha

        if alpha == 0:
            # Forward KL: KL(Teacher || Student) = sum_y Q(y) * log(Q(y)/P(y))
            loss = F.kl_div(s_log_probs, t_log_probs, reduction='batchmean', log_target=True)
        elif alpha == 1:
            # Reverse KL: KL(Student || Teacher) = sum_y P(y) * log(P(y)/Q(y))
            loss = F.kl_div(t_log_probs, s_log_probs, reduction='batchmean', log_target=True)
        else:
            # Generalized Jensen-Shannon divergence
            alpha_t = torch.tensor(alpha, dtype=student_logits.dtype, device=student_logits.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([s_log_probs + torch.log1p(-alpha_t), t_log_probs + torch.log(alpha_t)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, t_log_probs, reduction='sum', log_target=True)
            kl_student = F.kl_div(mixture_log_probs, s_log_probs, reduction='sum', log_target=True)
            loss = (alpha_t * kl_teacher + (1 - alpha_t) * kl_student) / num_valid

        return loss
