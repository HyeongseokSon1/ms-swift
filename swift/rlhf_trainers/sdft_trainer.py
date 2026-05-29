# Copyright (c) ModelScope Contributors. All rights reserved.
# Implementation of SDFT (Self-Distillation Fine-Tuning) from:
# "Self-Distillation Enables Continual Learning" (arXiv:2601.19897)

import torch
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional, Union

import torch.nn as nn
from transformers import PreTrainedModel

from swift.sequence_parallel import GatherLoss, sequence_parallel
from swift.trainers import disable_gradient_checkpointing
from swift.utils import get_logger
from .gkd_trainer import DataSource, GKDTrainer

logger = get_logger()


class SDFTTrainer(GKDTrainer):
    """Self-Distillation Fine-Tuning (SDFT) Trainer.

    SDFT uses the same model as both teacher and student:
    - Student: model conditioned on the query only, π(·|x)
    - Teacher: EMA of student, conditioned on query + demonstration context, π(·|x, c)

    The teacher uses EMA weights swapped into the same model for forward pass.
    This avoids needing a separate model copy and works with DeepSpeed ZeRO-2/3
    since only the local parameter data (partitioned for ZeRO-3) is cloned.

    θ_ema ← ema_decay * θ_ema + (1 - ema_decay) * θ_student

    Reference: https://arxiv.org/abs/2601.19897
    """

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        args = kwargs.get('args')

        # SDFT always uses on-policy generation (lambda=1.0)
        if args is not None:
            args.lmbda = 1.0

        self.ema_decay = getattr(args, 'ema_decay', 0.99) if args else 0.99
        self._ema_params = None  # Initialized lazily on first training step

        super().__init__(model, *_args, **kwargs)

        self.sdft_alpha = getattr(args, 'sdft_alpha', 1.0)
        self.sdft_demo_prefix = getattr(args, 'sdft_demo_prefix', 'Reference answer: ')

        if self.template.sequence_parallel_size > 1:
            # Each rank in an SP group only contributes the gradient of its local sequence slice; the
            # framework averages gradients across the whole process group. Scaling gradient accumulation
            # by the SP world size compensates for this (mirrors GRPO). The matching loss-side correction
            # is handled by `GatherLoss` (it multiplies the backward gradient by `world_size`).
            self.args.gradient_accumulation_steps = (
                self.args.gradient_accumulation_steps * sequence_parallel.world_size)

        ema_status = f'decay={self.ema_decay}' if self.ema_decay > 0 else 'disabled (self-distillation)'
        logger.info(f'SDFT initialized with sdft_alpha={self.sdft_alpha} '
                     f'({"reverse KL" if self.sdft_alpha == 1 else "forward KL" if self.sdft_alpha == 0 else "GJS"}), '
                     f'temperature={self.temperature}, EMA {ema_status}')

    @staticmethod
    def _get_param_data(param):
        """Get stored parameter data: ds_tensor for ZeRO-3, param.data otherwise."""
        if hasattr(param, 'ds_tensor'):
            return param.ds_tensor.data
        return param.data

    @staticmethod
    def _set_param_data(param, data):
        """Set parameter data: ds_tensor for ZeRO-3, param.data otherwise."""
        if hasattr(param, 'ds_tensor'):
            param.ds_tensor.data.copy_(data)
        else:
            param.data.copy_(data)

    def _init_ema_params(self):
        """Initialize EMA parameter storage by cloning current model parameters.

        For DeepSpeed ZeRO-3, clones ds_tensor (local partition) instead of param.data
        (which is empty/size-0 when not gathered).
        """
        unwrapped = self.accelerator.unwrap_model(self.model)
        self._ema_params = {
            name: self._get_param_data(param).clone()
            for name, param in unwrapped.named_parameters()
        }
        logger.info(f'EMA parameters initialized ({len(self._ema_params)} params)')

    @torch.no_grad()
    def _update_ema_params(self):
        """Update EMA parameters: θ_ema ← decay * θ_ema + (1 - decay) * θ_student"""
        unwrapped = self.accelerator.unwrap_model(self.model)
        decay = self.ema_decay
        for name, param in unwrapped.named_parameters():
            self._ema_params[name].mul_(decay).add_(self._get_param_data(param), alpha=1.0 - decay)

    @contextmanager
    def _ema_weight_context(self):
        """Swap model weights to EMA for teacher forward, then restore student weights."""
        unwrapped = self.accelerator.unwrap_model(self.model)
        student_params = {}
        for name, param in unwrapped.named_parameters():
            student_params[name] = self._get_param_data(param).clone()
            self._set_param_data(param, self._ema_params[name])
        try:
            yield
        finally:
            for name, param in unwrapped.named_parameters():
                self._set_param_data(param, student_params[name])

    def _get_train_sampler(self, train_dataset=None):
        if self.template.sequence_parallel_size > 1:
            # Hold each generation batch for `steps_per_generation * world_size` steps so the buffered,
            # SP-gathered chunks are fully consumed before regenerating (see `GKDTrainer._prepare_inputs`).
            from trl.trainer.utils import RepeatSampler
            return RepeatSampler(
                data_source=train_dataset or self.train_dataset,
                mini_repeat_count=1,
                batch_size=self.args.generation_batch_size,
                repeat_count=self.args.steps_per_generation * sequence_parallel.world_size,
                shuffle=getattr(self, 'shuffle_dataset', True),
                seed=self.args.seed,
            )
        return super()._get_train_sampler(train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.template.sequence_parallel_size > 1:
            loss = self._compute_loss_sp(model, inputs)
            if return_outputs:
                return (loss, None)
            return loss

        if self.ema_decay <= 0 or self._ema_params is None:
            # EMA disabled or not yet initialized: use parent's compute_loss
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # --- EMA-enabled path: use self-distillation with EMA weight swap ---
        data_source = inputs.pop('_data_source', DataSource.DATASET)
        inputs.pop('_teacher_api_logprobs', None)
        inputs.pop('_teacher_api_indices', None)
        opsd_teacher_inputs = inputs.pop('_opsd_teacher_inputs', None)

        model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}

        if opsd_teacher_inputs is not None:
            teacher_fwd_inputs = {k: v for k, v in model_inputs.items()}
            teacher_fwd_inputs.update({k: v for k, v in opsd_teacher_inputs.items() if k != 'labels'})
        else:
            teacher_fwd_inputs = None

        # Student forward (current weights)
        if self.args.sft_alpha > 0:
            model_inputs['labels'] = inputs['labels']
        outputs_student = model(**model_inputs)

        # Teacher forward (EMA weights swapped in temporarily)
        t_fwd = teacher_fwd_inputs if teacher_fwd_inputs is not None else {
            k: v for k, v in model_inputs.items() if k != 'labels'
        }

        with torch.no_grad(), self._ema_weight_context(), \
                disable_gradient_checkpointing(model, self.args.gradient_checkpointing_kwargs):
            outputs_teacher = model(**t_fwd)

        loss = self._compute_jsd_loss(outputs_student, outputs_teacher, inputs, opsd_teacher_inputs)

        if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
            loss = loss + self.args.sft_alpha * outputs_student.loss

        if return_outputs:
            return (loss, outputs_student)
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Initialize EMA params on first step (model is fully on device by now)
        if self.ema_decay > 0 and self._ema_params is None:
            self._init_ema_params()

        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.ema_decay > 0 and self._ema_params is not None:
            self._update_ema_params()
        return loss

    def _build_opsd_teacher_data(self, inputs):
        """Build teacher data for SDFT.

        If `teacher_prompt` exists in the data, uses it directly (same as GKD OPSD).
        Otherwise, auto-generates the teacher prompt by appending the assistant's
        response as a demonstration to the original user query.

        Example (auto-generated):
          Student: "Solve: 2x + 3 = 7"
          Teacher: "Solve: 2x + 3 = 7\n\nReference answer: x = 2\n\nNow answer with your own response."
        """
        teacher_data = []
        for data in inputs:
            teacher_item = {k: v for k, v in data.items() if k != 'teacher_prompt'}
            messages = [dict(m) for m in data.get('messages', [])]

            if 'teacher_prompt' in data and data['teacher_prompt']:
                # Use explicit teacher_prompt
                teacher_prompt = data['teacher_prompt']
            else:
                # Auto-generate: find the last user message and assistant response
                user_content = None
                assistant_content = None
                for msg in messages:
                    if msg['role'] == 'user':
                        user_content = msg['content']
                    elif msg['role'] == 'assistant':
                        assistant_content = msg['content']

                if user_content is None or assistant_content is None:
                    return None  # Cannot build teacher data without both user and assistant

                teacher_prompt = (
                    f'{user_content}\n\n'
                    f'{self.sdft_demo_prefix}{assistant_content}\n\n'
                    f'Now answer with your own response.'
                )

            # Remove assistant message, replace last user message with teacher_prompt
            if messages and messages[-1]['role'] == 'assistant':
                messages.pop()
            for msg in reversed(messages):
                if msg['role'] == 'user':
                    msg['content'] = teacher_prompt
                    break
            teacher_item['messages'] = messages
            teacher_data.append(teacher_item)
        return teacher_data

    def _compute_jsd_loss(self, outputs_student, outputs_teacher, inputs, opsd_teacher_inputs):
        """Override GKD's JSD loss to use SDFT's KL divergence."""
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

    # ------------------------------------------------------------------
    # Sequence parallel (Ulysses) support
    # ------------------------------------------------------------------
    def _compute_loss_sp(self, model, inputs):
        """SDFT loss under sequence parallel (Ulysses, padding_free).

        The student sequence and the (demonstration-conditioned) teacher sequence have different
        lengths, so they are split independently across the SP group. We:
          1. Forward the EMA teacher under SP and gather its response-token distributions to every
             rank, in global response order (no grad).
          2. Forward the student under SP, keeping only the local logits slice (grad).
          3. Compute the per-token KL for the student response tokens this rank owns, against the
             aligned teacher distributions, then reduce with ``GatherLoss`` (which gathers the
             per-token loss across the SP group and applies the correct gradient scaling).

        Only ``sdft_alpha`` in {0, 1} (forward / reverse KL) is supported here; this is enforced in
        ``RLHFArguments._check_sdft``.
        """
        if sequence_parallel.rp_world_size > 1:
            raise NotImplementedError(
                'SDFT sequence parallel currently supports Ulysses only (ring attention / '
                'rp_world_size > 1 is not supported). This usually means num_attention_heads is not '
                'divisible by sequence_parallel_size.')

        data_source = inputs.pop('_data_source', DataSource.DATASET)
        inputs.pop('_teacher_api_logprobs', None)
        inputs.pop('_teacher_api_indices', None)
        inputs.pop('_opsd_teacher_messages', None)
        opsd_teacher_inputs = inputs.pop('_opsd_teacher_inputs', None)
        # data_source is unused under SP (sft_alpha > 0 is disallowed), kept for parity/readability.
        del data_source

        student_labels = inputs['labels']
        model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}

        # --- Teacher forward (EMA weights, no grad) -> gathered response distributions ---
        if opsd_teacher_inputs is not None:
            teacher_inputs = {k: v for k, v in model_inputs.items()}
            teacher_inputs.update({k: v for k, v in opsd_teacher_inputs.items() if k != 'labels'})
            teacher_labels = opsd_teacher_inputs['labels']
        else:
            teacher_inputs = {k: v for k, v in model_inputs.items()}
            teacher_labels = student_labels

        with torch.no_grad(), self._ema_weight_context(), \
                disable_gradient_checkpointing(model, self.args.gradient_checkpointing_kwargs):
            teacher_resp_logits = self._sp_response_logits(model, teacher_inputs, teacher_labels)

        # --- Student forward (current weights, grad) ---
        student_sp_inputs = {k: v for k, v in model_inputs.items()}
        student_sp_inputs['labels'] = student_labels
        sequence_parallel.prepare_inputs(student_sp_inputs)
        student_pos_ids = sequence_parallel.real_position_ids
        local_labels = student_sp_inputs.pop('labels')  # padded + shifted + split: [1, s_local]
        student_logits = model(**student_sp_inputs).logits  # [1, s_local, V]

        local_mask = local_labels != -100  # [1, s_local]
        n_local = int(local_mask.sum().item())

        # Align local student response tokens to the global response order used by `teacher_resp_logits`.
        # Reproduce the exact pad-then-shift order that `pad_and_split_inputs` uses for labels, so that
        # the per-rank contiguous split matches `local_labels` token-for-token.
        full_shifted = sequence_parallel.pad(student_labels, padding_value=-100, position_ids=student_pos_ids)
        full_shifted = torch.roll(full_shifted, shifts=-1, dims=-1)
        full_mask = (full_shifted != -100)[0]  # [S_s_pad]
        total_seq = full_mask.shape[0]
        chunk = total_seq // sequence_parallel.world_size
        my_start = sequence_parallel.sp_rank * chunk
        offset = int(full_mask[:my_start].sum().item())
        num_resp = int(full_mask.sum().item())

        assert teacher_resp_logits.shape[0] == num_resp, (
            f'SDFT SP response-token count mismatch: teacher={teacher_resp_logits.shape[0]}, '
            f'student={num_resp}. Student and teacher must share the same response tokens.')

        # Per-token KL for the response tokens owned by this rank, laid back into a [1, s_local] tensor.
        # Seed from `student_logits.sum(-1) * 0` (zeros) so the tensor stays connected to the autograd
        # graph even on ranks whose local slice holds no response tokens (n_local == 0); otherwise the
        # backward pass through GatherLoss would have nothing to scatter on those ranks.
        per_token = student_logits.sum(dim=-1).mul(0.0)  # [1, s_local], grad-connected zeros
        if n_local > 0:
            student_resp = student_logits[local_mask]  # [n_local, V]
            teacher_resp = teacher_resp_logits[offset:offset + n_local]  # [n_local, V]
            kl = self._sdft_kl_per_token(student_resp, teacher_resp)  # [n_local]
            per_token = per_token.clone()
            per_token[local_mask] = kl.to(per_token.dtype)

        # Gather the per-token loss across the SP group (GatherLoss handles gradient scaling), then
        # normalize by the global number of response tokens to recover the batchmean reduction.
        gathered, _ = GatherLoss.apply(per_token, local_labels, 1, student_pos_ids)
        loss = gathered.sum() / max(num_resp, 1)
        return loss

    def _sp_response_logits(self, model, fwd_inputs, full_labels):
        """Forward `fwd_inputs` under SP and return response-token logits gathered to full sequence
        order, shape [N, V]. Used for the (no-grad) teacher pass."""
        prep = {k: v for k, v in fwd_inputs.items() if k != 'labels'}
        prep['labels'] = full_labels
        sequence_parallel.prepare_inputs(prep)
        pos_ids = sequence_parallel.real_position_ids
        prep.pop('labels', None)
        local_logits = model(**prep).logits  # [1, t_local, V]
        full_logits = sequence_parallel.gather(local_logits, dim=1, position_ids=pos_ids)  # [1, S_t_pad, V]

        shifted = sequence_parallel.pad(full_labels, padding_value=-100, position_ids=pos_ids)
        shifted = torch.roll(shifted, shifts=-1, dims=-1)
        mask = shifted != -100  # [1, S_t_pad]
        return full_logits[mask]  # [N, V]

    def _sdft_kl_per_token(self, student_logits, teacher_logits):
        """Per-token KL divergence (summed over vocab) for sdft_alpha in {0, 1}.

        Returns a 1-D tensor of shape [num_tokens]. The outer caller divides the gathered sum by the
        global response-token count, which reproduces the ``batchmean`` reduction of ``_sdft_kl_loss``.
        """
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature

        # Handle vocab size mismatch (defensive; student == teacher model, so normally identical).
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

        if self.sdft_alpha == 0:
            # Forward KL: KL(Teacher || Student)
            kl = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True)
        else:
            # Reverse KL: KL(Student || Teacher)  (sdft_alpha == 1)
            kl = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True)
        return kl.sum(dim=-1)
