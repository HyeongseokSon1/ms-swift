# Copyright (c) ModelScope Contributors. All rights reserved.
import trl
from dataclasses import dataclass
from packaging import version
from transformers.utils.versions import require_version

if version.parse(trl.__version__) <= version.parse('0.28'):
    from trl import CPOConfig as HfCPOConfig
    from trl import GKDConfig as HfGKDConfig
    from trl import ORPOConfig as HfORPOConfig
    from trl import PPOConfig as HfPPOConfig
else:
    from trl.experimental.cpo import CPOConfig as HfCPOConfig
    from trl.experimental.gkd import GKDConfig as HfGKDConfig
    from trl.experimental.orpo import ORPOConfig as HfORPOConfig
    from trl.experimental.ppo import PPOConfig as HfPPOConfig

from trl import DPOConfig as HfDPOConfig
from trl import GRPOConfig as HfGRPOConfig
from trl import KTOConfig as HfKTOConfig
from trl import RewardConfig as HfRewardConfig
from typing import Optional

from swift.trainers import TrainArgumentsMixin
from .args_mixin import GRPOArgumentsMixin, RolloutTrainerArgumentsMixin


@dataclass
class DPOConfig(TrainArgumentsMixin, HfDPOConfig):
    ld_alpha: Optional[float] = None  # compat trl==0.15
    # Fields removed in trl 0.29, kept here for backward compatibility
    rpo_alpha: Optional[float] = None
    ref_adapter_name: Optional[str] = None
    reference_free: Optional[bool] = None

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfDPOConfig.__post_init__(self)


@dataclass
class CPOConfig(TrainArgumentsMixin, HfCPOConfig):

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfCPOConfig.__post_init__(self)


@dataclass
class ORPOConfig(TrainArgumentsMixin, HfORPOConfig):

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfORPOConfig.__post_init__(self)


@dataclass
class KTOConfig(TrainArgumentsMixin, HfKTOConfig):

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfKTOConfig.__post_init__(self)


@dataclass
class RewardConfig(TrainArgumentsMixin, HfRewardConfig):

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfRewardConfig.__post_init__(self)


@dataclass
class PPOConfig(TrainArgumentsMixin, HfPPOConfig):

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfPPOConfig.__post_init__(self)


@dataclass
class GKDConfig(RolloutTrainerArgumentsMixin, TrainArgumentsMixin, HfGKDConfig):
    sft_alpha: float = 0

    offload_teacher_model: bool = False
    max_completion_length: int = 512
    log_completions: bool = False

    def __post_init__(self):
        RolloutTrainerArgumentsMixin.__post_init__(self)
        TrainArgumentsMixin.__post_init__(self)
        HfGKDConfig.__post_init__(self)
        self._init_generation_batch_params()


@dataclass
class SDFTConfig(GKDConfig):
    """Config for Self-Distillation Fine-Tuning (SDFT).

    SDFT uses the same model as both teacher and student. The teacher sees
    demonstration-augmented prompts while the student sees the original prompt.

    Args:
        sdft_alpha: Controls the KL divergence variant.
            0 = Forward KL: KL(Teacher || Student)
            1 = Reverse KL: KL(Student || Teacher) [paper default]
            0 < alpha < 1 = Generalized Jensen-Shannon
        ema_decay: EMA decay rate for the teacher model.
            θ_ema ← ema_decay * θ_ema + (1 - ema_decay) * θ_student
            Paper explores (1-decay) ∈ {0.01, 0.02, 0.05}, i.e. decay ∈ {0.99, 0.98, 0.95}.
            Set to 0 to disable EMA (falls back to self-distillation via disable_adapter).
    """
    sdft_alpha: float = 1.0
    sdft_demo_prefix: str = 'Reference answer: '
    ema_decay: float = 0.99

    def __post_init__(self):
        # SDFT always uses on-policy generation
        self.lmbda = 1.0
        GKDConfig.__post_init__(self)


@dataclass
class GRPOConfig(GRPOArgumentsMixin, TrainArgumentsMixin, HfGRPOConfig):

    def __post_init__(self):
        require_version('trl>=0.26')
        GRPOArgumentsMixin.__post_init__(self)
        TrainArgumentsMixin.__post_init__(self)
        HfGRPOConfig.__post_init__(self)
        if self.vllm_reasoning_parser is not None:
            raise ValueError('vllm_reasoning_parser is not supported for GRPO Training, please unset it.')

        if self.cosine_max_len is None:
            self.cosine_max_len = self.max_completion_length

        if self.deepspeed and 'zero_optimization' in self.deepspeed and self.deepspeed['zero_optimization'][
                'stage'] == 3:
            # https://github.com/modelscope/ms-swift/issues/3237
            self.deepspeed['zero_optimization']['stage3_prefetch_bucket_size'] = 0
            self.deepspeed_plugin.hf_ds_config.config['zero_optimization']['stage3_prefetch_bucket_size'] = 0

        # https://github.com/modelscope/ms-swift/issues/3863
        self.dataloader_drop_last = True

        if self.num_generations < 2:
            raise ValueError(
                'GRPO requires at least 2 generations per prompt to calculate the advantages. You provided '
                f'{self.num_generations}, which is less than the minimum required.')
        self._init_generation_batch_params()
