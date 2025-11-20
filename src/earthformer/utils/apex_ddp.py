# Find the original code and discussion at https://github.com/PyTorchLightning/pytorch-lightning/discussions/10922
# Updated to use PyTorch's native DDP instead of Apex

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.overrides.base import (
    _LightningModuleWrapperBase,
    _LightningPrecisionModuleWrapperBase,
)

def unwrap_lightning_module(wrapped_model):
    model = wrapped_model
    if isinstance(model, DistributedDataParallel):
        model = unwrap_lightning_module(model.module)
    if isinstance(
        model, (_LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase)
    ):
        model = unwrap_lightning_module(model.module)
    return model


class ApexDDPStrategy(DDPStrategy):
    """ApexDDPStrategy replacement using PyTorch's native DDP.
    
    This maintains the same interface as the original ApexDDPStrategy
    but uses PyTorch's DistributedDataParallel instead.
    """
    
    def __init__(self, *args, delay_allreduce=None, **kwargs):
        # Ignore Apex-specific parameters for backward compatibility
        super().__init__(*args, **kwargs)
    
    def _setup_model(self, model):
        # Use PyTorch's native DDP instead of Apex
        # Note: delay_allreduce is an Apex-specific parameter that doesn't exist in PyTorch DDP
        return DistributedDataParallel(
            model, 
            device_ids=self.determine_ddp_device_ids(),
            **self._ddp_kwargs
        )

    @property
    def lightning_module(self):
        return unwrap_lightning_module(self._model)


if __name__ == "__main__":
    # Updated usage with PyTorch DDP
    import pytorch_lightning as pl
    trainer = pl.Trainer(
        strategy=ApexDDPStrategy(find_unused_parameters=False),  # delay_allreduce is ignored
    )
