import torch
from torch import amp
from contextlib import nullcontext

def validate_loss_factory(val_loader, device="cuda"):
    us_amp = (torch.device(device).type == 'cuda')
    ac = amp.autocast(device_type='cuda', dtype=torch.float16) if us_amp \
        else nullcontext()

    @torch.no_grad()
    def _run(model, _device=None):
        # RetinaNet computes loss only in training mode; FrozenBN in torchvision avoids BN drift.
        was_training = model.training
        model.train()

        num_batches = 0
        loss_sum = 0.0
        classification_sum = 0.0
        regression_sum = 0.0

        for imgs, targets in val_loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with ac:
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())

            num_batches += 1
            loss_sum += float(loss.item())
            classification_sum += float(loss_dict["classification"].item())
            regression_sum += float(loss_dict["bbox_regression"].item())

        if not was_training:
            model.eval()

        return (loss_sum / max(num_batches, 1),
                classification_sum / max(num_batches, 1),
                regression_sum / max(num_batches, 1))
    return _run
