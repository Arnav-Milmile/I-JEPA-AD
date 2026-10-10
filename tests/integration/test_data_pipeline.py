"""CPU-only data pipeline: MVTec discovery → nominal split → encoder-ready batch."""

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn

from ijepa_ad.data.mvtec_dataset import validate_category_splits
from ijepa_ad.data.splits import split_nominal_train_val
from ijepa_ad.data.transforms import IMAGE_SIZE, preprocess_batch
from ijepa_ad.encoder import ijepa_wrapper
from ijepa_ad.encoder.ijepa_wrapper import IJEPAEncoder


class _TinyPatchEncoder(nn.Module):
    """Stand-in with the ViT-g/16 output contract; no checkpoint required."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3 * 16 * 16, 1408)

    def forward(self, image_batch: Tensor) -> Tensor:
        patches = image_batch.unfold(2, 16, 16).unfold(3, 16, 16)
        patches = patches.permute(0, 2, 3, 1, 4, 5).flatten(3)
        return self.projection(patches.flatten(0, 2)).view(
            image_batch.shape[0], 196, 1408
        )


def _make_mvtec(root: Path) -> None:
    for index in range(10):
        path = root / f"bottle/train/good/{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (300, 300), (index * 20, 40, 90)).save(path)
    for label, shade in (("good", 200), ("broken_large", 250)):
        path = root / f"bottle/test/{label}/000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (300, 300), shade).save(path)
    mask = root / "bottle/ground_truth/broken_large/000_mask.png"
    mask.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (300, 300), 255).save(mask)


def test_nominal_split_batches_match_frozen_encoder_input(tmp_path: Path) -> None:
    _make_mvtec(tmp_path / "mvtec")
    train, test = validate_category_splits(tmp_path / "mvtec", "bottle")
    test_before = test.samples

    split = split_nominal_train_val(train, val_fraction=0.2, seed=0)
    train_batch = preprocess_batch([s.image_path for s in split.train])
    val_batch = preprocess_batch([s.image_path for s in split.val])

    assert IMAGE_SIZE == ijepa_wrapper.IMAGE_SIZE
    assert train_batch.shape == (8, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert val_batch.shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert test.samples == test_before

    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"target_encoder": _TinyPatchEncoder().state_dict()}, checkpoint)
    encoder = IJEPAEncoder(checkpoint, model_factory=_TinyPatchEncoder)

    latents = encoder.encode(val_batch)

    assert latents.shape == (2, ijepa_wrapper.NUM_PATCHES, ijepa_wrapper.EMBEDDING_DIM)
