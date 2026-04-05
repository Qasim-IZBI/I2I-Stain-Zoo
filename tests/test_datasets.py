"""
Dataset pipeline tests.

Covers:
  - UnpairedDataset: returns dict with A/B/path_A/path_B, tensors in [-1, 1]
  - SingleDomainDataset: returns (tensor, path) tuple
  - TargetOnlyDataset: returns dict with B/path_B
  - data_range: loading from numbered subdirectories
  - Transform: output dtype, shape, value range
"""
import os
import sys

import pytest
import torch
from PIL import Image
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.transforms import default_train_transform
from datasets.unpaired_dataset import UnpairedDataset
from datasets.single_domain_dataset import SingleDomainDataset
from datasets.target_only_dataset import TargetOnlyDataset

IMAGE_SIZE = 64
TRANSFORM = default_train_transform(image_size=IMAGE_SIZE)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _make_image_dir(base_dir, n=4, size=IMAGE_SIZE):
    """Create base_dir/images/ with n tif images."""
    img_dir = os.path.join(base_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (size, size), color=(i * 40, 100, 200))
        img.save(os.path.join(img_dir, f"{i+1:07d}.tif"))
    return base_dir


def _make_numbered_dirs(base_dir, folder_indices, n_images=2):
    """Create base_dir/001/images/, base_dir/002/images/, etc."""
    for idx in folder_indices:
        sub = os.path.join(base_dir, f"{idx:03d}", "images")
        os.makedirs(sub, exist_ok=True)
        for i in range(n_images):
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(i * 30, 80, 160))
            img.save(os.path.join(sub, f"{i+1:07d}.tif"))
    return base_dir


# ------------------------------------------------------------------ #
#  UnpairedDataset                                                     #
# ------------------------------------------------------------------ #

class TestUnpairedDataset:
    def test_len_positive(self, tmp_unpaired_dirs):
        dA, dB = tmp_unpaired_dirs
        ds = UnpairedDataset(dA, dB, transform=TRANSFORM)
        assert len(ds) > 0

    def test_item_keys(self, tmp_unpaired_dirs):
        dA, dB = tmp_unpaired_dirs
        ds = UnpairedDataset(dA, dB, transform=TRANSFORM)
        item = ds[0]
        assert set(item.keys()) >= {"A", "B", "path_A", "path_B"}

    def test_tensor_shapes(self, tmp_unpaired_dirs):
        dA, dB = tmp_unpaired_dirs
        ds = UnpairedDataset(dA, dB, transform=TRANSFORM)
        item = ds[0]
        assert item["A"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert item["B"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_tensor_dtype(self, tmp_unpaired_dirs):
        dA, dB = tmp_unpaired_dirs
        ds = UnpairedDataset(dA, dB, transform=TRANSFORM)
        item = ds[0]
        assert item["A"].dtype == torch.float32
        assert item["B"].dtype == torch.float32

    def test_value_range(self, tmp_unpaired_dirs):
        dA, dB = tmp_unpaired_dirs
        ds = UnpairedDataset(dA, dB, transform=TRANSFORM)
        item = ds[0]
        assert item["A"].min() >= -1.0 - 1e-4
        assert item["A"].max() <= 1.0 + 1e-4
        assert item["B"].min() >= -1.0 - 1e-4
        assert item["B"].max() <= 1.0 + 1e-4

    def test_paths_are_strings(self, tmp_unpaired_dirs):
        dA, dB = tmp_unpaired_dirs
        ds = UnpairedDataset(dA, dB, transform=TRANSFORM)
        item = ds[0]
        assert isinstance(item["path_A"], str)
        assert isinstance(item["path_B"], str)

    def test_requires_transform(self, tmp_unpaired_dirs):
        dA, dB = tmp_unpaired_dirs
        ds = UnpairedDataset(dA, dB, transform=None)
        with pytest.raises(ValueError):
            _ = ds[0]

    def test_data_range(self):
        with tempfile.TemporaryDirectory() as dA:
            with tempfile.TemporaryDirectory() as dB:
                _make_numbered_dirs(dA, [1, 2, 3], n_images=2)
                _make_numbered_dirs(dB, [1, 2, 3], n_images=2)
                ds_full  = UnpairedDataset(dA, dB, transform=TRANSFORM)
                ds_range = UnpairedDataset(dA, dB, transform=TRANSFORM, data_range=(1, 2))
                assert len(ds_range) < len(ds_full)
                assert len(ds_range) > 0


# ------------------------------------------------------------------ #
#  SingleDomainDataset                                                 #
# ------------------------------------------------------------------ #

class TestSingleDomainDataset:
    def test_len_positive(self, tmp_image_dir):
        ds = SingleDomainDataset(tmp_image_dir, transform=TRANSFORM)
        assert len(ds) > 0

    def test_item_is_tensor_path_pair(self, tmp_image_dir):
        ds = SingleDomainDataset(tmp_image_dir, transform=TRANSFORM)
        item = ds[0]
        assert isinstance(item, (tuple, list)) and len(item) == 2
        tensor, path = item
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(path, str)

    def test_tensor_shape(self, tmp_image_dir):
        ds = SingleDomainDataset(tmp_image_dir, transform=TRANSFORM)
        tensor, _ = ds[0]
        assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_value_range(self, tmp_image_dir):
        ds = SingleDomainDataset(tmp_image_dir, transform=TRANSFORM)
        tensor, _ = ds[0]
        assert tensor.min() >= -1.0 - 1e-4
        assert tensor.max() <= 1.0 + 1e-4

    def test_data_range(self):
        with tempfile.TemporaryDirectory() as d:
            _make_numbered_dirs(d, [1, 2, 3], n_images=2)
            ds_full  = SingleDomainDataset(d, transform=TRANSFORM)
            ds_range = SingleDomainDataset(d, transform=TRANSFORM, data_range=(1, 2))
            assert len(ds_range) < len(ds_full)
            assert len(ds_range) > 0


# ------------------------------------------------------------------ #
#  TargetOnlyDataset                                                   #
# ------------------------------------------------------------------ #

class TestTargetOnlyDataset:
    def test_len_positive(self, tmp_image_dir):
        ds = TargetOnlyDataset(tmp_image_dir, transform=TRANSFORM)
        assert len(ds) > 0

    def test_item_keys(self, tmp_image_dir):
        ds = TargetOnlyDataset(tmp_image_dir, transform=TRANSFORM)
        item = ds[0]
        assert "B" in item
        assert "path_B" in item

    def test_tensor_shape(self, tmp_image_dir):
        ds = TargetOnlyDataset(tmp_image_dir, transform=TRANSFORM)
        item = ds[0]
        assert item["B"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_value_range(self, tmp_image_dir):
        ds = TargetOnlyDataset(tmp_image_dir, transform=TRANSFORM)
        item = ds[0]
        assert item["B"].min() >= -1.0 - 1e-4
        assert item["B"].max() <= 1.0 + 1e-4

    def test_path_is_string(self, tmp_image_dir):
        ds = TargetOnlyDataset(tmp_image_dir, transform=TRANSFORM)
        item = ds[0]
        assert isinstance(item["path_B"], str)

    def test_data_range(self):
        with tempfile.TemporaryDirectory() as d:
            _make_numbered_dirs(d, [1, 2, 3], n_images=2)
            ds_full  = TargetOnlyDataset(d, transform=TRANSFORM)
            ds_range = TargetOnlyDataset(d, transform=TRANSFORM, data_range=(1, 2))
            assert len(ds_range) < len(ds_full)
            assert len(ds_range) > 0
