import numpy as np

from vbsegm2step.config import Config
from vbsegm2step.vertebrae.vertebra import VertebraInfo


def _image_props(shape):
    z, y, x = shape
    return {
        "shape": shape,
        "spacing": (1.0, 1.0, 1.0),
        "size": (x, y, z),
    }


def _make_vertebra(mask):
    return VertebraInfo(
        label=15,
        segmentation=mask,
        image_props=_image_props(mask.shape),
    )


def test_bbox_initialize_includes_single_voxel_with_zero_padding(monkeypatch):
    monkeypatch.setattr(Config, "PAD_X", 0)
    monkeypatch.setattr(Config, "PAD_Y", 0)
    monkeypatch.setattr(Config, "PAD_Z", 0)

    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    mask[2, 3, 4] = 1

    vertebra = _make_vertebra(mask)

    assert vertebra.bbox == {"z0": 2, "z1": 3, "y0": 3, "y1": 4, "x0": 4, "x1": 5}
    assert vertebra.segmentation.shape == (1, 1, 1)
    assert vertebra.segmentation[0, 0, 0] == 1


def test_bbox_initialize_clips_edge_touching_mask_to_image_size(monkeypatch):
    monkeypatch.setattr(Config, "PAD_X", 0)
    monkeypatch.setattr(Config, "PAD_Y", 0)
    monkeypatch.setattr(Config, "PAD_Z", 0)

    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    mask[4, 5, 6] = 1

    vertebra = _make_vertebra(mask)

    assert vertebra.bbox == {"z0": 4, "z1": 5, "y0": 5, "y1": 6, "x0": 6, "x1": 7}
    assert vertebra.segmentation.shape == (1, 1, 1)
    assert vertebra.segmentation[0, 0, 0] == 1


def test_bbox_initialize_multi_voxel_shape_matches_python_slicing(monkeypatch):
    monkeypatch.setattr(Config, "PAD_X", 0)
    monkeypatch.setattr(Config, "PAD_Y", 0)
    monkeypatch.setattr(Config, "PAD_Z", 0)

    mask = np.zeros((6, 7, 8), dtype=np.uint8)
    mask[1:4, 2:6, 3:7] = 1

    vertebra = _make_vertebra(mask)

    assert vertebra.bbox == {"z0": 1, "z1": 4, "y0": 2, "y1": 6, "x0": 3, "x1": 7}
    assert vertebra.segmentation.shape == (3, 4, 4)
    assert np.all(vertebra.segmentation == 1)
