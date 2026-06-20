import numpy as np

from vbsegm2step.config import Config
from vbsegm2step.segmentation.local import LocalPredictor
from vbsegm2step.vertebrae.canvas import VertebraCanvas
from vbsegm2step.vertebrae.vertebra import VertebraInfo


def _image_props(shape=(4, 5, 6), spacing=(0.5, 2.0, 4.0)):
    z, y, x = shape
    return {
        "shape": shape,
        "spacing": spacing,
        "size": (x, y, z),
    }


def test_bbox_padding_uses_spacing_xyz_but_slices_shape_zyx():
    config = Config()
    config.PAD_X = 1.5  # 3 voxels at spacing x=0.5
    config.PAD_Y = 3.1  # 2 voxels at spacing y=2.0
    config.PAD_Z = 8.1  # 3 voxels at spacing z=4.0

    image_props = _image_props(shape=(11, 13, 17), spacing=(0.5, 2.0, 4.0))
    mask = np.zeros(image_props["shape"], dtype=np.uint8)
    mask[5, 6, 7] = 1

    vertebra = VertebraInfo(15, mask, image_props, config=config)

    assert vertebra.bbox == {
        "z0": 2,
        "z1": 9,
        "y0": 4,
        "y1": 9,
        "x0": 4,
        "x1": 11,
    }
    assert vertebra.segmentation.shape == (7, 5, 7)
    assert vertebra.segmentation[3, 2, 3] == 1


def test_canvas_add_places_bbox_crop_in_zyx_array_order():
    image_props = _image_props(shape=(4, 5, 6))
    canvas = VertebraCanvas(image_props)
    bbox = {"z0": 1, "z1": 3, "y0": 2, "y1": 5, "x0": 0, "x1": 2}
    probabilities = np.zeros((5, 2, 3, 2), dtype=np.float32)
    probabilities[2] = 0.75

    canvas.add(probabilities, weights=1.0, labelmap={2: 15}, bbox=bbox)

    channel = canvas.label_to_index[15]
    expected = np.zeros(image_props["shape"], dtype=canvas.probability_sum.dtype)
    expected[1:3, 2:5, 0:2] = 0.75
    assert np.array_equal(canvas.probability_sum[channel], expected)


def test_local_expansion_maps_numpy_borders_to_bbox_axes():
    config = Config()
    config.BORDER_MARGIN = 1
    predictor = LocalPredictor.__new__(LocalPredictor)
    predictor.config = config
    image_props = _image_props(shape=(6, 5, 6))
    bbox = {"z0": 1, "z1": 4, "y0": 1, "y1": 4, "x0": 2, "x1": 5}
    segmentation = np.zeros((3, 3, 3), dtype=np.uint8)
    segmentation[1, 1, 0] = 2   # x0 face
    segmentation[1, -1, 1] = 3  # y1 face
    segmentation[0, 1, 1] = 4   # z0 face

    required, directions = predictor.check_expansion_required(
        segmentation,
        bbox,
        image_props,
    )

    assert required
    assert directions == {
        "x0": True,
        "x1": False,
        "y0": False,
        "y1": True,
        "z0": True,
        "z1": False,
    }
