import numpy as np

from vbsegm2step.config import Config
from vbsegm2step.segmentation.local import LocalPredictor
from vbsegm2step.vertebrae.vertebra import VertebraInfo


def _image_props(shape=(5, 5, 5)):
    z, y, x = shape
    return {
        "shape": shape,
        "spacing": (1.0, 1.0, 1.0),
        "size": (x, y, z),
    }


def test_predict_with_expansion_returns_arrays_matching_final_bbox_when_limit_reached():
    config = Config()
    config.PAD_X = 0
    config.PAD_Y = 0
    config.PAD_Z = 0
    config.EXPANSION_MM = 1
    config.MAX_EXPANSION_ITERATIONS = 2
    config.BORDER_MARGIN = 1

    image_props = _image_props()
    image_np = np.zeros(image_props["shape"], dtype=np.float32)
    anchor_mask = np.zeros(image_props["shape"], dtype=np.uint8)
    anchor_mask[2, 2, 2] = 1
    vertebra = VertebraInfo(15, anchor_mask, image_props, config=config)

    predictor = LocalPredictor.__new__(LocalPredictor)
    predictor.config = config

    calls = []

    def fake_predict(image_0000_np, image_0001_np, image_props):
        calls.append(image_0000_np.shape)
        segmentation = np.full(image_0000_np.shape, 2, dtype=np.uint8)
        probabilities = np.zeros((5,) + image_0000_np.shape, dtype=np.float32)
        probabilities[2] = 1.0
        return segmentation, probabilities

    predictor.predict = fake_predict

    segmentation, probabilities, bbox = predictor.predict_with_expansion(
        image_np,
        image_props,
        vertebra,
    )

    expected_shape = (
        bbox["z1"] - bbox["z0"],
        bbox["y1"] - bbox["y0"],
        bbox["x1"] - bbox["x0"],
    )
    assert calls == [(1, 1, 1), (3, 3, 3), (5, 5, 5)]
    assert segmentation.shape == expected_shape
    assert probabilities.shape == (5,) + expected_shape
