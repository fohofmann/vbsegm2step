import numpy as np

from vbsegm2step.config import Config
from vbsegm2step.segmentation.spine import SpinePredictor
from vbsegm2step.vertebrae.vertebra import VertebraInfo


def _image_props(shape=(1, 1, 2)):
    z, y, x = shape
    return {
        "shape": shape,
        "spacing": (1.0, 1.0, 1.0),
        "size": (x, y, z),
    }


def test_anchor_selection_uses_same_processable_labels_as_queue(monkeypatch):
    config = Config()
    predictor = SpinePredictor.__new__(SpinePredictor)
    predictor.config = config
    image_props = _image_props()
    segmentation = np.zeros(image_props["shape"], dtype=np.uint8)
    segmentation[0, 0, 0] = 21
    segmentation[0, 0, 1] = 15
    probabilities = np.zeros((len(config.VERTEBRA_LABELS) + 1,) + image_props["shape"], dtype=np.float32)
    seen_labels = []

    def fake_calculate(segmentation, probabilities, label, image_props):
        seen_labels.append(label)
        mask = (segmentation == label).astype(np.uint8)
        return VertebraInfo(label, mask, image_props, config=config), 1.0

    monkeypatch.setattr(predictor, "calculate_vertebra_confidence", fake_calculate)

    anchor = predictor.select_best_anchor(segmentation, probabilities, image_props)

    assert seen_labels == [15]
    assert anchor.label == 15


def test_anchor_selection_returns_none_when_only_unprocessable_labels_exist(monkeypatch):
    config = Config()
    predictor = SpinePredictor.__new__(SpinePredictor)
    predictor.config = config
    image_props = _image_props(shape=(1, 1, 1))
    segmentation = np.full(image_props["shape"], 21, dtype=np.uint8)
    probabilities = np.zeros((len(config.VERTEBRA_LABELS) + 1,) + image_props["shape"], dtype=np.float32)

    def fail_if_called(*args):
        raise AssertionError("unprocessable labels should not be scored")

    monkeypatch.setattr(predictor, "calculate_vertebra_confidence", fail_if_called)

    assert predictor.select_best_anchor(segmentation, probabilities, image_props) is None
