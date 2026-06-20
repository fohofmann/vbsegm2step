from pathlib import Path

import numpy as np

from vbsegm2step.config import Config
import vbsegm2step.pipeline as pipeline_module
from vbsegm2step.pipeline import VBSegm2StepPipeline
from vbsegm2step.vertebrae.canvas import VertebraCanvas
from vbsegm2step.vertebrae.queue import VertebraQueue
from vbsegm2step.vertebrae.vertebra import VertebraInfo


def _image_props(shape=(10, 10, 10)):
    z, y, x = shape
    return {
        "shape": shape,
        "spacing": (1.0, 1.0, 1.0),
        "size": (x, y, z),
    }


def _mask(shape=(10, 10, 10), center=(5, 5, 5)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[center] = 1
    return mask


def test_vertebra_info_uses_instance_config_for_bbox_padding():
    config = Config()
    config.PAD_X = 1
    config.PAD_Y = 2
    config.PAD_Z = 3

    vertebra = VertebraInfo(15, _mask(), _image_props(), config=config)

    assert vertebra.bbox == {"z0": 2, "z1": 9, "y0": 3, "y1": 8, "x0": 4, "x1": 7}


def test_vertebra_info_uses_instance_config_for_bbox_expansion():
    config = Config()
    config.PAD_X = 0
    config.PAD_Y = 0
    config.PAD_Z = 0
    config.EXPANSION_MM = 2

    vertebra = VertebraInfo(15, _mask(), _image_props(), config=config)

    assert vertebra.bbox == {"z0": 5, "z1": 6, "y0": 5, "y1": 6, "x0": 5, "x1": 6}
    assert vertebra.bbox_expand(["x0", "y1", "z1"]) == {
        "z0": 5,
        "z1": 8,
        "y0": 5,
        "y1": 8,
        "x0": 3,
        "x1": 6,
    }


def test_queue_passes_instance_config_to_neighbor_vertebrae():
    config = Config()
    config.PAD_X = 0
    config.PAD_Y = 0
    config.PAD_Z = 0
    anchor = VertebraInfo(15, _mask(), _image_props(), config=config)
    queue = VertebraQueue(_image_props(), anchor, config=config)

    segmentation = np.zeros(_image_props()["shape"], dtype=np.uint8)
    segmentation[5, 5, 5] = 2
    segmentation[7, 5, 5] = 3

    assert queue.process_segmentation(segmentation)
    neighbor = queue.queue[1]
    assert neighbor.config is config


def test_queue_default_tracking_uses_config_label_set():
    config = Config()
    config.VERTEBRA_LABELS = {
        "T1": 101,
        "T2": 102,
        "sacrum": 119,
        "coccyx": 120,
        "T13": 121,
    }
    anchor = VertebraInfo(101, _mask(), _image_props(), config=config)

    queue = VertebraQueue(_image_props(), anchor, config=config)

    assert queue.tracker_all == [101, 102]


def test_processable_vertebra_labels_excludes_non_queue_labels():
    config = Config()

    assert config.processable_vertebra_labels() == list(range(1, 19))


def test_default_vertebra_labels_match_released_model_contract():
    assert Config.VERTEBRA_LABELS == {
        "T1": 1,
        "T2": 2,
        "T3": 3,
        "T4": 4,
        "T5": 5,
        "T6": 6,
        "T7": 7,
        "T8": 8,
        "T9": 9,
        "T10": 10,
        "T11": 11,
        "T12": 12,
        "L1": 13,
        "L2": 14,
        "L3": 15,
        "L4": 16,
        "L5": 17,
        "L6": 18,
        "sacrum": 19,
        "coccyx": 20,
        "T13": 21,
    }


def test_volume_validation_uses_instance_thresholds():
    config = Config()
    config.MIN_VERTEBRA_VOLUME_CM3 = 1
    config.MAX_VERTEBRA_VOLUME_CM3 = 1

    assert config.is_valid_vertebra_volume(1000, (1.0, 1.0, 1.0))
    assert not config.is_valid_vertebra_volume(2000, (1.0, 1.0, 1.0))


def test_canvas_uses_instance_config_for_dtype_and_label_set():
    config = Config()
    config.CANVAS_DTYPE = np.float32
    config.VERTEBRA_LABELS = {"T1": 1, "L3": 15}

    canvas = VertebraCanvas(_image_props(shape=(2, 3, 4)), config=config)

    assert canvas.config is config
    assert canvas.vertebra_labels == [0, 1, 15]
    assert canvas.probability_sum.dtype == np.float32
    assert canvas.probability_wts.dtype == np.float32
    assert canvas.label_to_index == {0: 0, 1: 1, 15: 2}


def test_canvas_fusion_output_is_stable_across_slab_sizes():
    config = Config()
    config.CANVAS_DTYPE = np.float32
    config.VERTEBRA_LABELS = {"T1": 1, "L3": 15}
    image_props = _image_props(shape=(5, 2, 2))
    probabilities = np.zeros((3,) + image_props["shape"], dtype=np.float32)
    probabilities[0] = 0.05
    probabilities[1] = 0.20
    probabilities[2] = 0.75

    def fuse_with_slab_size(slab_size):
        config.FUSION_SLAB_Z = slab_size
        canvas = VertebraCanvas(image_props, config=config)
        canvas.add(probabilities)
        return canvas.export_fusion(return_probabilities=True)

    small_slab_segmentation, small_slab_probabilities = fuse_with_slab_size(1)
    large_slab_segmentation, large_slab_probabilities = fuse_with_slab_size(99)

    assert np.array_equal(small_slab_segmentation, large_slab_segmentation)
    assert np.allclose(small_slab_probabilities, large_slab_probabilities)


def test_pipeline_passes_instance_config_to_queue_and_canvas(monkeypatch):
    config = Config()
    image_props = _image_props(shape=(2, 3, 4))
    image_np = np.zeros(image_props["shape"], dtype=np.float32)
    spine_segmentation = np.zeros(image_props["shape"], dtype=np.uint8)
    spine_probabilities = np.zeros((len(config.VERTEBRA_LABELS) + 1,) + image_props["shape"], dtype=np.float32)
    anchor = VertebraInfo(15, _mask(shape=image_props["shape"], center=(1, 1, 1)), image_props, config=config)
    received_configs = []

    class FakeSpinePredictor:
        def predict(self, image_np, image_props):
            return spine_segmentation, spine_probabilities

        def select_best_anchor(self, segmentation, probabilities, image_props):
            return anchor

    class FakeQueue:
        tracker_done = []

        def __init__(self, image_props, vertebra_first, config=None):
            received_configs.append(("queue", config))

        def __bool__(self):
            return False

    class FakeCanvas:
        def __init__(self, image_props, config=None):
            received_configs.append(("canvas", config))

        def add(self, probabilities, weights=1.0, labelmap=None, bbox=None):
            return None

        def export_fusion(self, return_probabilities=False):
            return spine_segmentation, None

    def fake_load_image(input_path):
        return None, image_np, image_props

    monkeypatch.setattr(pipeline_module, "load_image", fake_load_image)
    monkeypatch.setattr(pipeline_module, "VertebraQueue", FakeQueue)
    monkeypatch.setattr(pipeline_module, "VertebraCanvas", FakeCanvas)

    pipeline = VBSegm2StepPipeline.__new__(VBSegm2StepPipeline)
    pipeline.config = config
    pipeline.spine_predictor = FakeSpinePredictor()
    pipeline.local_predictor = None

    result = pipeline.process_file(Path("ct.nii.gz"))

    assert result[0] is spine_segmentation
    assert result[1] is None
    assert received_configs == [("queue", config), ("canvas", config)]
