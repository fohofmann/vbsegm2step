import numpy as np

from vbsegm2step.config import Config
from vbsegm2step.vertebrae.canvas import VertebraCanvas
from vbsegm2step.vertebrae.queue import VertebraQueue
from vbsegm2step.vertebrae.vertebra import VertebraInfo


def _image_props(shape=(1, 1, 1)):
    z, y, x = shape
    return {
        "shape": shape,
        "spacing": (1.0, 1.0, 1.0),
        "size": (x, y, z),
    }


def _one_voxel_canvas():
    return VertebraCanvas(_image_props())


def _one_voxel_queue(anchor_label):
    image_props = _image_props()
    anchor_mask = np.ones(image_props["shape"], dtype=np.uint8)
    anchor = VertebraInfo(anchor_label, anchor_mask, image_props)
    return VertebraQueue(image_props, anchor)


def _task601_probabilities(winner_label, winner_probability, runner_up=None):
    n_channels = len(Config.VERTEBRA_LABELS) + 1
    probabilities = np.zeros((n_channels, 1, 1, 1), dtype=np.float32)

    assigned = {0, winner_label}
    probabilities[winner_label, 0, 0, 0] = winner_probability

    if runner_up is not None:
        runner_label, runner_probability = runner_up
        probabilities[runner_label, 0, 0, 0] = runner_probability
        assigned.add(runner_label)

    remaining = 1.0 - float(probabilities[:, 0, 0, 0].sum())
    other_labels = [label for label in range(n_channels) if label not in assigned]
    if other_labels:
        probabilities[other_labels, 0, 0, 0] = remaining / len(other_labels)

    return probabilities


def test_default_fusion_keeps_strong_global_only_winner():
    canvas = _one_voxel_canvas()
    canvas.add(_task601_probabilities(winner_label=15, winner_probability=0.90))

    segmentation, _ = canvas.export_fusion()

    assert segmentation[0, 0, 0] == 15


def test_default_dirichlet_smoothing_is_more_conservative_than_average_fusion():
    canvas = _one_voxel_canvas()
    canvas.add(_task601_probabilities(winner_label=15, winner_probability=0.70))

    smoothed_segmentation, _ = canvas.export_fusion()
    average_segmentation, _ = canvas.export_fusion(use_dirichlet=False)

    assert smoothed_segmentation[0, 0, 0] == 0
    assert average_segmentation[0, 0, 0] == 15


def test_default_dirichlet_fusion_matches_reference_formula_across_slabs():
    config = Config()
    config.CANVAS_DTYPE = np.float32
    config.FUSION_SLAB_Z = 1
    config.VERTEBRA_LABELS = {"T1": 1, "T2": 2}
    canvas = VertebraCanvas(_image_props(shape=(2, 1, 1)), config=config)
    probabilities = np.zeros((3, 2, 1, 1), dtype=np.float32)
    probabilities[:, 0, 0, 0] = [0.10, 0.80, 0.10]
    probabilities[:, 1, 0, 0] = [0.20, 0.45, 0.35]
    alphas = np.asarray([0.20, 0.05, 0.05], dtype=np.float32).reshape(3, 1, 1, 1)

    canvas.add(probabilities)
    segmentation, post = canvas.export_fusion(return_probabilities=True)
    expected_post = (probabilities + alphas) / (probabilities + alphas).sum(axis=0, keepdims=True)

    np.testing.assert_allclose(post, expected_post, rtol=1e-6)
    assert segmentation[:, 0, 0].tolist() == [1, 0]


def test_default_fusion_rejects_ambiguous_top_two_global_evidence():
    canvas = _one_voxel_canvas()
    canvas.add(
        _task601_probabilities(
            winner_label=15,
            winner_probability=0.45,
            runner_up=(14, 0.36),
        )
    )

    segmentation, _ = canvas.export_fusion()

    assert segmentation[0, 0, 0] == 0


def test_local_background_channel_is_accumulated_by_queue_labelmap():
    canvas = _one_voxel_canvas()
    segmentation = np.zeros((1, 1, 1), dtype=np.uint8)
    probabilities = np.zeros((5, 1, 1, 1), dtype=np.float32)
    probabilities[0, 0, 0, 0] = 0.90
    probabilities[1, 0, 0, 0] = 0.10

    weights = canvas.weights(segmentation, probabilities)
    canvas.add(probabilities, weights=weights, labelmap=_one_voxel_queue(15).labelmap)

    assert canvas.probability_wts[0, 0, 0, 0] > 0
    assert canvas.probability_sum[0, 0, 0, 0] > 0


def test_empty_local_prediction_without_low_any_does_not_write_background_weight():
    canvas = _one_voxel_canvas()
    segmentation = np.zeros((1, 1, 1), dtype=np.uint8)
    probabilities = np.zeros((5, 1, 1, 1), dtype=np.float32)
    probabilities[0, 0, 0, 0] = 0.10
    probabilities[1, 0, 0, 0] = 0.90

    weights = canvas.weights(segmentation, probabilities)

    assert np.all(weights == 0)


def test_local_foreground_weights_keep_all_atomic_channels_for_valid_voxels():
    canvas = _one_voxel_canvas()
    segmentation = np.full((1, 1, 1), 2, dtype=np.uint8)
    probabilities = np.zeros((5, 1, 1, 1), dtype=np.float32)
    probabilities[2, 0, 0, 0] = 0.80
    probabilities[3, 0, 0, 0] = 0.10
    probabilities[4, 0, 0, 0] = 0.05

    weights = canvas.weights(segmentation, probabilities, tau_vert=0.20)

    assert weights[2, 0, 0, 0] > 0
    assert weights[3, 0, 0, 0] > 0
    assert weights[4, 0, 0, 0] > 0


def test_absent_neighbor_channel_is_not_mapped_to_background():
    canvas = _one_voxel_canvas()
    segmentation = np.full((1, 1, 1), 3, dtype=np.uint8)
    probabilities = np.zeros((5, 1, 1, 1), dtype=np.float32)
    probabilities[3, 0, 0, 0] = 0.90

    weights = canvas.weights(segmentation, probabilities)
    canvas.add(probabilities, weights=weights, labelmap=_one_voxel_queue(1).labelmap)

    assert 3 not in _one_voxel_queue(1).labelmap
    assert canvas.probability_wts[0, 0, 0, 0] == 0
    assert canvas.probability_sum[0, 0, 0, 0] == 0


def test_canvas_exports_anatomical_label_for_non_contiguous_config():
    config = Config()
    config.CANVAS_DTYPE = np.float32
    config.VERTEBRA_LABELS = {"T1": 1, "L3": 15}
    canvas = VertebraCanvas(_image_props(), config=config)
    probabilities = np.zeros((3, 1, 1, 1), dtype=np.float32)
    probabilities[0, 0, 0, 0] = 0.10
    probabilities[1, 0, 0, 0] = 0.20
    probabilities[2, 0, 0, 0] = 0.70

    canvas.add(probabilities)
    segmentation, _ = canvas.export_fusion()

    assert segmentation[0, 0, 0] == 15
    assert canvas.probability_wts[2, 0, 0, 0] == 1.0
