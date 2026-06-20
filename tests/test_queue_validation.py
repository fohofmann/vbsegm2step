import numpy as np

from vbsegm2step.vertebrae.queue import VertebraQueue
from vbsegm2step.vertebrae.vertebra import VertebraInfo


def _image_props(shape=(10, 6, 6)):
    z, y, x = shape
    return {
        "shape": shape,
        "spacing": (1.0, 1.0, 1.0),
        "size": (x, y, z),
    }


def _queue(anchor_label=15):
    image_props = _image_props()
    anchor_mask = np.zeros(image_props["shape"], dtype=np.uint8)
    anchor_mask[5, 3, 3] = 1
    anchor = VertebraInfo(anchor_label, anchor_mask, image_props)
    return VertebraQueue(image_props, anchor)


def test_queue_iteration_limit_matches_tracking_list_length():
    image_props = _image_props()
    anchor_mask = np.zeros(image_props["shape"], dtype=np.uint8)
    anchor_mask[5, 3, 3] = 1
    anchor = VertebraInfo(15, anchor_mask, image_props)

    queue = VertebraQueue(image_props, anchor, vertebra_all=[14, 15, 16])

    assert queue.settings_counter_max == 3


def _segmentation_with_labels(*labels_at_z):
    segmentation = np.zeros(_image_props()["shape"], dtype=np.uint8)
    for label, z in labels_at_z:
        segmentation[z, 3, 3] = label
    return segmentation


def test_process_segmentation_accepts_center_only_prediction():
    queue = _queue()

    assert queue.process_segmentation(_segmentation_with_labels((2, 5)))
    assert queue.tracker_done == [15]
    assert queue.tracker_done_z == [5]


def test_process_segmentation_rejects_missing_center_label():
    queue = _queue()

    assert not queue.process_segmentation(_segmentation_with_labels((3, 6)))
    assert queue.tracker_done == []
    assert queue.tracker_done_z == []


def test_process_segmentation_rejects_cranial_only_prediction():
    queue = _queue()

    assert not queue.process_segmentation(_segmentation_with_labels((3, 6)))


def test_process_segmentation_rejects_caudal_only_prediction():
    queue = _queue()

    assert not queue.process_segmentation(_segmentation_with_labels((4, 4)))


def test_process_segmentation_rejects_internal_cranial_monotonicity_violation():
    queue = _queue()

    assert not queue.process_segmentation(_segmentation_with_labels((2, 5), (3, 4)))


def test_process_segmentation_rejects_internal_caudal_monotonicity_violation():
    queue = _queue()

    assert not queue.process_segmentation(_segmentation_with_labels((2, 5), (4, 6)))


def test_process_segmentation_ignores_missing_processed_neighbor_z_position():
    queue = _queue()
    queue.tracker_done = [14]
    queue.tracker_done_z = [None]

    assert queue.process_segmentation(_segmentation_with_labels((2, 5)))


def test_labelmap_includes_background_and_valid_neighbors():
    assert _queue(anchor_label=15).labelmap == {0: 0, 2: 15, 3: 14, 4: 16}


def test_labelmap_omits_missing_cranial_neighbor_at_tracking_boundary():
    assert _queue(anchor_label=1).labelmap == {0: 0, 2: 1, 4: 2}


def test_labelmap_omits_missing_caudal_neighbor_at_tracking_boundary():
    assert _queue(anchor_label=18).labelmap == {0: 0, 2: 18, 3: 17}
