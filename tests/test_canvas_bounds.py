import numpy as np

from vbsegm2step.vertebrae.canvas import VertebraCanvas


def _image_props(shape=(2, 3, 4)):
    z, y, x = shape
    return {
        "shape": shape,
        "spacing": (1.0, 1.0, 1.0),
        "size": (x, y, z),
    }


def test_canvas_rejects_channel_equal_to_probability_channel_count():
    canvas = VertebraCanvas(_image_props())
    probabilities = np.ones((2, 2, 3, 4), dtype=np.float32)

    canvas.add(probabilities, labelmap={2: 1})

    assert np.all(canvas.probability_sum == 0)
    assert np.all(canvas.probability_wts == 0)


def test_canvas_rejects_negative_channel_index():
    canvas = VertebraCanvas(_image_props())
    probabilities = np.ones((2, 2, 3, 4), dtype=np.float32)

    canvas.add(probabilities, labelmap={-1: 1})

    assert np.all(canvas.probability_sum == 0)
    assert np.all(canvas.probability_wts == 0)
