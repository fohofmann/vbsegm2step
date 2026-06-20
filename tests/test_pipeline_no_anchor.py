from pathlib import Path

import numpy as np

import vbsegm2step.pipeline as pipeline_module
from vbsegm2step.pipeline import VBSegm2StepPipeline


class _NoAnchorSpinePredictor:
    def __init__(self, segmentation, probabilities):
        self.segmentation = segmentation
        self.probabilities = probabilities

    def predict(self, image_np, image_props):
        return self.segmentation, self.probabilities

    def select_best_anchor(self, segmentation, probabilities, image_props):
        return None


def _pipeline_without_init(segmentation, probabilities):
    pipeline = VBSegm2StepPipeline.__new__(VBSegm2StepPipeline)
    pipeline.spine_predictor = _NoAnchorSpinePredictor(segmentation, probabilities)
    pipeline.local_predictor = None
    return pipeline


def _patch_io(monkeypatch):
    image_np = np.zeros((2, 3, 4), dtype=np.float32)
    image_props = {"shape": image_np.shape}
    save_calls = []

    def fake_load_image(input_path):
        return None, image_np, image_props

    def fake_save_segmentation(segmentation, output_path, props):
        save_calls.append((segmentation.copy(), Path(output_path), props))

    monkeypatch.setattr(pipeline_module, "load_image", fake_load_image)
    monkeypatch.setattr(pipeline_module, "save_segmentation", fake_save_segmentation)
    return save_calls


def test_process_file_rejects_equivalent_input_output_paths_before_loading(monkeypatch, tmp_path, capsys):
    def fail_if_loaded(input_path):
        raise AssertionError("image should not be loaded")

    monkeypatch.setattr(pipeline_module, "load_image", fail_if_loaded)
    pipeline = VBSegm2StepPipeline.__new__(VBSegm2StepPipeline)
    input_path = tmp_path / "ct.nii.gz"
    output_path = tmp_path / "nested" / ".." / "ct.nii.gz"

    result = pipeline.process_file(input_path, output_path=output_path)

    assert result is False
    captured = capsys.readouterr()
    assert "Input and output paths cannot be the same" in captured.out


def test_no_anchor_does_not_save_when_output_path_is_none(monkeypatch):
    segmentation = np.ones((2, 3, 4), dtype=np.uint8)
    probabilities = np.ones((2, 2, 3, 4), dtype=np.float32)
    save_calls = _patch_io(monkeypatch)

    result_segmentation, result_probabilities = _pipeline_without_init(
        segmentation, probabilities
    ).process_file(Path("ct.nii.gz"), output_path=None)

    assert np.array_equal(result_segmentation, segmentation)
    assert result_probabilities is None
    assert save_calls == []


def test_no_anchor_saves_when_output_path_is_provided(monkeypatch):
    segmentation = np.ones((2, 3, 4), dtype=np.uint8)
    probabilities = np.ones((2, 2, 3, 4), dtype=np.float32)
    save_calls = _patch_io(monkeypatch)

    result_segmentation, result_probabilities = _pipeline_without_init(
        segmentation, probabilities
    ).process_file(Path("ct.nii.gz"), output_path=Path("out.nii.gz"))

    assert np.array_equal(result_segmentation, segmentation)
    assert result_probabilities is None
    assert len(save_calls) == 1
    assert save_calls[0][1] == Path("out.nii.gz")


def test_no_anchor_returns_probabilities_only_when_requested(monkeypatch):
    segmentation = np.ones((2, 3, 4), dtype=np.uint8)
    probabilities = np.ones((2, 2, 3, 4), dtype=np.float32)
    _patch_io(monkeypatch)

    result_segmentation, result_probabilities = _pipeline_without_init(
        segmentation, probabilities
    ).process_file(
        Path("ct.nii.gz"),
        output_path=None,
        return_probabilities=True,
    )

    assert np.array_equal(result_segmentation, segmentation)
    assert np.array_equal(result_probabilities, probabilities)
