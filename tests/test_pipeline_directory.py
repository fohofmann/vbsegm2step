from pathlib import Path

from vbsegm2step.pipeline import VBSegm2StepPipeline


def test_process_directory_preserves_relative_paths_for_recursive_inputs(tmp_path):
    input_dir = tmp_path / "inputs"
    site_a = input_dir / "site_a"
    site_b = input_dir / "site_b"
    site_a.mkdir(parents=True)
    site_b.mkdir(parents=True)
    file_a = site_a / "ct.nii.gz"
    file_b = site_b / "ct.nii.gz"
    file_a.write_text("a", encoding="utf-8")
    file_b.write_text("b", encoding="utf-8")
    output_dir = tmp_path / "outputs"

    pipeline = VBSegm2StepPipeline.__new__(VBSegm2StepPipeline)
    calls = []

    def fake_process_file(input_path: Path, output_path: Path):
        calls.append((input_path, output_path))
        return True

    pipeline.process_file = fake_process_file

    success_count = pipeline.process_directory(
        input_dir,
        output_dir,
        pattern="**/*.nii.gz",
    )

    assert success_count == 2
    assert calls == [
        (file_a, output_dir / "site_a" / "ct.nii.gz"),
        (file_b, output_dir / "site_b" / "ct.nii.gz"),
    ]
