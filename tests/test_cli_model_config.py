from pathlib import Path

import pytest

import vbsegm2step.cli as cli
from vbsegm2step.config import Config


def _make_model_root(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    trainer = root / Config.TRAINER_FOLDER
    (trainer / "fold_all").mkdir(parents=True)
    (trainer / "plans.json").write_text("{}", encoding="utf-8")
    (trainer / "dataset.json").write_text("{}", encoding="utf-8")
    (trainer / "fold_all" / "checkpoint_final.pth").write_text("weights", encoding="utf-8")
    return root


def test_config_from_overrides_keeps_default_paths_without_env_or_cli(monkeypatch):
    monkeypatch.delenv(Config.MODEL601_ENV, raising=False)
    monkeypatch.delenv(Config.MODEL602_ENV, raising=False)

    config = Config.from_overrides()

    assert config.PATH_NNUNET601 == Config.PATH_NNUNET601
    assert config.PATH_NNUNET602 == Config.PATH_NNUNET602


def test_config_from_overrides_uses_environment_paths(monkeypatch, tmp_path):
    model601 = tmp_path / "env601"
    model602 = tmp_path / "env602"
    monkeypatch.setenv(Config.MODEL601_ENV, str(model601))
    monkeypatch.setenv(Config.MODEL602_ENV, str(model602))

    config = Config.from_overrides()

    assert config.PATH_NNUNET601 == model601
    assert config.PATH_NNUNET602 == model602


def test_config_from_overrides_cli_paths_take_precedence_over_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(Config.MODEL601_ENV, str(tmp_path / "env601"))
    monkeypatch.setenv(Config.MODEL602_ENV, str(tmp_path / "env602"))
    cli_model601 = tmp_path / "cli601"
    cli_model602 = tmp_path / "cli602"

    config = Config.from_overrides(model601=cli_model601, model602=cli_model602)

    assert config.PATH_NNUNET601 == cli_model601
    assert config.PATH_NNUNET602 == cli_model602


def test_validate_model_paths_accepts_expected_nnunet_layout(tmp_path):
    config = Config()
    config.PATH_NNUNET601 = _make_model_root(tmp_path, "model601")
    config.PATH_NNUNET602 = _make_model_root(tmp_path, "model602")

    assert config.validate_model_paths()


def test_validate_model_paths_rejects_missing_checkpoint(tmp_path):
    config = Config()
    config.PATH_NNUNET601 = _make_model_root(tmp_path, "model601")
    config.PATH_NNUNET602 = _make_model_root(tmp_path, "model602")
    (config.nnunet602_trainer_path() / "fold_all" / "checkpoint_final.pth").unlink()

    assert not config.validate_model_paths()


def test_cli_predict_passes_model_overrides(monkeypatch, tmp_path):
    calls = []

    def fake_predict(input_file, output_file, model601=None, model602=None):
        calls.append((input_file, output_file, model601, model602))

    monkeypatch.setattr(cli, "predict", fake_predict)
    monkeypatch.setattr(
        "sys.argv",
        [
            "vbsegm2step",
            "predict",
            "-i",
            "ct.nii.gz",
            "-o",
            "seg.nii.gz",
            "--model601",
            str(tmp_path / "model601"),
            "--model602",
            str(tmp_path / "model602"),
        ],
    )

    cli.main()

    assert calls == [
        (
            Path("ct.nii.gz"),
            Path("seg.nii.gz"),
            tmp_path / "model601",
            tmp_path / "model602",
        )
    ]


def test_cli_validate_exits_nonzero_for_invalid_layout(tmp_path):
    config = Config()
    config.PATH_NNUNET601 = tmp_path / "missing601"
    config.PATH_NNUNET602 = tmp_path / "missing602"

    with pytest.raises(SystemExit) as exc:
        cli.validate(model601=config.PATH_NNUNET601, model602=config.PATH_NNUNET602)

    assert exc.value.code == 1


def test_cli_predict_rejects_same_input_and_output_before_model_loading(monkeypatch, tmp_path, capsys):
    def fail_if_initialized(config):
        raise AssertionError("pipeline should not be initialized")

    monkeypatch.setattr(cli, "VBSegm2StepPipeline", fail_if_initialized)
    input_file = tmp_path / "ct.nii.gz"

    with pytest.raises(SystemExit) as exc:
        cli.predict(input_file, input_file)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Input and output paths cannot be the same" in captured.out


def test_cli_predict_rejects_missing_input_before_model_loading(monkeypatch, tmp_path, capsys):
    def fail_if_initialized(config):
        raise AssertionError("pipeline should not be initialized")

    monkeypatch.setattr(cli, "VBSegm2StepPipeline", fail_if_initialized)
    input_file = tmp_path / "missing.nii.gz"
    output_file = tmp_path / "outputs" / "seg.nii.gz"

    with pytest.raises(SystemExit) as exc:
        cli.predict(input_file, output_file)

    assert exc.value.code == 1
    assert not output_file.parent.exists()
    captured = capsys.readouterr()
    assert "Input file not found" in captured.out
    assert str(input_file) in captured.out


def test_cli_batch_rejects_empty_input_before_model_loading(monkeypatch, tmp_path, capsys):
    def fail_if_initialized(config):
        raise AssertionError("pipeline should not be initialized")

    monkeypatch.setattr(cli, "VBSegm2StepPipeline", fail_if_initialized)
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    with pytest.raises(SystemExit) as exc:
        cli.batch(input_dir, tmp_path / "outputs")

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "No files found" in captured.out


def test_cli_batch_accepts_recursive_glob_before_model_loading(monkeypatch, tmp_path):
    input_dir = tmp_path / "inputs"
    nested = input_dir / "site_a"
    nested.mkdir(parents=True)
    (nested / "ct.nii.gz").write_text("not a real nifti", encoding="utf-8")
    calls = []

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def process_directory(self, input_dir, output_dir, pattern):
            calls.append((input_dir, output_dir, pattern))
            return 1

    monkeypatch.setattr(cli, "VBSegm2StepPipeline", FakePipeline)

    cli.batch(input_dir, tmp_path / "outputs", pattern="**/*.nii.gz")

    assert calls == [(input_dir, tmp_path / "outputs", "**/*.nii.gz")]


def test_cli_batch_does_not_print_duplicate_success(monkeypatch, tmp_path, capsys):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "ct.nii.gz").write_text("not a real nifti", encoding="utf-8")

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def process_directory(self, input_dir, output_dir, pattern):
            print("✅ Batch processing completed: 1/1 files successful")
            return 1

    monkeypatch.setattr(cli, "VBSegm2StepPipeline", FakePipeline)

    cli.batch(input_dir, tmp_path / "outputs")

    captured = capsys.readouterr()
    assert captured.out.count("Batch processing completed") == 1


def test_cli_batch_exits_nonzero_when_all_files_fail(monkeypatch, tmp_path, capsys):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "ct.nii.gz").write_text("not a real nifti", encoding="utf-8")

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def process_directory(self, input_dir, output_dir, pattern):
            print("✅ Batch processing completed: 0/1 files successful")
            return 0

    monkeypatch.setattr(cli, "VBSegm2StepPipeline", FakePipeline)

    with pytest.raises(SystemExit) as exc:
        cli.batch(input_dir, tmp_path / "outputs")

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Batch processing failed: 0/1 files successful" in captured.out


def test_cli_predict_suppresses_traceback_for_expected_model_path_errors(monkeypatch, tmp_path, capsys):
    input_file = tmp_path / "ct.nii.gz"
    output_file = tmp_path / "seg.nii.gz"
    input_file.write_text("not a real nifti", encoding="utf-8")

    def raise_expected_error(config):
        raise ValueError("Model paths are not valid. Please check configuration.")

    monkeypatch.setattr(cli, "VBSegm2StepPipeline", raise_expected_error)

    with pytest.raises(SystemExit) as exc:
        cli.predict(input_file, output_file)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Model paths are not valid" in captured.out
    assert "Traceback" not in captured.out
