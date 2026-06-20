from pathlib import Path

from vbsegm2step.io import load_directory


def test_load_directory_supports_recursive_glob_patterns(tmp_path):
    input_dir = tmp_path / "inputs"
    nested = input_dir / "site_a"
    nested.mkdir(parents=True)
    root_file = input_dir / "root.nii.gz"
    nested_file = nested / "ct.nii.gz"
    ignored_file = nested / "ct.txt"
    root_file.write_text("root", encoding="utf-8")
    nested_file.write_text("nested", encoding="utf-8")
    ignored_file.write_text("ignored", encoding="utf-8")

    assert load_directory(input_dir, "**/*.nii.gz") == [
        root_file,
        nested_file,
    ]


def test_load_directory_default_pattern_is_not_recursive(tmp_path):
    input_dir = tmp_path / "inputs"
    nested = input_dir / "site_a"
    nested.mkdir(parents=True)
    root_file = input_dir / "root.nii.gz"
    nested_file = nested / "ct.nii.gz"
    root_file.write_text("root", encoding="utf-8")
    nested_file.write_text("nested", encoding="utf-8")

    assert load_directory(input_dir) == [root_file]
