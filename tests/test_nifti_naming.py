from pathlib import Path

from vbsegm2step.utils import nifti_output_path, nifti_stem


def test_nifti_stem_strips_compound_nii_gz_suffix():
    assert nifti_stem(Path("ct.nii.gz")) == "ct"


def test_nifti_stem_strips_single_nii_suffix():
    assert nifti_stem(Path("ct.nii")) == "ct"


def test_nifti_stem_handles_plain_stem_input():
    assert nifti_stem(Path("ct")) == "ct"


def test_nifti_output_path_uses_single_nii_gz_suffix():
    assert nifti_output_path(Path("ct.nii.gz"), Path("out")) == Path("out/ct.nii.gz")
    assert nifti_output_path(Path("ct.nii"), Path("out")) == Path("out/ct.nii.gz")
    assert nifti_output_path(Path("ct"), Path("out")) == Path("out/ct.nii.gz")


def test_nifti_output_path_preserves_relative_parent_when_root_is_provided():
    assert nifti_output_path(
        Path("inputs/site_a/ct.nii.gz"),
        Path("out"),
        input_root=Path("inputs"),
    ) == Path("out/site_a/ct.nii.gz")
