import numpy as np
import SimpleITK as sitk

from vbsegm2step.io import load_image, save_probabilities, save_segmentation


def _write_native_image(path):
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.25, 2.5, 3.75))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(image, str(path))
    return image


def _write_permuted_native_image(path):
    array = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing((1.25, 2.5, 3.75))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(image, str(path))
    return image, array


def test_save_segmentation_restores_native_metadata(tmp_path):
    input_path = tmp_path / "ct.nii.gz"
    output_path = tmp_path / "seg.nii.gz"
    native_image = _write_native_image(input_path)

    _, image_np, image_props = load_image(input_path)
    segmentation = np.ones(image_np.shape, dtype=np.uint8)
    save_segmentation(segmentation, output_path, image_props)

    output_image = sitk.ReadImage(str(output_path))
    assert output_image.GetSpacing() == native_image.GetSpacing()
    assert output_image.GetOrigin() == native_image.GetOrigin()
    assert output_image.GetDirection() == native_image.GetDirection()
    assert output_image.GetSize() == native_image.GetSize()


def test_save_segmentation_restores_non_identity_native_orientation(tmp_path):
    input_path = tmp_path / "ct.nii.gz"
    output_path = tmp_path / "seg.nii.gz"
    native_image, native_array = _write_permuted_native_image(input_path)

    _, image_np, image_props = load_image(input_path)
    assert image_np.shape != native_array.shape

    save_segmentation(image_np.astype(np.uint8), output_path, image_props)

    output_image = sitk.ReadImage(str(output_path))
    output_array = sitk.GetArrayFromImage(output_image)
    assert output_image.GetSpacing() == native_image.GetSpacing()
    assert output_image.GetOrigin() == native_image.GetOrigin()
    assert output_image.GetDirection() == native_image.GetDirection()
    assert output_image.GetSize() == native_image.GetSize()
    assert np.array_equal(output_array, native_array)


def test_save_probabilities_restores_native_metadata(tmp_path):
    input_path = tmp_path / "ct.nii.gz"
    output_path = tmp_path / "prob.nii.gz"
    native_image = _write_native_image(input_path)

    _, image_np, image_props = load_image(input_path)
    probabilities = np.ones((2,) + image_np.shape, dtype=np.float32)
    save_probabilities(probabilities, channel_idx=1, output_path=output_path, image_props=image_props)

    output_image = sitk.ReadImage(str(output_path))
    assert output_image.GetSpacing() == native_image.GetSpacing()
    assert output_image.GetOrigin() == native_image.GetOrigin()
    assert output_image.GetDirection() == native_image.GetDirection()
    assert output_image.GetSize() == native_image.GetSize()
