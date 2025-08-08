from unittest.mock import patch

import pytest

from scdga.interaction import LocalInteraction
from scdga.local_four_point import LocalFourPoint
import numpy as np

from scdga.n_point_base import SpinChannel


@pytest.mark.parametrize("n", [1, 2, 3])
def test_exponentiation_with_positive_power_1(n):
    mat = np.random.rand(2,2,2,2,21,20) + 1j * np.random.rand(2,2,2,2,21,20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=1)
    identity = LocalFourPoint.identity_like(obj)
    result = obj.pow(n, identity)
    expected = obj
    for _ in range(n - 1):
        expected = expected @ obj
    assert np.allclose(result.mat, expected.mat, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_exponentiation_with_positive_power_2(n):
    mat = np.random.rand(2,2,2,2,21,20,20) + 1j * np.random.rand(2,2,2,2,21,20,20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=2)
    identity = LocalFourPoint.identity_like(obj)
    result = obj.pow(n, identity)
    expected = obj
    for _ in range(n - 1):
        expected = expected @ obj
    assert np.allclose(result.mat, expected.mat, rtol=1e-2)


def test_exponentiation_with_zero_power_returns_identity_1():
    mat = np.random.rand(2,2,2,2,21,20) + 1j * np.random.rand(2,2,2,2,21,20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=1)
    identity = LocalFourPoint.identity_like(obj)
    result = obj.pow(0, identity)
    assert np.allclose(result.mat, identity.mat, rtol=1e-2)


def test_exponentiation_with_zero_power_returns_identity_2():
    mat = np.random.rand(2,2,2,2,21,20,20) + 1j * np.random.rand(2,2,2,2,21,20,20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=2)
    identity = LocalFourPoint.identity_like(obj)
    result = obj.pow(0, identity)
    assert np.allclose(result.mat, identity.mat, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_exponentiation_with_negative_power_1(n):
    mat = np.random.rand(2,2,2,2,21,20) + 1j * np.random.rand(2,2,2,2,21,20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=1)
    identity = LocalFourPoint.identity_like(obj)
    result = obj.pow(-n, identity)
    expected = obj.invert()
    for _ in range(n-1):
        expected = expected @ obj.invert()
    assert np.allclose(result.mat, expected.mat, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_exponentiation_with_negative_power_2(n):
    mat = np.random.rand(2,2,2,2,21,20,20) + 1j * np.random.rand(2,2,2,2,21,20,20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=2)
    identity = LocalFourPoint.identity_like(obj)
    result = obj.pow(-n, identity)
    expected = obj.invert()
    for _ in range(n - 1):
        expected = expected @ obj.invert()
    assert np.allclose(result.mat, expected.mat, rtol=1e-2)


def test_exponentiation_with_non_integer_power_raises_error():
    mat = np.random.rand(2, 2, 2, 2, 21, 20, 20) + 1j * np.random.rand(2, 2, 2, 2, 21, 20, 20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=2)
    identity = LocalFourPoint.identity_like(obj)
    with pytest.raises(ValueError):
        obj.pow(2.5, identity)


def test_symmetrizes_square_matrix_correctly():
    mat = np.array([[[[[[1, 2.5], [2.5, 4]]]]]])
    obj = LocalFourPoint(mat)
    result = obj.symmetrize_v_vp()
    expected = np.array([[[[[[1, 2.5], [2.5, 4]]]]]])
    assert np.allclose(result.mat, expected, rtol=1e-2)


def test_symmetrizes_random_matrix_correctly():
    mat = np.random.rand(2, 2, 2, 2, 5, 3, 3)
    obj = LocalFourPoint(mat)
    result = obj.symmetrize_v_vp()
    expected = 0.5 * (mat + np.swapaxes(mat, -1, -2))
    assert np.allclose(result.mat, expected, rtol=1e-2)


def test_handles_symmetric_matrix_without_modification():
    mat = np.array([[[[[[1, 2], [2, 4]]]]]])
    obj = LocalFourPoint(mat)
    result = obj.symmetrize_v_vp()
    assert np.allclose(result.mat, mat, rtol=1e-2)


def test_raises_error_for_non_square_last_two_axes():
    mat = np.random.rand(2, 2, 2, 2, 5, 3, 4)
    obj = LocalFourPoint(mat)
    with pytest.raises(ValueError):
        obj.symmetrize_v_vp()


def test_raises_error_for_not_having_two_vn_dimensions():
    mat = np.random.rand(2, 2, 2, 2, 5, 3)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    with pytest.raises(ValueError):
        obj.symmetrize_v_vp()


def test_sums_over_orbitals_correctly_1():
    mat = np.random.rand(2, 2, 2, 2, 5, 3)
    obj = LocalFourPoint(mat)
    result = obj.sum_over_orbitals("abcd->ad")
    assert result.mat.shape == (2, 2, 5, 3)
    assert np.allclose(result.mat, np.sum(mat, axis=(1, 2)), rtol=1e-2)


def test_sums_over_orbitals_correctly_2():
    mat = np.random.rand(2, 2, 2, 2, 5, 3, 3)
    obj = LocalFourPoint(mat)
    result = obj.sum_over_orbitals("abcd->ad")
    assert result.mat.shape == (2, 2, 5, 3, 3)
    assert np.allclose(result.mat, np.sum(mat, axis=(1,2)), rtol=1e-2)


def test_raises_error_for_invalid_orbital_contraction_format():
    mat = np.random.rand(2, 2, 2, 2, 5, 3, 3)
    obj = LocalFourPoint(mat)
    with pytest.raises(ValueError):
        obj.sum_over_orbitals("abc->ad")


def test_handles_no_orbital_reduction():
    mat = np.random.rand(2, 2, 2, 2, 5, 3, 3)
    obj = LocalFourPoint(mat)
    result = obj.sum_over_orbitals("abcd->abcd")
    assert np.allclose(result.mat, mat, rtol=1e-2)


def test_reduces_orbital_dimensions_correctly_1():
    mat = np.random.rand(3, 3, 3, 3, 5, 4, 4)
    obj = LocalFourPoint(mat)
    result = obj.sum_over_orbitals("abcd->a")
    assert result.mat.shape == (3, 5, 4, 4)
    assert np.allclose(result.mat, np.sum(mat, axis=(1, 2, 3)), rtol=1e-2)


def test_reduces_orbital_dimensions_correctly_2():
    mat = np.random.rand(3, 3, 3, 3, 5, 4, 4)
    obj = LocalFourPoint(mat)
    result = obj.sum_over_orbitals("abcd->ab")
    assert result.mat.shape == (3,3, 5, 4, 4)
    assert np.allclose(result.mat, np.sum(mat, axis=(2, 3)), rtol=1e-2)


def test_reduces_orbital_dimensions_correctly_3():
    mat = np.random.rand(3, 3, 3, 3, 5, 4, 4)
    obj = LocalFourPoint(mat)
    result = obj.sum_over_orbitals("abcd->abc")
    assert result.mat.shape == (3,3,3, 5, 4, 4)
    assert np.allclose(result.mat, np.sum(mat, axis=(3,)), rtol=1e-2)


def test_sums_over_single_vn_dimension_correctly_1():
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    beta = 10.0
    result = obj.sum_over_vn(beta, axis=(-1,))
    expected_mat = 1 / beta * np.sum(mat, axis=-1)
    assert np.allclose(result.mat, expected_mat, rtol=1e-2)
    assert result.num_vn_dimensions == 0


@pytest.mark.parametrize("n", [1, 2])
def test_sums_over_single_vn_dimension_correctly_2(n):
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    beta = 10.0
    result = obj.sum_over_vn(beta, axis=(-n,))
    expected_mat = 1 / beta * np.sum(mat, axis=(-n,))
    assert np.allclose(result.mat, expected_mat, rtol=1e-2)
    assert result.num_vn_dimensions == 1


def test_sums_over_multiple_vn_dimensions_correctly():
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    beta = 10.0
    result = obj.sum_over_vn(beta, axis=(-2, -1))
    expected_mat = 1 / beta**2 * np.sum(mat, axis=(-2, -1))
    assert np.allclose(result.mat, expected_mat, rtol=1e-2)
    assert result.num_vn_dimensions == 0


def test_raises_error_when_summing_over_too_many_vn_dimensions():
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    beta = 10.0
    with pytest.raises(ValueError):
        obj.sum_over_vn(beta, axis=(-2, -1))


def test_sums_over_all_vn_with_double_vn_dimensions_correctly():
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    beta = 10.0
    result = obj.sum_over_all_vn(beta)
    expected_mat = 1 / beta**2 * np.sum(mat, axis=(-2, -1))
    assert np.allclose(result.mat, expected_mat, rtol=1e-2)
    assert result.num_vn_dimensions == 0


def test_sums_over_all_vn_with_single_vn_dimension_correctly():
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    beta = 10.0
    result = obj.sum_over_all_vn(beta)
    expected_mat = 1 / beta * np.sum(mat, axis=-1)
    assert np.allclose(result.mat, expected_mat, rtol=1e-2)
    assert result.num_vn_dimensions == 0


def test_handles_no_vn_dimensions_without_modification_for_sum():
    mat = np.random.rand(2, 2, 2, 2, 5)
    obj = LocalFourPoint(mat, num_vn_dimensions=0)
    beta = 10.0
    result = obj.sum_over_all_vn(beta)
    assert np.allclose(result.mat, mat, rtol=1e-2)
    assert result.num_vn_dimensions == 0

def test_contracts_legs_correctly_with_two_vn_dimensions():
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    beta = 10.0
    result = obj.contract_legs(beta)
    assert result.mat.shape == (2, 2, 5)
    assert np.allclose(result.mat, 1./beta**2*np.einsum("abcdefg->ade", mat), rtol=1e-2)

def test_raises_error_when_contracting_legs_with_invalid_vn_dimensions():
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    beta = 10.0
    with pytest.raises(ValueError):
        obj.contract_legs(beta)

def test_calls_sum_over_all_vn_and_sum_over_orbitals():
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    beta = 10.0

    with patch.object(LocalFourPoint, 'sum_over_all_vn', autospec=True) as mock_sum_vn, \
         patch.object(LocalFourPoint, 'sum_over_orbitals', autospec=True) as mock_sum_orb:
        mock_sum_vn.return_value = obj
        mock_sum_orb.return_value = obj

        obj.contract_legs(beta)

        mock_sum_vn.assert_called_once_with(obj, beta)
        mock_sum_orb.assert_called_once_with(obj, "abcd->ad")


def test_converts_to_compound_indices_with_no_vn_dimensions():
    mat = np.random.rand(2, 2, 2, 2, 5)
    obj = LocalFourPoint(mat, num_vn_dimensions=0)
    result = obj.to_compound_indices()
    assert result.mat.shape == (5, 4, 4)
    assert np.allclose(result.mat, mat.transpose(4, 0, 1, 3, 2).reshape(5,4,4), rtol=1e-2)


def test_converts_to_compound_indices_with_one_vn_dimension():
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    result = obj.to_compound_indices()
    assert result.mat.shape == (5, 16, 16)


def test_calls_extend_vn_to_diagonal_with_one_vn_dimension_and_executes_original():
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    with patch.object(LocalFourPoint, 'extend_vn_to_diagonal', autospec=True, wraps=LocalFourPoint.extend_vn_to_diagonal) as mock_extend:
        result = obj.to_compound_indices()
        mock_extend.assert_called_once_with(obj)
        assert result.mat.shape == (5, 16, 16)


def test_converts_to_compound_indices_with_two_vn_dimensions():
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    result = obj.to_compound_indices()
    assert np.allclose(result.mat, mat.transpose(4, 0, 1, 5, 3, 2, 6).reshape(5,16,16), rtol=1e-2)


def test_raises_error_for_missing_bosonic_frequencies():
    mat = np.random.rand(2, 2, 2, 2, 4, 4)
    obj = LocalFourPoint(mat, num_wn_dimensions=0)
    with pytest.raises(ValueError):
        obj.to_compound_indices()


def test_handles_already_compound_indices_without_modification():
    mat = np.random.rand(5, 4, 4)
    obj = LocalFourPoint(mat, num_wn_dimensions=1, num_vn_dimensions=2)
    result = obj.to_compound_indices()
    assert np.allclose(result.mat, mat, rtol=1e-2)


@pytest.mark.parametrize("num_vn_dimensions,expected_shape,compound_shape", [
    (0, (2, 2, 2, 2, 5), (5,4,4)),
    (2, (2, 2, 2, 2, 5, 4, 4), (5,16,16))
])
def test_converts_compound_indices_to_full_indices_correctly(num_vn_dimensions, expected_shape, compound_shape):
    mat = np.random.rand(*expected_shape)
    obj = LocalFourPoint(mat, num_vn_dimensions=num_vn_dimensions)
    obj = obj.to_compound_indices()
    assert obj.mat.shape == compound_shape
    result = obj.to_full_indices()
    assert result.mat.shape == expected_shape


def test_converts_compound_indices_to_full_indices_correctly_for_one_vn_dimension():
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    obj = obj.to_compound_indices()
    assert obj.mat.shape == (5, 16, 16)
    result = obj.to_full_indices()
    assert result.mat.shape == (2, 2, 2, 2, 5, 4, 4)
    assert result.num_vn_dimensions == 2
    assert np.allclose(mat, result.take_vn_diagonal().mat, rtol=1e-2)


def test_raises_error_for_invalid_current_shape():
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    with pytest.raises(ValueError, match="Converting to full indices with shape .* not supported."):
        obj.to_full_indices()


@pytest.mark.parametrize("num_wn_dimensions,num_vn_dimensions,shape", [
    (0,0,(2,2,2,2)),
    (1,0,(2,2,2,2,5)),
    (0,1,(2,2,2,2,4)),
    (1,1,(2,2,2,2,5,4)),
    (0,2,(2,2,2,2,4,4)),
    (1,2,(2,2,2,2,5,4,4)),
])
def test_returns_original_object_when_already_in_full_indices(num_wn_dimensions,num_vn_dimensions,shape):
    mat = np.random.rand(*shape)
    obj = LocalFourPoint(mat, num_wn_dimensions=num_wn_dimensions, num_vn_dimensions=num_vn_dimensions)
    result = obj.to_full_indices()
    assert result.mat.shape == shape
    assert np.allclose(result.mat, mat, rtol=1e-2)
    assert result.num_wn_dimensions == num_wn_dimensions
    assert result.num_vn_dimensions == num_vn_dimensions


def test_handles_diagonal_extraction_for_single_vn_dimension():
    mat = np.random.rand(5, 16, 16)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    obj._original_shape = (2, 2, 2, 2, 5, 4)
    result = obj.to_full_indices()
    assert result.mat.shape == (2, 2, 2, 2, 5, 4)

    mat = mat.reshape((5,) + (2,2,4,) * 2).transpose(1, 2, 5, 4, 0, 3, 6).diagonal(axis1=-2, axis2=-1)
    assert np.allclose(result.mat, mat, rtol=1e-2)


def test_raises_error_for_invalid_bosonic_frequency_dimensions():
    mat = np.random.rand(1,16,16)
    obj = LocalFourPoint(mat, num_wn_dimensions=0, num_vn_dimensions=2)
    with pytest.raises(ValueError):
        obj.to_full_indices()


@pytest.mark.parametrize("full_niw_range", [True, False])
def test_assures_invert_calls_to_half_niw_range_to_compound_indices_and_to_full_indices(full_niw_range):
    mat = np.random.rand(2, 2, 2, 2, 11, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2, full_niw_range=full_niw_range)
    with patch.object(LocalFourPoint, 'to_half_niw_range', autospec=True, wraps=LocalFourPoint.to_half_niw_range) as mock_half_niw, \
         patch.object(LocalFourPoint, 'to_compound_indices', autospec=True, wraps=LocalFourPoint.to_compound_indices) as mock_compound, \
         patch.object(LocalFourPoint, 'to_full_indices', autospec=True, wraps=LocalFourPoint.to_full_indices) as mock_full:
        obj.invert()
        mock_half_niw.assert_called_once()
        mock_compound.assert_called_once()
        mock_full.assert_called_once()


def test_assures_invert_always_returns_half_niw_range():
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    with patch.object(LocalFourPoint, 'to_half_niw_range', autospec=True, wraps=LocalFourPoint.to_half_niw_range) as mock_half_niw:
        result = obj.invert()
        mock_half_niw.assert_called()
        assert not result.full_niw_range


def test_assures_invert_calls_to_full_indices_with_default_shape():
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    with patch.object(LocalFourPoint, 'to_full_indices', autospec=True, wraps=LocalFourPoint.to_full_indices) as mock_full:
        obj.invert()
        args, kwargs = mock_full.call_args
        # shape=None ist default
        assert kwargs.get('shape', None) is None


def test_multiplies_two_objects_with_no_vn_dimensions_correctly():
    mat1 = np.random.rand(2, 2, 2, 2, 5)
    mat2 = np.random.rand(2, 2, 2, 2, 5)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=0)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=0)
    result1 = obj1 @ obj2
    result2 = obj2 @ obj1
    expected1 = np.einsum("abcdw,dcefw->abefw", mat1, mat2, optimize=True)
    expected2 = np.einsum("abcdw,dcefw->abefw", mat2, mat1, optimize=True)
    assert np.allclose(result1.mat, expected1, rtol=1e-2)
    assert np.allclose(result2.mat, expected2, rtol=1e-2)


def test_multiplies_two_objects_with_one_vn_dimension_correctly():
    mat1 = np.random.rand(2, 2, 2, 2, 5, 4)
    mat2 = np.random.rand(2, 2, 2, 2, 5, 4)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=1)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=1)
    result1 = obj1 @ obj2
    result2 = obj2 @ obj1
    expected1 = np.einsum("abcdwv,dcefwv->abefwv", mat1, mat2, optimize=True)
    expected2 = np.einsum("abcdwv,dcefwv->abefwv", mat2, mat1, optimize=True)
    assert np.allclose(result1.mat, expected1, rtol=1e-2)
    assert np.allclose(result2.mat, expected2, rtol=1e-2)


@pytest.mark.parametrize("full_niw_range1,full_niw_range2", [(False, False), (True, True), (False, True), (True, False)])
def test_assures_matmul_calls_to_compound_indices_for_two_vn_dimensions(full_niw_range1,full_niw_range2):
    mat1 = np.random.rand(2, 2, 2, 2, 21 if full_niw_range1 else 11, 4, 4)
    mat2 = np.random.rand(2, 2, 2, 2, 21 if full_niw_range2 else 11, 4, 4)
    count_full_niw_range = [full_niw_range1, full_niw_range2].count(True)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=2, full_niw_range=full_niw_range1)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=2, full_niw_range=full_niw_range2)
    with (patch.object(LocalFourPoint, 'to_compound_indices', autospec=True, wraps=LocalFourPoint.to_compound_indices) as mock_compound,
          patch.object(LocalFourPoint, 'to_half_niw_range', autospec=True, wraps=LocalFourPoint.to_half_niw_range) as mock_half_niw,
          patch.object(LocalFourPoint, 'to_full_niw_range', autospec=True, wraps=LocalFourPoint.to_full_niw_range) as mock_full_niw,
          patch.object(LocalFourPoint, 'to_full_indices', autospec=True, wraps=LocalFourPoint.to_full_indices) as mock_to_full_indices,
          patch("numpy.matmul", autospec=True, wraps=np.matmul) as mock_matmul):
        obj1 @ obj2
        assert mock_half_niw.call_count == 2
        mock_matmul.assert_called_once()
        assert mock_compound.call_count == 2
        assert mock_full_niw.call_count == count_full_niw_range
        assert mock_to_full_indices.call_count == 3


def test_raises_error_for_invalid_multiplication_with_non_local_four_point():
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    with pytest.raises(ValueError, match="Multiplication .* not supported."):
        obj @ np.random.rand(4, 4)


def test_handles_multiplication_with_local_interaction_correctly():
    mat1 = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    mat2 = np.random.rand(2,2,2,2)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=2)
    obj2 = LocalInteraction(mat2)
    result1 = obj1 @ obj2
    result2 = obj2 @ obj1
    expected1 = np.einsum("abcdwvp,dcef->abefwvp", mat1, mat2, optimize=True)
    expected2 = np.einsum("abcd,dcefwvp->abefwvp", mat2, mat1, optimize=True)
    assert np.allclose(result1.mat, expected1, rtol=1e-2)
    assert np.allclose(result2.mat, expected2, rtol=1e-2)


def test_multiplies_objects_with_mixed_vn_dimensions_correctly():
    mat1 = np.random.rand(2, 2, 2, 2, 5, 4)
    mat2 = np.random.rand(2, 2, 2, 2, 5)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=1)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=0)
    result1 = obj1 @ obj2
    result2 = obj2 @ obj1
    expected1 = np.einsum("abcdwv,dcefw->abefwv", mat1, mat2, optimize=True)
    expected2 = np.einsum("abcdw,dcefwv->abefwv", mat2, mat1, optimize=True)
    assert np.allclose(result1.mat, expected1, rtol=1e-2)
    assert np.allclose(result2.mat, expected2, rtol=1e-2)
    assert result1.num_vn_dimensions == 1
    assert result2.num_vn_dimensions == 1


def test_multiplies_with_full_niw_range_and_restores_shape():
    mat1 = np.random.rand(2, 2, 2, 2, 21, 4, 4)
    mat2 = np.random.rand(2, 2, 2, 2, 21, 4, 4)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=2, full_niw_range=True)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=2, full_niw_range=True)
    result = obj1 @ obj2
    assert result.mat.shape == (2, 2, 2, 2, 11, 4, 4)
    assert not result.full_niw_range
