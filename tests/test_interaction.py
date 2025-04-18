import pytest
import numpy as np
from scdga.interaction import Interaction, LocalInteraction, SpinChannel


def test_localinteraction_adds_correctly():
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    interaction1 = LocalInteraction(mat1)
    interaction2 = LocalInteraction(mat2)
    result = interaction1 + interaction2
    assert np.array_equal(result.mat, mat1 + mat2)


def test_localinteraction_raises_error_on_invalid_permutation():
    mat = np.array([[1, 2], [3, 4]])
    interaction = LocalInteraction(mat)
    with pytest.raises(ValueError, match="Invalid permutation."):
        interaction.permute_orbitals("invalid->permutation")


def test_interaction_handles_channel_transformation():
    mat = np.array([[1, 2], [3, 4]])
    interaction = Interaction(mat, SpinChannel.NONE)
    result = interaction.as_channel(SpinChannel.DENS)
    assert np.array_equal(result.mat, 2 * mat)


def test_interaction_raises_error_on_invalid_channel_transformation():
    mat = np.array([[1, 2], [3, 4]])
    interaction = Interaction(mat, SpinChannel.DENS)
    with pytest.raises(ValueError, match="Cannot transform interaction from channel"):
        interaction.as_channel(SpinChannel.MAGN)


def test_interaction_exponentiates_correctly():
    mat = np.array([[1, 0], [0, 1]])
    interaction = Interaction(mat)
    result = interaction**2
    assert np.array_equal(result.mat, mat @ mat)


def test_interaction_raises_error_on_invalid_exponentiation():
    mat = np.array([[1, 0], [0, 1]])
    interaction = Interaction(mat)
    with pytest.raises(ValueError, match="Exponentiation of Interaction objects only supports positive powers"):
        interaction**0


def test_localinteraction_handles_identity_permutation():
    mat = np.array([[1, 2], [3, 4]])
    interaction = LocalInteraction(mat)
    result = interaction.permute_orbitals("abcd->abcd")
    assert np.array_equal(result.mat, mat)


def test_localinteraction_raises_error_on_invalid_exponentiation_zero():
    mat = np.array([[1, 0], [0, 1]])
    interaction = LocalInteraction(mat)
    with pytest.raises(ValueError, match="Exponentiation of Interaction objects only supports positive powers"):
        interaction**0


def test_interaction_handles_compressed_q_dimension_exponentiation():
    mat = np.random.rand(2, 2, 2, 2, 2)
    interaction = Interaction(mat, has_compressed_q_dimension=True)
    result = interaction**2
    assert result.mat.shape == mat.shape
    assert np.allclose(result.mat, np.einsum("qabcd,qdcef->qabef", mat, mat, optimize=True))


def test_interaction_raises_error_on_invalid_permutation():
    mat = np.random.rand(2, 2, 2, 2, 2)
    interaction = Interaction(mat)
    with pytest.raises(ValueError, match="Invalid permutation."):
        interaction.permute_orbitals("invalid->permutation")
