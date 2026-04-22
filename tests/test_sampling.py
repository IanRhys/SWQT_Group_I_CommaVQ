import numpy as np
import pytest

from utils.sampling import multinomial, softmax


@pytest.mark.unit
def test_softmax_normalizes_rows():
  """softmax across axis=1 makes each row a proper probability distribution summing to 1."""
  x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

  y = softmax(x, axis=1)

  np.testing.assert_allclose(y.sum(axis=1), np.ones(2), rtol=1e-6)


@pytest.mark.unit
def test_softmax_is_shift_invariant():
  """Adding a constant to every logit does not change the softmax output (numerical-stability property)."""
  x = np.array([[1.0, 2.0, 3.0]])

  np.testing.assert_allclose(softmax(x, axis=1), softmax(x + 1000.0, axis=1))


@pytest.mark.unit
def test_softmax_axis_none_normalizes_entire_array():
  """axis=None normalizes across the entire array, not a single axis."""
  x = np.array([1.0, 2.0, 3.0])

  y = softmax(x)

  np.testing.assert_allclose(y.sum(), 1.0, rtol=1e-6)


@pytest.mark.unit
def test_softmax_normalizes_columns_with_axis_zero():
  """softmax with axis=0 normalizes down columns instead of across rows."""
  x = np.array([[1.0, 3.0], [2.0, 4.0]])

  y = softmax(x, axis=0)

  np.testing.assert_allclose(y.sum(axis=0), np.ones(2), rtol=1e-6)


@pytest.mark.unit
def test_softmax_handles_large_magnitude_values():
  """Stays finite and normalized for extreme positive and negative inputs (stability under edge-case magnitudes)."""
  x = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -999.0, -998.0]])

  y = softmax(x, axis=1)

  assert np.all(np.isfinite(y))
  np.testing.assert_allclose(y.sum(axis=1), np.ones(2), rtol=1e-6)


@pytest.mark.unit
def test_softmax_preserves_input_shape():
  """Output shape matches input shape exactly."""
  x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

  y = softmax(x, axis=1)

  assert y.shape == x.shape


@pytest.mark.unit
def test_softmax_preserves_logit_ordering_within_row():
  """Larger logits map to larger probabilities within the same row."""
  x = np.array([[1.0, 2.0, 3.0]])

  y = softmax(x, axis=1)

  assert y[0, 2] > y[0, 1] > y[0, 0]


@pytest.mark.unit
def test_multinomial_returns_column_vector():
  """Returns one sampled index per row, shaped as a column vector."""
  prob_matrix = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]])

  result = multinomial(prob_matrix.copy())

  assert result.shape == (2, 1)


@pytest.mark.unit
def test_multinomial_one_hot_rows_are_deterministic():
  """One-hot rows always sample the single nonzero index."""
  prob_matrix = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

  result = multinomial(prob_matrix.copy())

  np.testing.assert_array_equal(result, np.array([[1], [0]]))


@pytest.mark.unit
def test_multinomial_normalizes_rows_in_place():
  """Rows are normalized in place before sampling (documents the in-place side effect)."""
  prob_matrix = np.array([[2.0, 3.0], [1.0, 1.0]])

  multinomial(prob_matrix)

  np.testing.assert_allclose(prob_matrix.sum(axis=1), np.ones(2), rtol=1e-6)


@pytest.mark.unit
def test_multinomial_normalizes_expected_row_values():
  """In-place normalization produces the exact expected row values for a simple input."""
  prob_matrix = np.array([[2.0, 3.0], [1.0, 1.0]])

  multinomial(prob_matrix)

  np.testing.assert_allclose(prob_matrix, np.array([[0.4, 0.6], [0.5, 0.5]]), rtol=1e-6)


@pytest.mark.unit
def test_multinomial_single_column_always_returns_zero():
  """A single-class distribution always samples index 0."""
  prob_matrix = np.array([[1.0], [5.0], [0.2]])

  result = multinomial(prob_matrix.copy())

  np.testing.assert_array_equal(result, np.zeros((3, 1), dtype=int))


@pytest.mark.unit
def test_multinomial_supports_single_row_input():
  """Handles batch-size-1 input and returns a valid in-bounds index."""
  prob_matrix = np.array([[1.0, 3.0, 6.0]])

  result = multinomial(prob_matrix.copy())

  assert result.shape == (1, 1)
  assert 0 <= result[0, 0] < prob_matrix.shape[1]


@pytest.mark.unit
def test_multinomial_seeded_sampling_is_reproducible():
  """Fixing the seed makes sampled output reproducible across identical calls."""
  prob_matrix = np.array([[1.0, 3.0, 6.0], [2.0, 5.0, 3.0]])

  np.random.seed(42)
  first = multinomial(prob_matrix.copy())
  np.random.seed(42)
  second = multinomial(prob_matrix.copy())

  np.testing.assert_array_equal(first, second)


@pytest.mark.unit
def test_multinomial_output_indices_stay_in_bounds():
  """Sampled indices always fall within [0, n_classes)."""
  prob_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 1.0, 5.0], [7.0, 2.0, 1.0]])

  result = multinomial(prob_matrix.copy())

  assert np.all(result >= 0)
  assert np.all(result < prob_matrix.shape[1])
