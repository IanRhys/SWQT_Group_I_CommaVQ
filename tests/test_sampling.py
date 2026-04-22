import numpy as np

from utils.sampling import multinomial, softmax


# Verifies each row becomes a proper probability distribution when softmax is applied across rows.
def test_softmax_normalizes_rows():
  x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

  y = softmax(x, axis=1)

  np.testing.assert_allclose(y.sum(axis=1), np.ones(2), rtol=1e-6)


# Verifies adding a constant offset to every logit does not change the softmax output.
def test_softmax_is_shift_invariant():
  x = np.array([[1.0, 2.0, 3.0]])

  np.testing.assert_allclose(softmax(x, axis=1), softmax(x + 1000.0, axis=1))


# Verifies the default axis=None behavior normalizes across the entire input array.
def test_softmax_axis_none_normalizes_entire_array():
  x = np.array([1.0, 2.0, 3.0])

  y = softmax(x)

  np.testing.assert_allclose(y.sum(), 1.0, rtol=1e-6)


# Verifies softmax can normalize down columns when axis=0 is requested.
def test_softmax_normalizes_columns_with_axis_zero():
  x = np.array([[1.0, 3.0], [2.0, 4.0]])

  y = softmax(x, axis=0)

  np.testing.assert_allclose(y.sum(axis=0), np.ones(2), rtol=1e-6)


# Verifies the implementation stays numerically stable for very large positive and negative values.
def test_softmax_handles_large_magnitude_values():
  x = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -999.0, -998.0]])

  y = softmax(x, axis=1)

  assert np.all(np.isfinite(y))
  np.testing.assert_allclose(y.sum(axis=1), np.ones(2), rtol=1e-6)


# Verifies softmax preserves the original array shape.
def test_softmax_preserves_input_shape():
  x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

  y = softmax(x, axis=1)

  assert y.shape == x.shape


# Verifies larger logits in the same row map to larger output probabilities.
def test_softmax_preserves_logit_ordering_within_row():
  x = np.array([[1.0, 2.0, 3.0]])

  y = softmax(x, axis=1)

  assert y[0, 2] > y[0, 1] > y[0, 0]


# Verifies multinomial returns one sampled index per row as a column vector.
def test_multinomial_returns_column_vector():
  prob_matrix = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]])

  result = multinomial(prob_matrix.copy())

  assert result.shape == (2, 1)


# Verifies one-hot rows always sample the only possible nonzero index.
def test_multinomial_one_hot_rows_are_deterministic():
  prob_matrix = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

  result = multinomial(prob_matrix.copy())

  np.testing.assert_array_equal(result, np.array([[1], [0]]))


# Verifies multinomial normalizes each row in place before sampling.
def test_multinomial_normalizes_rows_in_place():
  prob_matrix = np.array([[2.0, 3.0], [1.0, 1.0]])

  multinomial(prob_matrix)

  np.testing.assert_allclose(prob_matrix.sum(axis=1), np.ones(2), rtol=1e-6)


# Verifies in-place normalization produces the expected row values for a simple input.
def test_multinomial_normalizes_expected_row_values():
  prob_matrix = np.array([[2.0, 3.0], [1.0, 1.0]])

  multinomial(prob_matrix)

  np.testing.assert_allclose(prob_matrix, np.array([[0.4, 0.6], [0.5, 0.5]]), rtol=1e-6)


# Verifies a single available class always yields index 0 for every row.
def test_multinomial_single_column_always_returns_zero():
  prob_matrix = np.array([[1.0], [5.0], [0.2]])

  result = multinomial(prob_matrix.copy())

  np.testing.assert_array_equal(result, np.zeros((3, 1), dtype=int))


# Verifies the sampler handles the batch-size-1 case and still returns a valid index.
def test_multinomial_supports_single_row_input():
  prob_matrix = np.array([[1.0, 3.0, 6.0]])

  result = multinomial(prob_matrix.copy())

  assert result.shape == (1, 1)
  assert 0 <= result[0, 0] < prob_matrix.shape[1]


# Verifies that fixing the random seed makes the sampled output reproducible.
def test_multinomial_seeded_sampling_is_reproducible():
  prob_matrix = np.array([[1.0, 3.0, 6.0], [2.0, 5.0, 3.0]])

  np.random.seed(0)
  first = multinomial(prob_matrix.copy())
  np.random.seed(0)
  second = multinomial(prob_matrix.copy())

  np.testing.assert_array_equal(first, second)


# Verifies sampled indices always stay within the valid class index range.
def test_multinomial_output_indices_stay_in_bounds():
  prob_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 1.0, 5.0], [7.0, 2.0, 1.0]])

  result = multinomial(prob_matrix.copy())

  assert np.all(result >= 0)
  assert np.all(result < prob_matrix.shape[1])
