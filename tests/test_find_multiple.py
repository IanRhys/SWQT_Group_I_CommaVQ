""""
FILE:  test the function find_multiple_tests() in gpt.py. 

Func being tested: Finds the next multiple of k that is greater than or equal to n.

HOW-TO-RUN:  
    python3 -m pytest tests/test_find_multiple.py  (in root)
"""
from utils.gpt import find_multiple



# tests that a number that is not a multiple of k is rounded up correctly
def test_find_multiple_rounds_up():
    assert find_multiple(10, 8) == 16


# tests that a number already divisible by k is returned unchanged
def test_find_multiple_returns_same_if_multiple():
    assert find_multiple(16, 8) == 16


# tests correct behavior for small input values
def test_find_multiple_small_values():
    assert find_multiple(3, 4) == 4


# tests edge case where k = 1 (every number should be unchanged)
def test_find_multiple_k_one():
    assert find_multiple(7, 1) == 7


# tests edge case where input value is zero
def test_find_multiple_zero():
    assert find_multiple(0, 8) == 0


# tests behavior with realistic larger model-related values
def test_find_multiple_large_number():
    assert find_multiple(1025, 8) == 1032