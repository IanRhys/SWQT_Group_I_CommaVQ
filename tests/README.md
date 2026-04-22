# Testing with pytest or pytest-cov

Run all commands in the this directory (`./tests/`). To get there from the repository root directory, run:

```bash
cd tests
```

Note that the virtual environment and running tests may add files and folders that are not included in version control (local only), such as:
- `./tests/__pycache__/`
- `./tests/.pytest_cache/`
- `./tests/.venv/`
- `./tests/htmlcov/`
- `./tests/.coverage`
- `./utils/__pycache__/`

## Setting up local environment

Run the following to set up the virtual environment with all of the packages for testing. Please note that the last two commands, installing the required packages, may take a few minutes.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r ../requirements.txt
pip install -r test_requirements.txt
```

### Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r ../requirements.txt
pip install -r test_requirements.txt
```

## Running tests

This automatically runs all test files found in this directory (`./tests/`).
- The test files are of the form `test_*.py` and `*_test.py`.
- The executed functions in these files of the form `test_*()` and `*_test()`.

### Running tests without coverage results

To simply run the tests:

```bash
pytest
```

If, say, all 15 tests pass, this will output something like:
- `15 passed in 0.72s`

If, say, 3 of 15 tests fail, this will output something like:
- `3 failed, 12 passed in 0.72s`

### Running tests with coverage results

The following runs the same tests but creates a report of the metric **statement coverage** for the files `./utils/gpt.py` and `./utils/sampling.py`, which are our testing scope.

```bash
pytest --cov-report=html --cov=utils.gpt --cov=utils.sampling
```

This command creates (or updates) a coverage report folder `./tests/htmlcov` and coverage data file `./tests/.coverage`.

The report can be found locally at `./tests/htmlcov/index.html`, which you can view in your browser.
