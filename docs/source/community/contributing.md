# Contributing to aimmd

There are many ways you can contribute to aimmd, be it by writing or expanding its documentation, with with bug reports or feature requests, or by submitting patches for new or fixed behavior.

## Bug reports and feature requests

If you have encountered a problem using aimmd or have an idea for a new feature, please open a [Github issue].
For bug reports, please include the full error message and, if possible, a (minimal) example resulting in the bug.

## Contribute code

The aimmd source code is managed using git and [hosted on Github][Github]. The recommended way for contributors to submit code is to fork this repository and open a [pull request][Github pr].

(contributing-getting-started)=
### Getting started

Before starting a patch, it is recommended to check for open [issues][Github issue] or [pull requests][Github pr] relating to the topic.

The basic steps to start developing on aimmd are:

1. [Fork](https://github.com/bio-phys/aimmd/fork) the repository on Github.
2. (Optional but highly recommended) Create and activate a python virtual environment using your favorite environment manager (e.g. virtualenv, conda).
3. Clone the repository and install it in editable mode using the dev target (see the [installation instructions](#developer-installation)).
4. Create a new working branch and write your code. Please try to write tests for your code and make sure that all tests pass (see [below](#tests-and-linting)).
5. Add a bullet point to `CHANGELOG.md` if the fix or feature is not trivial, then commit.
6. Push the changes and open a [pull request on Github][Github pr].

### Coding style

Please follow these guidelines when writing code for aimmd:

- Try to use the same code style as the rest of the project.
- Update `CHANGELOG.md` for non-trivial changes. If your changes alter existing behavior, please document this.
- New features should be documented. If possible, also include them in the example notebooks or add a new example notebook showcasing the feature.
- Add appropriate unit tests.

(tests-and-linting)=
### Tests and linting

#### Tests

aimmd uses [pytest] for its tests. When you install aimmd using the [developer installation target](#developer-installation), it will also install everything needed to run tests and get a coverage report (i.e. [coverage] and [pytest-cov]). To, e.g., get a html coverage report you can then run the tests as

```bash
pytest --cov=aimmd --cov-report=html
```

#### Linting

aimmd uses [pylint] to perform linting and ensure code quality. It will also be installed with the [developer installation](#developer-installation) such that you can run it locally and it will also be run automatically on pull requests on github.

```{important}
The current configuration fails only if [pylint] finds an [error](https://pylint.readthedocs.io/en/latest/user_guide/messages/messages_overview.html#error-category) or if the [pylint] rating decreases to below 7.
Note that this somewhat relaxed setting is due to historical code and will be tightened in the future as soon as we refactored the old code.
**All new code should strive to not add any [pylint] messages, but at least not result in additional error or warning messages.**
```

You can run pytest using something like the following:

```bash
pylint --output-format="colorized" aimmd
# or, to also get detailed reports and summaries printed at the end use:
pylint --output-format="colorized" --reports=y aimmd
# or, to only see the errors and warnings (disable refactor and convention):
pylint --output-format="colorized" --disable="R,C" aimmd
```

```{note}
You need to be somewhere in the repository folder hierarchy for the aimmd-specific pylint configuration (stored in `pyproject.toml`) to take effect.
If you are not at the correct spot in the folder hierarchy the command will (probably) still work, but might give a different result due to missing extensions and differing settings
```

## Contribute documentation

To contribute documentation you will need to modify the source files in the `docs/source` folder. To get started follow the steps in [Getting started](#contributing-getting-started), but instead of writing code (and tests) modify the documentation source files. You can then build the documentation locally (see the [installation instructions](#documentation-installation)) to check that everything works as expected. Additionally, after submitting a [pull request][Github pr], you can preview your changes as they will be rendered on readthedocs directly.

[coverage]: https://pypi.org/project/coverage/
[Github]: https://github.com/bio-phys/aimmd
[Github issue]: https://github.com/bio-phys/aimmd/issues
[Github pr]: https://github.com/bio-phys/aimmd/pulls
[pylint]: https://pylint.readthedocs.io/en/latest/index.html
[pytest]: https://docs.pytest.org/en/latest/
[pytest-cov]: https://pypi.org/project/pytest-cov/
