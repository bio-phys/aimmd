# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add ``examples`` installation target to make install of everything needed for the example notebooks more convenient.
- Add documentation build with sphinx, myst-nb, and the spinx-book-theme. Also hosted on [read the docs](https://aimmd.readthedocs.io/en/latest/).
- Add pylint configuration and github workflow.

## [0.9.2] - 2025-07-29

### Added

- Add `CHANGELOG.md` file

### Changed

- symmetry function compilation with cython now uses the correct types for mdtraj >= v1.11, require mdtraj >= v1.11 for installation.
- update aimmd.distributed (and the aimmd.Storage) to the new (v0.4.1) asyncmd
- aimmd.distributed.committors: Remove hardcoded TRR trajectory type for output trajectories and instead use the same output file type as the engine.

## [0.9.1dev2] - 2025-02-06

### Added

- Shooting Point selectors in aimmd.distributed.spselectors now support uniform selection along the committor and the log-committor. In addition users can supply their own weight functions for the SP selection.
- Vastly expanded docstrings for aimmd.distributed.spselectors

### Changed

- Improved progress printing for TPS simulations in aimmd.distributed
- Updated jupyter example notebooks for TPS simulations with aimmd.distributed

### Fixed

- Ensure the correct training feedback for model when reinitializing a TPS simulation in aimmd.distributed

## [0.9.1dev1] - 2025-01-21

### Added

- First release on PyPi

[unreleased]: https://github.com/bio-phys/aimmd/compare/v0.9.2...HEAD
[0.9.2]: https://github.com/bio-phys/aimmd/compare/v0.9.1dev2...v0.9.2
[0.9.1dev2]: https://github.com/bio-phys/aimmd/compare/v0.9.1dev1...v0.9.1dev2
[0.9.1dev1]: https://github.com/bio-phys/aimmd/releases/tag/v0.9.1dev1
