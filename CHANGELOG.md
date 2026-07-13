# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `aimmd.ops`: The density collector is now attached to the `aimmd.ops.hooks.DensityCollectionHook` and uses the new density collector implementation from `aimmd.base.density_collection`
- `aimmd.distributed`: Introduced `PathSamplingSimStateInfo` dataclass to make it easier to access any information about the current state of the path sampling simulation at any level when performing a MC step.
- `aimmd.distributed`: Improve `aimmd.distributed.pathmovers.PathMover` class inheritance structure, shooting pathmovers now use uninitialized shooting point selectors as init argument and use the additional `sp_selector_kwargs` argument to specify any arguments to the shooting point selector.
- `aimmd.distributed`: The density adaption is now a part of the `aimmd.distributed.SPSelector` classes, the additional density adaption scheme "lazzeri" has been added.
- `aimmd.distributed.Brain`: use tqdm progress bars
- `aimmd.distributed.pathmovers`: shooting pathmovers now use the `aimmd.distributed.MDEngineSpec` dataclass as input argument to specify all the MD engine/propagation options together instead of as previously specifying each argument separately
- Complete rewrite of `aimmd.distributed.CommittorSimulation`, update and improve corresponding example notebook.

### Removed

- `aimmd.keras`: removed complete submodule (has not seen updates for a long time, the `aimmd.pytorch` submodule is better maintained and offers better functionality)
- `aimmd.ops`: removed unused legacy code
- `aimmd.base.rcmodel`: Removed old TrajectoryDensityCollector (attached to model) as it is no longer needed with the ops-based aimmd also now using the new `aimmd.base.density_collector.DensityCollector` class
- `aimmd.distributed`: The density collection `BrainTask` has been removed as it is no longer needed due to the rework of density collection (see changed).

## [0.9.3] - 2025-08-03

### Added

- Add ``examples`` installation target to make install of everything needed for the example notebooks more convenient.
- Add documentation build with sphinx, myst-nb, and the spinx-book-theme. Also hosted on [read the docs](https://aimmd.readthedocs.io/en/latest/).
- Add pylint configuration and github workflow.

### Changed

- Reorganize example notebooks. Host large input files used in examples on figshare, download them in the examples only if needed. Document example notebooks with readme files and add them to the documentation.

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

[unreleased]: https://github.com/bio-phys/aimmd/compare/v0.9.3...HEAD
[0.9.3]: https://github.com/bio-phys/aimmd/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/bio-phys/aimmd/compare/v0.9.1dev2...v0.9.2
[0.9.1dev2]: https://github.com/bio-phys/aimmd/compare/v0.9.1dev1...v0.9.1dev2
[0.9.1dev1]: https://github.com/bio-phys/aimmd/releases/tag/v0.9.1dev1
