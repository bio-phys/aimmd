## Pathmovers

The module {mod}`aimmd.distributed.pathmovers` contains all {class}`PathMover <aimmd.distributed.pathmovers.PathMover>` implementations.
A pathmover generates a new trial trajectory, usually from a given input Monte Carlo step containing a transition trajectory.

While many different move types to generate new trial paths are possible, currently only {class}`ShootingPathMover <aimmd.distributed.pathmovers.ShootingPathMover>` variants are implemented.
All {class}`ShootingPathMover <aimmd.distributed.pathmovers.ShootingPathMover>` need a {class}`SPSelector <aimmd.distributed.spselectors.SPSelector>` to provide shooting points (SPs) to initialize the trajectory propagation.

**TODO: a few words on other move types and a theory/background section**

```{toctree}
:maxdepth: 2
:caption: PathMover classes

Two way shooting pathmovers <pathsamp_classdoc/pathmovers_twoway>
Abstract base classes for pathmovers <pathsamp_classdoc/pathmovers_abcs>
```
