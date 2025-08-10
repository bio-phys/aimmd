# Committor simulations

The {mod}`aimmd.distributed` contains a class to perform committor simulations for an arbitrary number of configurations and trials in parallel.
The {class}`CommittorSimulation <aimmd.distributed.CommittorSimulation>` allows to perform committor trials in one direction only, or optionally will also perform for every trial a propagation with inverted momenta at the shooting point.
This feature can be particularly useful to generate initial transitions as input for a transition path sampling simulation.
The input configurations for the {class}`CommittorSimulation <aimmd.distributed.CommittorSimulation` are expected to be wrapped as {class}`CommittorConfiguration  <aimmd.distributed.CommittorConfiguration` dataclasses.
The MD engine and other propagation options for the trials are defined (potentially on a per-configuration basis) via the dataclass {class}`CommittorEngineSpec <aimmd.distributed.CommittorEngineSpec`.
Note, that a number of options to the {class}`CommittorSimulation <aimmd.distributed.CommittorSimulation>` can be specified on a per-configuration basis (in addition to the {class}`CommittorEngineSpec <aimmd.distributed.CommittorEngineSpec`).
See the docstring of the {class}`CommittorSimulation <aimmd.distributed.CommittorSimulation>` for more.

```{seealso}
The example notebooks on the {doc}`CommittorSimulation </examples_link/distributed/CommittorSimulation>`.
```

```{eval-rst}
.. autoclass:: aimmd.distributed.CommittorConfiguration

.. autoclass:: aimmd.distributed.CommittorEngineSpec

.. autoclass:: aimmd.distributed.CommittorSimulation
    :class-doc-from: both
    :member-order: groupwise
    :members:
    :inherited-members:
```
