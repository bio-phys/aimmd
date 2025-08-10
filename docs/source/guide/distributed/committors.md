# Committor simulations

The :mod:`aimmd.distributed` contains a class to perform committor simulations for an arbitrary number of configurations and trials in parallel.
The :class:`CommittorSimulation <aimmd.distributed.CommittorSimulation>` allows to perform committor trials in one direction only, or optionally will also perform for every trial a propagation with inverted momenta at the shooting point.
This feature can be particularly useful to generate initial transitions as input for a transition path sampling simulation.

```{seealso}
The example notebooks on the {doc}`CommittorSimulation </examples_link/distributed/CommittorSimulation>`.
```

```{eval-rst}
.. autoclass:: aimmd.distributed.CommittorConfiguration

.. autoclass:: aimmd.distributed.CommittorEngineSpec

.. autoclass:: aimmd.distributed.CommittorSimulation
    :class-doc-from: both
```
