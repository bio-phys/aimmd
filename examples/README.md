# Example Notebooks

```{toctree}
:maxdepth: 2
:caption: All notebooks

aimmd.distributed <distributed/README>
classic <classic/README>
```

## Example notebooks for ``aimmd.distributed``

The example notebooks found in the ``distributed`` folder (and its subfolders) are all concerned with how to setup, run, and analyze a (large) number of TPS simulations simultaneously, all steered by one common committor model.

## Example notebooks for "classic" TPS using openpathsampling

The example notebooks in the ``classic`` folder (and its subfolders) teach you how to use aimmd to perform AI-guided TPS with [openpathsampling].
As opposed to the simulations performed using the ``aimmd.distributed`` module, here one committor model steers only one TPS (or other path sampling simulation) and since the actual (T)PS simulation is performed with [openpathsampling] all bonuses and constraints of using it apply.
There are currently examples for 2D toy potentials, LiCl dissociation (including solvent), and capped alanine dipeptide.
