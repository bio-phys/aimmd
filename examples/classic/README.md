# aimmd example notebooks for (classic) TPS with openpathsampling

```{toctree}
:maxdepth: 1
:caption: "classic" notebooks

Toy system(s) <toy_systems/README>
Lithium chloride dissociation <LiCl_single_ion_dissociation/README>
Capped alanine dipeptide <capped_alanine_dipeptide/README>
```

This folder contains example notebooks showing how to use aimmd to perform AI-guided TPS with [openpathsampling].
As opposed to the simulations performed using the ``aimmd.distributed`` module, here one committor model steers only one TPS (or other path sampling simulation) and since the actual (T)PS simulation is performed with [openpathsampling] all bonuses and constraints of using it apply.
Note that the tensorflow models will most likely be soon deprecated, so its probably best to have a look at the pytorch models preferentially.

## Toy system(s)

This folder contains notebooks showcasing aimmd usage on a 2D (plus hidden/irrelevant orthogonal dimensions) potential.
The notebooks start from performing the TPS simulation and end with a symbolic regression finding a simple mathematical description of the transition in the most relevant coordinates.
There are various variants of these notebooks using different neural network architectures and machine learning backends, they do not differ in any other regard, so it is probably enough to do one "sequence" of notebooks.
These notebooks are useful if you want to be able to see exactly what is going on and have an analytical reference solution to compare the committor models with.

## Lithium chloride dissociation

This folder contains notebooks showcasing how to learn the solvent-dependent committor for the dissociation of lithium chloride.
In these notebooks you will use symmetry functions to describe the solvent in a permutation and translation invariant way.

## Capped alanine dipeptide

This folder contains notebooks showcasing how to learn the committor of the transition between the C7_eq and α_R states of capped alanine dipeptide.
In these notebooks you will use an internal coordinate representation of the protein to describe the transition.

[openpathsampling]: http://openpathsampling.org/latest/
