# ``aimmd.distributed`` example notebooks

```{toctree}
:maxdepth: 1
:caption: aimmd.distributed notebooks

TPS 1: Setup and run simulation <TPS_1_setup_and_run_simulation>
TPS 2: Continue simulation <TPS_2_continue_simulation>
TPS 3: Analyze simulation <TPS_3_analyze_simulation>
TPS 4: Rerun with changed parameters or recover crashed simulations <TPS_4_rerun_with_changed_parameters_or_recover_crashed_simulations>

TPS with EQ SPs 1: Generate SPs from UmbrellaSampling <TPS_with_EQ_SPs_1_generate_SPs_from_UmbrellaSampling>
TPS with EQ SPs 2: Setup and run simulation <TPS_with_EQ_SPs_2_setup_and_run_simulation>
TPS with EQ SPs 3: Continue and analyze simulation <TPS_with_EQ_SPs_3_continue_and_analyze_simulation>
TPS with EQ SPs 4: Rerun with changed parameters or recover crashed simulations <TPS_with_EQ_SPs_4_rerun_with_changed_parameters_or_recover_crashed_simulations>

Committor simulation <CommittorSimulation>

Advanced topics: Customizing your TPS simulations using BrainTasks <Advanced_topics/Customizing_your_TPS_simulations_using_BrainTasks>
```

Using ``aimmd.distributed`` all TPS simulations you perform are steered by one common committor model and you can perform an arbitrary number of TPS simulations simultaneously (also on a cluster via using asyncmd).
An important choice for your TPS simulation are where you will get your shooting points from and while ``aimmd.distributed`` supports many different choices, here you will find example notebooks for two common cases: using configurations from previously accepted transitions (the notebooks with the ``TPS_`` prefix) or using configurations with a know equilibrium weight (the notebooks with the ``TPS_with_EQ_SPs_`` prefix).
Both of these notebook sequences guid you through setting up, performing, and finally analyzing the simulation.

You will also find a notebook on how to perform committor simulations in parallel.
These can be useful not only to validate a committor model prediction, but can also be (mis)-used to generate initial transition paths to seed a TPS simulation.

Finally, the subfolder ``Advanced_topics`` contains notebooks on various more advanced topics, such as writing your own python code to modify the behavior of your TPS simulations.
