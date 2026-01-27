## This is the [hast](https://bitbucket.org/vperret/hast) bitbucket repository.

Hast is an open source code to select halo of interest in Ramses unigrid cosmological simulations. Hast uses pynbody to read the simulation data, finds the halos thanks to Ramses clump finder, and computes the volume and density in the initial conditions of the convex hull defining the Lagrangian region.


Download the code by cloning the git repository using 
```
$ git clone https://bitbucket.org/vperret/hast
```

## Quick start (selection)
1) Edit the example config:
   - `examples/selection.conf`
2) Run the selection script:
```
python examples/selection.py examples/selection.conf
```

This reads RAMSES outputs and clump finder files to select halos and trace
their Lagrangian regions. See `examples/selection.conf` for required fields.
