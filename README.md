# observingtng
###### A few scripts to help you (or, maybe just me, who knows) explore TNG
---

## IllustrisTNG
The IllustrisTNG project is a suite of cosmological magnetohydrodynamic simulations of galaxy formation and evolution. For simulation details, science use cases, and data access, please visit the [team page](https://www.tng-project.org/).

## A few analysis codes (python)

### Particle tracking in TNG
To track particles in a given galaxy (or list of galaxies) across the simulation from z=20 to z=0, you can run `build_stellar_particle_catalog.py`. The key outputs from this script are the snapshots (redshifts) at which each particle (a) formed; (b) crossed the virial radius of its z=0 host for the first time; and (c) became bound to its z=0 host. 

Example usage:

<code>$ python build_stellar_particle_catalog.py -v -w -t '/path/to/TNG/outputs/' subfindID </code>

### Image generation

To generate idealized light or stellar mass surface density images, run the code `adaptivebox.py`. 

Example usage:

<code>$ python adaptivebox.py -v -m -g -r --subhaloID subfindID </code>


To generate mock observed images from the idealized frames, you can run the code `butterfly.py` (or similar). Please note that `butterfly.py` was not written in a way that was intended to be generalized for other datasets or projects. So -- proceed with caution.

### Acknowledgements
If you use these scripts in your work, please cite [Merritt+ (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4570M/abstract). Thank you!
