# LGLParticleTracking
A demonstration of particle tracking model for the Laurentian Great Lakes. This code is associated with work by Daniel E. Sandborn and Jay A. Austin (Large Lakes Observatory, University of Minesota Duluth), and provides a public archive supporting manuscripts in preparation. 

# To Use

Clone this repository to a local directory and prepare a Python environment as described in the OceanTracker [documentation](https://oceantracker.github.io/oceantracker/_build/html/info/installing.html). Ensure all Python packages imported in the code are installed in the environment before executing the Python file corresponding to the simulation you wish to run. 

Hydrodynamic model output must be downloaded (using utils/nc_compressor.py or another tool) before the demo experiment or the Jupyter notebook can run. 

# Associated work

## OceanTracker

[OceanTracker](https://github.com/oceantracker/oceantracker) is a Python package being developed by R. Vennell and collaborators. It uses hydrodynamic model output to translate physical forcing to particle tracks. OceanTracker is a required dependency of the model described here. It is described in further detail in the following publication:

Vennell, R., Scheel, M., Weppe, S., Knight, B., & Smeaton, M. (2021). Fast lagrangian particle tracking in unstructured ocean model grids. Ocean Dynamics, 71(4), 423–437. https://doi.org/10.1007/s10236-020-01436-7

## Lake Superior Observational Forecast System

Provides hydrodynamic model output which drives particle transport in this system. More information [here](https://tidesandcurrents.noaa.gov/ofs/lsofs/lsofs.html).

### Cite

Until the release of a publication, please cite:
Sandborn, D.E.; Austin J.A. "LGLParticleTracking" (2024) *Github repository*
