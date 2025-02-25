# LGLParticleTracking
A demonstration of particle tracking model for the Laurentian Great Lakes. This code is associated with work by Daniel E. Sandborn (School of Oceanography, University of Washington) and Jay A. Austin (Large Lakes Observatory, University of Minesota Duluth), and provides a public archive supporting manuscripts in preparation. 

# To Use

Clone this repository to a local directory and prepare a Python environment as described in the OceanTracker [documentation](https://oceantracker.github.io/oceantracker/_build/html/info/installing.html). Ensure all Python packages imported in the code are installed in the environment before executing the Python file corresponding to the simulation you wish to run. Hydrodynamic model output must be downloaded (using utils/nc_compressor.py or another tool) before the demo experiment or the Jupyter notebook can run. 

Please reach out to the authors with questions or suggestions!

# Associated work

Sandborn, D.E.; Austin, J.A.; and Lafrancois, B.M. "Developing a Lake Superior particle tracking model for research and management applications". NPS Reports. _In review._

## OceanTracker

[OceanTracker](https://github.com/oceantracker/oceantracker) is a Python package being developed by R. Vennell and collaborators. It uses hydrodynamic model output to translate physical forcing to particle tracks. OceanTracker is a required dependency of the model described here. It is described in further detail in the following publication:

Vennell, R., Scheel, M., Weppe, S., Knight, B., & Smeaton, M. (2021). Fast lagrangian particle tracking in unstructured ocean model grids. Ocean Dynamics, 71(4), 423–437. https://doi.org/10.1007/s10236-020-01436-7

## Lake Superior Observational Forecast System

Provides hydrodynamic model output which drives particle transport in this system. More information [here](https://tidesandcurrents.noaa.gov/ofs/lsofs/lsofs.html).

### Cite

Please cite:
Sandborn, D.E.; Austin J.A. (2025). "LGLParticleTracking" https://github.com/d-sandborn/LGLParticleTracking

## Disclaimer

The material embodied in this software is provided to you "as-is" and without warranty of any kind, express, implied or otherwise, including without limitation, any warranty of fitness for a particular purpose.In no event shall the authors be liable to you or anyone else for any direct, special, incidental, indirect or consequential damages of any kind, or any damages whatsoever, including without limitation, loss of profit, loss of use, savings or revenue, or the claims of third parties, whether or not the authors have been advised of the possibility of such loss, however caused and on any theory of liability, arising out of or in connection with the possession, use or performance of this software.
