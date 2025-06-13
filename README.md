(1) - git clone https://github.com/IsmaelZig-SU/p-DynSys_EncoderTransformerDecoder.git

(2) - cd p-DynSys_EncoderTransformerDecoder

(3.0) - python -m venv env

(3.1) - source env/bin/activate    # On Linux/macOS

(3.1) - env\Scripts\activate       # On Windows

(3.2) - pip install -r requirements.txt  #Takes about 10 minutes


# p-DynSys_EncoderTransformerDecoder

Parametrised Uncertainty-Aware ROM for Dynamical Systems : https://arxiv.org/abs/2503.23236
This repository implements a variational and parametrised equivalent of the DynSys_EncodeTransformerDecoder model. It is designed to handle parametrised and uncertainty-aware dynamic reduced-order models (ROMs) for dynamical systems, with a focus on unsteady flows. For a detailed theoretical background, please refer to the article:
"Parametrised and Uncertainty-Aware Dynamic Reduced-Order Model – Application to Unsteady Flows."

Expected Data Format
The model expects input data with the following dimensions:
[p, t, d + p.dim]

Dimensions Explained:
  -p: Number of distinct parameter sets (parameter dimension). Parameters refer to external variables (e.g., Reynolds number) that can influence the system's response. The dimension of the parameter space is referred to as p.dim.
  
  -t: Number of snapshots (time dimension).
  
  -d: Spatial dimension (for 1D systems, this corresponds to the number of spatial points).

Example: Navier-Stokes Emulator

Consider a Navier-Stokes emulator for a flow domain defined on a 100 × 100 grid, with 1500 snapshots. The flow is simulated under 5 different configurations, where a single parameter (e.g., Reynolds number) is varied.

Dataset Dimensions:
The dataset should have the shape: [5, 1500, 10001].

Explanation:
p = 5: There are 5 distinct parameter sets (e.g., 5 different Reynolds numbers).

t = 1500: Each configuration has 1500 snapshots in time.

d + p.dim = 10001:

d = 10000: The spatial dimension corresponds to the 100 × 100 grid (flattened to 10,000 points).

p.dim = 1: The parameter value (e.g., Reynolds number) is appended to the spatial vector, resulting in a total of 10,001 points. You are free to stack more than 1 parameter (Geometry, Viscosity, Reynolds...)

Parameter Stacking:
The parameter value (e.g., Reynolds number) is stacked at the end of the spatial vector for each snapshot. This parameter value is unique for each of the 5 configurations.

Key Notes:
  -The model is designed to handle parametrised dynamical systems and incorporates uncertainty quantification.
  -Ensure that the input data is properly formatted, with parameters correctly appended to the spatial vectors. Make sure that the data is normalised parameter wise to ensure equal importance is given by the model to each parameter set. 
  -For further details, refer to the associated article or reach out to the repository maintainers.
