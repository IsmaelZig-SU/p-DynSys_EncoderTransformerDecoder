# p-DynSys_EncoderTransformerDecoder

(1) - git clone https://github.com/IsmaelZig-SU/p-DynSys_EncoderTransformerDecoder.git

(2) - Run `cd p-DynSys_EncoderTransformerDecoder` to change the directory.


(3.0) - `python -m venv env`

(3.1) - `source env/bin/activate`    # Linux/macOS

(3.1) - `env\Scripts\activate`       # Windows

(3.2) - `pip install` -r requirements.txt

(4) - Download Data from [here](https://drive.google.com/file/d/1iZpAsPFqziRx3hTSnsfMqt9czlftcq3i/view?usp=sharing), unzip in the root folder.

(5) - Download Trained_Models from : [here](https://drive.google.com/file/d/17jrquMr-GZaQ3ohqxi2UOEweDfijiH7A/view?usp=sharing), unzip it in the root folder 

(6) - You can run the Notebook in Notebooks/2DCyl/Hands_on_UP-d-ROM.ipynb

<pre>
Your folder structure should be : 

|   .gitignore
|   main.py
|   README.md
|   requirements.txt
|
+---Data
|   \---2DCylinder
|       \---processed_data
|           \---npyfiles
|                   info.txt
|                   test.npy
|                   train.npy
|
+---Notebooks
|   \---2DCyl
|       |   Hands_on_UP-d-ROM.ipynb
|       |
|       \---.ipynb_checkpoints
|               Hands_on_UP-d-ROM-checkpoint.ipynb
|
+---src
|   |   Eval.py
|   |   Experiment.py
|   |
|   +---Layers
|   |   |   Network.py
|   |   |   transformer_cross_att.py
|   |   |   VAE.py
|   |   |
|   |   \---__pycache__
|   |           MZANetwork.cpython-311.pyc
|   |           Network.cpython-311.pyc
|   |           transformer_cross_att.cpython-311.pyc
|   |           VAE.cpython-311.pyc
|   |
|   +---PreProc_Data
|   |   |   DataProc.py
|   |   |   DynSystem_Data.py
|   |   |
|   |   \---__pycache__
|   |           DataProc.cpython-311.pyc
|   |           DynSystem_Data.cpython-311.pyc
|   |
|   +---Train_Methods
|   |   |   Train_Methodology.py
|   |   |
|   |   \---__pycache__
|   |           Train_Methodology.cpython-311.pyc
|   |
|   +---utils
|   |   |   make_dir.py
|   |   |
|   |   \---__pycache__
|   |           make_dir.cpython-311.pyc
|   |
|   \---__pycache__
|           Eval.cpython-311.pyc
|           Experiment.cpython-311.pyc
|           MZA_Experiment.cpython-311.pyc
|
\---Trained_models
    \---2DCyl_new
        |   info.txt
        |
        \---sl9_obs4_bs64_attblks1_atthds8_tr0_ph10_lbdaStateLoss1.0_nhd64_0.0002__
            |   args
            |
            +---model_weights
            |       at_epoch0
            |       at_epoch1000
            |       at_epoch1500
            |       at_epoch2000
            |       at_epoch2500
            |       at_epoch2999
            |       at_epoch500
            |       min_test_loss
            |       min_train_loss
            |
            \---out_log
                    AutoencoderLoss.png
                    log
                    StateLoss.png
                    TotalLoss.png
                    TransEvo.png


    </pre>

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
