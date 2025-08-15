# SINDy-for-Chemical-Kinetics
**Code Description**
data_generation.py
Simulates a realistic experimental scenario by: Defining a known 3-species nonlinear chemical reaction system as ground truth for a batch isothermal reactor with the following governing equations:
dA/dt = -0.15*A - 0.02*A*B + 0.01*B
dB/dt = 0.10*A - 0.21*B - 0.02*A*B  
dC/dt = 0.20*B + 0.05*A + 0.02*A*B
-Using Latin Hypercube Sampling to systematically design 50 experiments with varied initial conditions
-Adding realistic measurement noise (5%) to concentration data
-Estimating derivatives from noisy measurements using Savitzky-Golay filtering
-Generating comprehensive datasets that mimic real experimental conditions

doe_sindy.py
Implements the enhanced DoE-SINDy algorithm featuring: 
-Experimental-level subsampling to reduce bias compared to random point sampling
-Cross-validation framework for robust model selection across experimental conditions
-Iterative parameter refinement using validation data
-Statistical significance testing to remove non-contributing terms
-Identifiability analysis to prevent overly complex models
-Convergence monitoring for automated stopping criteria

Inspiration
Inspired by the paper: DoE-SINDy: an automated framework for model generation and selection in kinetic studies by Wenyao Lyu and Federico Galvanin.
 

