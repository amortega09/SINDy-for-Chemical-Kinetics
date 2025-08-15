# SINDy-for-Chemical-Kinetics

## Code Description

### **`data_generation.py`**
Simulates a realistic experimental scenario by:

- **Ground truth system**: 3-species nonlinear chemical reaction for a batch isothermal reactor:

-dA/dt = -0.15*A - 0.02*A*B + 0.01*B 

-dB/dt = 0.10*A - 0.21*B - 0.02*A*B 

-dC/dt = 0.20*B + 0.05*A + 0.02*A*B

- **Experimental design**: 50 initial conditions via **Latin Hypercube Sampling**
- **Realism enhancements**:
  - Add 5% Gaussian measurement noise to concentration data
  - Estimate derivatives using **Savitzky–Golay filtering**
- **Output**: Datasets that mimic real experimental conditions

---

### **`doe_sindy.py`**
Implements the **enhanced DoE-SINDy algorithm**, featuring:

- Experimental-level subsampling to reduce bias vs. random point sampling
- Cross-validation for robust model selection across experimental conditions
- Iterative parameter refinement using validation data
- Statistical significance testing to remove non-contributing terms
- Identifiability analysis to prevent overly complex models
- Convergence monitoring for automated stopping criteria

---

## 📖 Inspiration
Inspired by the paper:  
> Lyu, W., & Galvanin, F. (2022). DoE-SINDy: An automated framework for model generation and selection in kinetic studies. *Computers & Chemical Engineering*, 164, 107777. [https://doi.org/10.1016/j.compchemeng.2022.107777](https://doi.org/10.1016/j.compchemeng.2022.107777)
