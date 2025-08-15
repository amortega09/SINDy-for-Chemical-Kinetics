import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from scipy.stats import qmc

# --- True model definition (moderately complex system) ---
def true_model(t, y):
    A, B, C = y
    
    # Linear reactions
    k1 = 0.1  # A -> B
    k2 = 0.2  # B -> C
    k3 = 0.05 # A -> C
    
    # Nonlinear interactions
    k4 = 0.02 # A + B -> C (bimolecular)
    k5 = 0.01 # B -> A (reverse reaction)
    
    dA_dt = -k1 * A - k3 * A - k4 * A * B + k5 * B
    dB_dt = k1 * A - k2 * B - k4 * A * B - k5 * B
    dC_dt = k2 * B + k3 * A + k4 * A * B
    
    return [dA_dt, dB_dt, dC_dt]

# --- Simulation parameters ---
t_span = [0, 50]  # Shorter time span for better identifiability
t_eval = np.linspace(t_span[0], t_span[1], 200)  # More time points

# --- LHS Design of Experiments ---
n_experiments = 50  # Reduced number for better control
n_species = 3

# Define bounds for initial conditions
bounds = [
    [1.0, 3.0],   # A bounds - start with reasonable amounts
    [0.0, 0.5],   # B bounds - small initial amounts
    [0.0, 0.5]    # C bounds - small initial amounts
]

# Create Latin Hypercube Sampler
sampler = qmc.LatinHypercube(d=n_species, seed=42)
samples = sampler.random(n=n_experiments)

# Scale samples to bounds
initial_conditions_list = []
for i in range(n_experiments):
    scaled_sample = []
    for j in range(n_species):
        min_val, max_val = bounds[j]
        scaled_val = min_val + samples[i, j] * (max_val - min_val)
        scaled_sample.append(scaled_val)
    initial_conditions_list.append(scaled_sample)

# Species info
species_names = ['A', 'B', 'C']
colors = ['red', 'blue', 'green']

# Store all data
all_data = []
all_derivatives = []

print("🔬 Generating data with LHS Design of Experiments...")
print(f"📊 Number of experiments: {n_experiments}")
print(f"⏱️  Time points per experiment: {len(t_eval)}")

# --- Generate experiments ---
for i, initial_conditions in enumerate(initial_conditions_list):
    if i % 10 == 0:
        print(f"Processing experiment {i+1}/{n_experiments}")
    
    # Solve ODE
    solution = solve_ivp(
        true_model, t_span, initial_conditions,
        dense_output=True,
        method='RK45',
        rtol=1e-8,
        atol=1e-8
    )
    
    y_values = solution.sol(t_eval).T
    
    # Add moderate noise (5%)
    noise_level = 0.05
    signal_range = np.max(y_values, axis=0) - np.min(y_values, axis=0)
    sigma = signal_range * noise_level
    noise = np.random.normal(0, sigma, y_values.shape)
    noisy_signal = y_values + noise
    
    # Create DataFrame
    df = pd.DataFrame(noisy_signal, columns=species_names)
    df['time'] = t_eval
    df['experiment_id'] = i + 1
    all_data.append(df)
    
    # Calculate true derivatives
    true_derivs = np.array([true_model(t, y) for t, y in zip(t_eval, y_values)])
    
    # Estimate derivatives from noisy data using Savitzky-Golay
    estimated_derivs = {}
    for j, species in enumerate(species_names):
        estimated_derivs[f'd{species}_dt'] = savgol_filter(
            noisy_signal[:, j], 31, 2, deriv=1, delta=(t_eval[1] - t_eval[0])
        )
    
    # Create derivatives DataFrame
    df_deriv = pd.DataFrame(estimated_derivs)
    df_deriv['time'] = t_eval
    df_deriv['experiment_id'] = i + 1
    all_derivatives.append(df_deriv)
    
    # Plot first 3 experiments
    if i < 3:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Concentration plot
        for j, species in enumerate(species_names):
            ax1.plot(t_eval, y_values[:, j], color=colors[j], label=f'{species} (true)', linewidth=2)
            ax1.scatter(t_eval[::10], noisy_signal[::10, j], color=colors[j], alpha=0.6, s=20, label=f'{species} (noisy)')
        
        ax1.set_title(f'Experiment {i+1}: Concentration Trajectories')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Concentration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Derivative plot
        for j, species in enumerate(species_names):
            ax2.plot(t_eval, true_derivs[:, j], color=colors[j], label=f'd{species}/dt (true)', linewidth=2)
            ax2.plot(t_eval, estimated_derivs[f'd{species}_dt'], color=colors[j], linestyle='--', alpha=0.7, label=f'd{species}/dt (estimated)')
        
        ax2.set_title(f'Experiment {i+1}: Derivatives')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# --- Combine all data ---
final_data = pd.concat(all_data, ignore_index=True)
final_derivatives = pd.concat(all_derivatives, ignore_index=True)

# Save data
final_data.to_csv('experimental_data.csv', index=False)
final_derivatives.to_csv('estimated_derivatives.csv', index=False)

# Save initial conditions
initial_conditions_df = pd.DataFrame(initial_conditions_list, columns=species_names)
initial_conditions_df['experiment_id'] = range(1, n_experiments + 1)
initial_conditions_df.to_csv('initial_conditions.csv', index=False)

print("\n✅ Data generation completed!")
print(f"📈 Total data points: {len(final_data)}")
print(f"🔬 Experiments: {n_experiments}")
print(f"📊 Data shape: {final_data.shape}")

# --- Print true model for reference ---
print("\n🔍 True Model Equations:")
print("=" * 40)
print("dA/dt = -0.1*A - 0.05*A - 0.02*A*B + 0.01*B = -0.15*A - 0.02*A*B + 0.01*B")
print("dB/dt = 0.1*A - 0.2*B - 0.02*A*B - 0.01*B = 0.1*A - 0.21*B - 0.02*A*B")
print("dC/dt = 0.2*B + 0.05*A + 0.02*A*B")

# --- Data quality check ---
print("\n📊 Data Quality Summary:")
print("=" * 40)
for species in species_names:
    data_range = final_data[species].max() - final_data[species].min()
    print(f"{species}: Range = {data_range:.3f}, Mean = {final_data[species].mean():.3f}")

print(f"\n📁 Files saved:")
print("- experimental_data.csv")
print("- estimated_derivatives.csv") 
print("- initial_conditions.csv")
