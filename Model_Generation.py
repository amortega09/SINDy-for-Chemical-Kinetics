from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# --- True model definition ---
def True_Model(t, y, E1, E2, E3, E4, cp, p, dH1, dH2, dH3, dH4):
    CA, CB, CC, CD, CE, T = y
    
    k10 = 0.1
    k20 = 0.2
    k30 = 0.3
    k40 = 0.3
    
    k1 = k10 * np.exp(-E1 / (8.314 * T))
    k2 = k20 * np.exp(-E2 / (8.314 * T))
    k3 = k30 * np.exp(-E3 / (8.314 * T))
    k4 = k40 * np.exp(-E4 / (8.314 * T))
    
    dCA_dt = -k1 * CA - k2 * CA**2 - k3 * CA * CB
    dCB_dt = k1 * CA - k4 * CB - k3 * CA * CB
    dCC_dt = k4 * CB
    dCD_dt = k2 * CA**2
    dCE_dt = k3 * CA * CB
    dT_dt = -(dH1 * k1 * CA + dH2 * k2 * CA**2 + dH3 * k3 * CA * CB + dH4 * k4 * CB) / (cp * p)
    
    return [dCA_dt, dCB_dt, dCC_dt, dCD_dt, dCE_dt, dT_dt]

# --- Parameters ---
E1, E2, E3, E4 = 10000, 11000, 6000, 7000
cp = 10
p = 5
dH1, dH2, dH3, dH4 = -1000, -2000, -3000, -4000
params = (E1, E2, E3, E4, cp, p, dH1, dH2, dH3, dH4)

# --- Time span for the simulation ---
t_span = [0, 300]
t_eval = np.linspace(t_span[0], t_span[1], 500)

# --- Define a list of initial conditions to test ---
initial_conditions_list = [
    [1.0, 0.0, 0.0, 0.0, 0.3, 300.0],
    [1.5, 0.0, 0.3, 0.0, 0.0, 310.0],
    [0.8, 0.2, 0.0, 0.25, 0.0, 295.0],
    [1.2, 0.1, 0.0, 0.0, 0.0, 305.0],
    [2, 0.2, 0.2, 0.2, 0.1, 305.0]
]

# --- Initialize empty list to store data from all runs ---
all_noisy_data = []

# --- Set up the plotting ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Chemical Reactor Experiments: Concentration and Temperature Trajectories', fontsize=12)

# Flatten axes for easier indexing
axes_flat = axes.flatten()

# Colors for different species
colors = ['red', 'blue', 'green', 'orange', 'purple']
species_names = ['CA', 'CB', 'CC', 'CD', 'CE']

# --- Main loop to generate data for each initial condition ---
for i, initial_conditions in enumerate(initial_conditions_list):
    solution = solve_ivp(
        True_Model, t_span, initial_conditions,
        args=params,
        dense_output=True,
    )
    
    y_values = solution.sol(t_eval).T
    df_true = pd.DataFrame(y_values, columns=['CA', 'CB', 'CC', 'CD', 'CE', 'T'])
    
    # --- Add noise only to concentrations ---
    noise_level = 0.1
    signal_range = np.max(y_values[:, :5], axis=0) - np.min(y_values[:, :5], axis=0)
    sigma = signal_range * noise_level
    noise = np.random.normal(0, sigma, (len(t_eval), 5))
    
    noisy_signal = np.copy(y_values)
    noisy_signal[:, :5] += noise
    
    # --- Build noisy DataFrame (T remains clean) ---
    df_noisy = pd.DataFrame(noisy_signal, columns=['CA', 'CB', 'CC', 'CD', 'CE', 'T'])
    df_noisy['time'] = t_eval               # ✅ Add time column
    df_noisy['run_id'] = i + 1              # Run ID for grouping
    
    all_noisy_data.append(df_noisy)
    
    # --- Plotting for this experiment ---
    # Plot concentrations
    ax_conc = axes_flat[i]
    for j, species in enumerate(species_names):
        # Plot true values (solid lines)
        ax_conc.plot(t_eval, y_values[:, j], color=colors[j], linewidth=1.5, 
                    label=f'{species}', alpha=0.8)
        # Plot noisy measurements (scatter points)
        ax_conc.scatter(t_eval[::25], noisy_signal[::25, j], color=colors[j], 
                       s=8, alpha=0.5, marker='o')
    
    ax_conc.set_xlabel('Time (s)', fontsize=9)
    ax_conc.set_ylabel('Concentration (mol/L)', fontsize=9)
    ax_conc.set_title(f'Experiment {i+1}: Concentrations', fontsize=10)
    ax_conc.legend(fontsize=8, loc='upper right')
    ax_conc.grid(True, alpha=0.3)
    ax_conc.tick_params(labelsize=8)

# --- Temperature plots (use the last subplot) ---
ax_temp = axes_flat[5]
for i, initial_conditions in enumerate(initial_conditions_list):
    # Re-solve for temperature plotting (could be optimized by storing results)
    solution = solve_ivp(
        True_Model, t_span, initial_conditions,
        args=params,
        dense_output=True,
    )
    y_values = solution.sol(t_eval).T
    
    ax_temp.plot(t_eval, y_values[:, 5], linewidth=1.5, 
                label=f'Exp {i+1}', alpha=0.8)

ax_temp.set_xlabel('Time (s)', fontsize=9)
ax_temp.set_ylabel('Temperature (K)', fontsize=9)
ax_temp.set_title('All Experiments: Temperature Trajectories', fontsize=10)
ax_temp.legend(fontsize=8)
ax_temp.grid(True, alpha=0.3)
ax_temp.tick_params(labelsize=8)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# --- Concatenate all DataFrames into a single one ---
final_measurements_df = pd.concat(all_noisy_data, ignore_index=True)

# --- Save to CSV ---
final_measurements_df.to_csv('all_measurements.csv', index=False)

print(f"Data saved to 'all_measurements.csv'")
print(f"Total data points generated: {len(final_measurements_df)}")
print(f"Number of experiments: {len(initial_conditions_list)}")
print(f"Time points per experiment: {len(t_eval)}")

# --- Derivative estimation settings ---
window_length = 100  # Must be odd and <= len(t_eval)
polyorder = 2

# --- Initialize list to store all derivative data ---
all_derivatives = []

# --- Loop again over experiments to compute derivatives ---
for i, initial_conditions in enumerate(initial_conditions_list):
    # Solve to get true values again
    solution = solve_ivp(
        True_Model, t_span, initial_conditions,
        args=params,
        dense_output=True,
    )
    
    y_true = solution.sol(t_eval).T
    df_true = pd.DataFrame(y_true, columns=['CA', 'CB', 'CC', 'CD', 'CE', 'T'])

    # Get the corresponding noisy data
    df_noisy = all_noisy_data[i].copy()
    
    # --- Compute noisy derivatives using Savitzky-Golay filter ---
    noisy_derivatives = {}
    for col in ['CA', 'CB', 'CC', 'CD', 'CE', 'T']:
        noisy_derivatives[f'd{col}_dt'] = savgol_filter(df_noisy[col].values, window_length, polyorder, deriv=1, delta=(t_eval[1] - t_eval[0]))

    # --- Compute true derivatives from the ODE function ---
    true_derivatives = []
    for j in range(len(t_eval)):
        yj = y_true[j]
        dy_dt = True_Model(t_eval[j], yj, *params)
        true_derivatives.append(dy_dt)
    true_derivatives = np.array(true_derivatives)

    # --- Plot comparisons ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Experiment {i+1}: Noisy Derivative Estimates vs. True Derivatives", fontsize=12)
    axes_flat = axes.flatten()

    for k, species in enumerate(['CA', 'CB', 'CC', 'CD', 'CE']):
        ax = axes_flat[k]
        ax.plot(t_eval, true_derivatives[:, k], label='True', color='black')
        ax.plot(t_eval, noisy_derivatives[f'd{species}_dt'], label='Estimated (SG)', linestyle='--', color=colors[k])
        ax.set_title(f'd{species}/dt')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Temperature derivative
    ax = axes_flat[5]
    ax.plot(t_eval, true_derivatives[:, 5], label='True', color='black')
    ax.plot(t_eval, noisy_derivatives['dT_dt'], label='Estimated (SG)', linestyle='--', color='gray')
    ax.set_title('dT/dt')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- Save noisy derivatives with time to DataFrame ---
    df_deriv = pd.DataFrame(noisy_derivatives)
    df_deriv['time'] = t_eval
    df_deriv['run_id'] = i + 1
    all_derivatives.append(df_deriv)

# --- Concatenate all derivative DataFrames ---
final_derivatives_df = pd.concat(all_derivatives, ignore_index=True)

# --- Save to CSV ---
final_derivatives_df.to_csv('noisy_derivatives.csv', index=False)

print(f"Derivative estimates saved to 'noisy_derivatives.csv'")
print(f"Shape of derivative data: {final_derivatives_df.shape}")
