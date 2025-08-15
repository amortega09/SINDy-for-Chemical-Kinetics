import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ, SR3, SSR
import warnings
warnings.filterwarnings('ignore')

class OptimizedDoESINDy:
    def __init__(self, species_names, max_iterations=10, cv_folds=5):
        self.species_names = species_names
        self.max_iterations = max_iterations
        self.cv_folds = cv_folds
        self.best_model = None
        self.best_score = float('inf')
        self.convergence_history = []
        self.best_config = None
        
    def experimental_level_subsampling(self, X_list, X_dot_list, subsample_ratio=0.8):
        """Reduce bias by subsampling experiments rather than individual data points"""
        n_experiments = len(X_list)
        n_subsample = int(n_experiments * subsample_ratio)
        
        # Randomly select experiments
        selected_indices = np.random.choice(n_experiments, n_subsample, replace=False)
        
        X_subsampled = [X_list[i] for i in selected_indices]
        X_dot_subsampled = [X_dot_list[i] for i in selected_indices]
        
        return X_subsampled, X_dot_subsampled
    
    def robust_derivative_estimation(self, X, t_eval, window_size=15):
        """More robust derivative estimation for noisy data"""
        from scipy.signal import savgol_filter
        
        derivatives = []
        for i in range(X.shape[1]):
            # Use smaller window for noisy data
            window = min(window_size, len(X) // 3)
            if window % 2 == 0:
                window += 1
            
            # Ensure window is at least 3
            window = max(3, window)
            
            try:
                deriv = savgol_filter(X[:, i], window, 2, deriv=1, delta=(t_eval[1] - t_eval[0]))
                derivatives.append(deriv)
            except:
                # Fallback to simple finite difference
                deriv = np.gradient(X[:, i], t_eval)
                derivatives.append(deriv)
        
        return np.column_stack(derivatives)
    
    def get_optimized_configurations(self):
        """Get optimized model configurations for noisy data"""
        configs = [
            # Conservative configurations for noisy data
            {
                'name': 'Conservative STLSQ',
                'feature_library': PolynomialLibrary(degree=2, include_bias=False),
                'optimizer': STLSQ(threshold=0.01, alpha=0.1, max_iter=100)
            },
            {
                'name': 'Moderate STLSQ',
                'feature_library': PolynomialLibrary(degree=2, include_bias=False),
                'optimizer': STLSQ(threshold=0.005, alpha=0.05, max_iter=100)
            },
            {
                'name': 'Aggressive STLSQ',
                'feature_library': PolynomialLibrary(degree=2, include_bias=False),
                'optimizer': STLSQ(threshold=0.001, alpha=0.01, max_iter=100)
            },
            {
                'name': 'SR3 with L1',
                'feature_library': PolynomialLibrary(degree=2, include_bias=False),
                'optimizer': SR3(threshold=0.01, max_iter=100)
            },
            {
                'name': 'SR3 Conservative',
                'feature_library': PolynomialLibrary(degree=2, include_bias=False),
                'optimizer': SR3(threshold=0.02, max_iter=100)
            },
            {
                'name': 'SSR Optimizer',
                'feature_library': PolynomialLibrary(degree=2, include_bias=False),
                'optimizer': SSR(alpha=0.1)
            }
        ]
        return configs
    
    def robust_model_validation(self, model, X_val, X_dot_val):
        """More robust model validation for noisy data"""
        try:
            # Check if model has valid coefficients
            coef = model.coefficients()
            if coef is None or np.all(coef == 0):
                return False, "Zero coefficients"
            
            # Check for reasonable coefficient magnitudes
            if np.any(np.abs(coef) > 100):
                return False, "Unreasonable coefficients"
            
            # Try prediction
            X_dot_pred = model.predict(X_val)
            
            # Check for NaN or inf predictions
            if np.any(np.isnan(X_dot_pred)) or np.any(np.isinf(X_dot_pred)):
                return False, "Invalid predictions"
            
            # Calculate R² score
            r2_scores = []
            for i in range(X_dot_val.shape[1]):
                try:
                    r2 = r2_score(X_dot_val[:, i], X_dot_pred[:, i])
                    r2_scores.append(r2)
                except:
                    r2_scores.append(-np.inf)
            
            avg_r2 = np.mean(r2_scores)
            
            # More lenient thresholds for noisy data
            if avg_r2 < 0.1:  # Reduced from 0.3
                return False, f"Low R²: {avg_r2:.3f}"
            
            # Check model complexity
            n_nonzero = np.sum(np.abs(coef) > 1e-6)
            n_equations = len(self.species_names)
            
            if n_nonzero > n_equations * 8:  # Increased from 5
                return False, "Too complex"
            
            return True, f"Valid (R²: {avg_r2:.3f})"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def fit(self, X_list, X_dot_list):
        """Optimized fitting procedure for noisy data"""
        print("Starting Optimized DoE-SINDy fitting for noisy data...")
        print("=" * 70)
        
        configs = self.get_optimized_configurations()
        
        for iteration in range(self.max_iterations):
            print(f"\nIteration {iteration + 1}/{self.max_iterations}")
            
            # Experimental-level subsampling
            X_subsampled, X_dot_subsampled = self.experimental_level_subsampling(X_list, X_dot_list)
            print(f"Subsampled to {len(X_subsampled)} experiments")
            
            # Cross-validation
            X_all = np.vstack(X_subsampled)
            X_dot_all = np.vstack(X_dot_subsampled)
            
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=iteration)
            cv_scores = []
            valid_models = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
                X_train, X_val = X_all[train_idx], X_all[val_idx]
                X_dot_train, X_dot_val = X_dot_all[train_idx], X_dot_all[val_idx]
                
                best_fold_model = None
                best_fold_score = float('inf')
                best_fold_config = None
                
                for config in configs:
                    try:
                        model = SINDy(
                            feature_library=config['feature_library'],
                            optimizer=config['optimizer'],
                            feature_names=self.species_names
                        )
                        
                        model.fit(X_train, t=None, x_dot=X_dot_train)
                        
                        # Robust validation
                        is_valid, message = self.robust_model_validation(model, X_val, X_dot_val)
                        
                        if is_valid:
                            X_dot_pred = model.predict(X_val)
                            score = mean_squared_error(X_dot_val, X_dot_pred)
                            
                            if score < best_fold_score:
                                best_fold_score = score
                                best_fold_model = model
                                best_fold_config = config
                                
                    except Exception as e:
                        continue
                
                if best_fold_model is not None:
                    cv_scores.append(best_fold_score)
                    valid_models.append((best_fold_model, best_fold_config))
                    print(f"  Fold {fold+1}: {best_fold_config['name']} - MSE: {best_fold_score:.6f}")
                else:
                    print(f"  Fold {fold+1}: No valid model found")
            
            if cv_scores:
                avg_score = np.mean(cv_scores)
                self.convergence_history.append(avg_score)
                
                print(f"Average CV MSE: {avg_score:.8f}")
                
                # Update best model
                if avg_score < self.best_score:
                    self.best_score = avg_score
                    # Use the best model from the best fold
                    best_idx = np.argmin(cv_scores)
                    self.best_model = valid_models[best_idx][0]
                    self.best_config = valid_models[best_idx][1]
                    
                    print(f"New best model: {self.best_config['name']} - Score: {avg_score:.8f}")
            else:
                print("No valid models found in this iteration")
                # Create a simple fallback model
                if self.best_model is None:
                    try:
                        self.best_model = SINDy(
                            feature_library=PolynomialLibrary(degree=2, include_bias=False),
                            optimizer=STLSQ(threshold=0.01, alpha=0.1, max_iter=100),
                            feature_names=self.species_names
                        )
                        self.best_model.fit(X_all, t=None, x_dot=X_dot_all)
                        self.best_config = {'name': 'Fallback STLSQ'}
                        print("Created fallback model")
                    except:
                        print("Failed to create fallback model")
            
            # Check convergence
            if len(self.convergence_history) > 2:
                if abs(self.convergence_history[-1] - self.convergence_history[-2]) < 1e-8:
                    print("Convergence reached!")
                    break
        
        return self.best_model

# Load data
print("Loading experimental data...")
data_df = pd.read_csv('experimental_data.csv')
derivatives_df = pd.read_csv('estimated_derivatives.csv')

print(f"Data shape: {data_df.shape}")
print(f"Derivatives shape: {derivatives_df.shape}")

# Data preprocessing
species_names = ['A', 'B', 'C']
derivative_names = [f'd{species}_dt' for species in species_names]

# Prepare data for SINDy
X_list = []
X_dot_list = []

for exp_id in data_df['experiment_id'].unique():
    exp_data = data_df[data_df['experiment_id'] == exp_id]
    exp_deriv = derivatives_df[derivatives_df['experiment_id'] == exp_id]
    
    X = exp_data[species_names].values
    X_dot = exp_deriv[derivative_names].values
    
    # Remove NaN values and outliers
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(X_dot).any(axis=1))
    
    # Remove extreme outliers (beyond 3 standard deviations)
    for i in range(X.shape[1]):
        mean_val = np.mean(X[valid_mask, i])
        std_val = np.std(X[valid_mask, i])
        outlier_mask = np.abs(X[:, i] - mean_val) <= 3 * std_val
        valid_mask = valid_mask & outlier_mask
    
    X = X[valid_mask]
    X_dot = X_dot[valid_mask]
    
    if len(X) > 10:  # Ensure sufficient data points
        X_list.append(X)
        X_dot_list.append(X_dot)

print(f"Valid experiments: {len(X_list)}")
print(f"Total data points: {sum(len(x) for x in X_list)}")

# Fit model with optimized DoE-SINDy
doe_sindy = OptimizedDoESINDy(species_names, max_iterations=8, cv_folds=3)
best_model = doe_sindy.fit(X_list, X_dot_list)

# Results analysis
if best_model is not None:
    print("\nFinal SINDy Model:")
    print("=" * 50)
    for i, species in enumerate(species_names):
        print(f"d{species}/dt = {best_model.equations()[i]}")

    # Compare with true model
    print("\nComparison with True Model:")
    print("=" * 50)
    true_equations = [
        "dA/dt = -0.15 A - 0.02 A B + 0.01 B",
        "dB/dt = 0.1 A - 0.21 B - 0.02 A B",
        "dC/dt = 0.2 B + 0.05 A + 0.02 A B"
    ]

    for i, (sindy_eq, true_eq) in enumerate(zip(best_model.equations(), true_equations)):
        print(f"\n{species_names[i]}:")
        print(f"  SINDy: {sindy_eq}")
        print(f"  True:  {true_eq}")

    # Model coefficients
    print("\nModel Coefficients:")
    print("=" * 50)
    feature_names = best_model.feature_library.get_feature_names(species_names)
    coef_df = pd.DataFrame(
        best_model.coefficients(),
        index=species_names,
        columns=feature_names
    )
    print(coef_df)

    # Save results
    coef_df.to_csv('sindy_coefficients.csv', index=True)
    print("\nModel coefficients saved to 'sindy_coefficients.csv'")

    # Prediction plots
    print("\nGenerating prediction plots...")
    X_all = np.vstack(X_list)
    X_dot_all = np.vstack(X_dot_list)
    X_dot_pred = best_model.predict(X_all)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, species in enumerate(species_names):
        ax = axes[i]
        ax.scatter(X_dot_all[:, i], X_dot_pred[:, i], alpha=0.6, s=20)
        ax.plot([X_dot_all[:, i].min(), X_dot_all[:, i].max()], 
                [X_dot_all[:, i].min(), X_dot_all[:, i].max()], 'r--', linewidth=2)
        ax.set_xlabel(f'True d{species}/dt')
        ax.set_ylabel(f'Predicted d{species}/dt')
        ax.set_title(f'{species} Equation')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sindy_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Convergence plot
    if len(doe_sindy.convergence_history) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(doe_sindy.convergence_history, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Iteration')
        plt.ylabel('CV MSE')
        plt.title('Optimized DoE-SINDy Convergence History')
        plt.grid(True, alpha=0.3)
        plt.savefig('convergence_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Performance metrics
    r2_scores = []
    for i in range(len(species_names)):
        r2 = r2_score(X_dot_all[:, i], X_dot_pred[:, i])
        r2_scores.append(r2)

    print("\nModel Performance:")
    print("=" * 50)
    for i, species in enumerate(species_names):
        print(f"{species}: R² = {r2_scores[i]:.4f}")

    overall_r2 = np.mean(r2_scores)
    print(f"\nOverall R²: {overall_r2:.4f}")

    # Save final results
    results = {
        'overall_r2': overall_r2,
        'best_cv_score': doe_sindy.best_score,
        'n_experiments': len(X_list),
        'n_data_points': len(X_all),
        'convergence_iterations': len(doe_sindy.convergence_history),
        'best_config': doe_sindy.best_config['name'] if doe_sindy.best_config else 'None'
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv('sindy_results.csv', index=False)
    print("\nResults saved to 'sindy_results.csv'")

    print("\nOptimized DoE-SINDy fitting completed!")
    print(f"Final model trained on {len(X_all)} data points from {len(X_list)} experiments")
    print(f"Overall R²: {overall_r2:.4f}")
    print(f"Best configuration: {doe_sindy.best_config['name'] if doe_sindy.best_config else 'None'}")

else:
    print("\nFailed to fit a valid model. Consider:")
    print("   - Reducing noise in data generation")
    print("   - Increasing number of experiments")
    print("   - Adjusting model complexity")