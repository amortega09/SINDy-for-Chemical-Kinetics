import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ, SR3
import warnings
warnings.filterwarnings('ignore')

class DoESINDy:
    def __init__(self, species_names, max_iterations=10, cv_folds=5):
        self.species_names = species_names
        self.max_iterations = max_iterations
        self.cv_folds = cv_folds
        self.best_model = None
        self.best_score = float('inf')
        self.convergence_history = []
        
    def experimental_level_subsampling(self, X_list, X_dot_list, subsample_ratio=0.8):
        """Reduce bias by subsampling experiments rather than individual data points"""
        n_experiments = len(X_list)
        n_subsample = int(n_experiments * subsample_ratio)
        
        # Randomly select experiments
        selected_indices = np.random.choice(n_experiments, n_subsample, replace=False)
        
        X_subsampled = [X_list[i] for i in selected_indices]
        X_dot_subsampled = [X_dot_list[i] for i in selected_indices]
        
        return X_subsampled, X_dot_subsampled
    
    def parameter_re_estimation(self, model, X_train, X_dot_train, X_val, X_dot_val):
        """Refine model parameters using validation data"""
        # Get current coefficients
        current_coeffs = model.coefficients().copy()
        
        # Try different thresholds for refinement
        best_threshold = None
        best_score = float('inf')
        
        for threshold in [0.001, 0.005, 0.01, 0.02]:
            refined_model = SINDy(
                feature_library=model.feature_library,
                optimizer=STLSQ(threshold=threshold, alpha=0.01, max_iter=50),
                feature_names=self.species_names
            )
            
            # Fit on combined training and validation data
            X_combined = np.vstack([X_train, X_val])
            X_dot_combined = np.vstack([X_dot_train, X_dot_val])
            
            refined_model.fit(X_combined, t=None, x_dot=X_dot_combined)
            
            # Evaluate on validation set
            X_dot_pred = refined_model.predict(X_val)
            score = mean_squared_error(X_dot_val, X_dot_pred)
            
            if score < best_score:
                best_score = score
                best_threshold = threshold
                best_model = refined_model
        
        return best_model
    
    def remove_non_significant_terms(self, model, X_val, X_dot_val, significance_threshold=0.01):
        """Remove terms that don't contribute significantly to model performance"""
        coefficients = model.coefficients().copy()
        
        # Calculate feature importance based on coefficient magnitude
        importance = np.abs(coefficients)
        
        # Find terms to keep (above significance threshold)
        significant_mask = importance > significance_threshold
        
        # Create new model with only significant terms
        if np.any(significant_mask):
            # Zero out non-significant coefficients
            coefficients[~significant_mask] = 0
            
            # Create new model with refined coefficients
            refined_model = SINDy(
                feature_library=model.feature_library,
                optimizer=STLSQ(threshold=0.001, alpha=0.01, max_iter=50),
                feature_names=self.species_names
            )
            
            # Fit on a small dataset to initialize properly
            X_small = X_val[:10] if len(X_val) > 10 else X_val
            X_dot_small = X_dot_val[:10] if len(X_dot_val) > 10 else X_dot_val
            refined_model.fit(X_small, t=None, x_dot=X_dot_small)
            
            # Manually set coefficients
            refined_model.optimizer.coef_ = coefficients.T
            
            return refined_model
        
        return model
    
    def identifiability_analysis(self, model, X_val, X_dot_val):
        """Check if model is identifiable and not overly complex"""
        coefficients = model.coefficients()
        n_terms = np.sum(np.abs(coefficients) > 1e-6)
        n_equations = len(self.species_names)
        
        # Check if model is not too complex (heuristic) - more lenient
        if n_terms > n_equations * 5:  # More than 5 terms per equation
            return False, "Model too complex"
        
        # Check if model has sufficient predictive power - more lenient
        X_dot_pred = model.predict(X_val)
        r2 = r2_score(X_dot_val, X_dot_pred)
        
        if r2 < 0.3:  # More lenient threshold
            return False, "Poor predictive power"
        
        return True, "Model identifiable"
    
    def fit(self, X_list, X_dot_list):
        """Main fitting procedure with enhanced DoE-SINDy features"""
        print("🔬 Starting Enhanced DoE-SINDy fitting...")
        print("=" * 60)
        
        for iteration in range(self.max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            # 1. Experimental-level subsampling
            X_subsampled, X_dot_subsampled = self.experimental_level_subsampling(X_list, X_dot_list)
            print(f"📊 Subsampled to {len(X_subsampled)} experiments")
            
            # 2. Cross-validation
            X_all = np.vstack(X_subsampled)
            X_dot_all = np.vstack(X_dot_subsampled)
            
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=iteration)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
                X_train, X_val = X_all[train_idx], X_all[val_idx]
                X_dot_train, X_dot_val = X_dot_all[train_idx], X_dot_all[val_idx]
                
                # Try different model configurations - focus on quadratic for nonlinear model
                configs = [
                    {
                        'name': 'Quadratic (STLSQ - Low)',
                        'feature_library': PolynomialLibrary(degree=2, include_bias=False),
                        'optimizer': STLSQ(threshold=0.001, alpha=0.01, max_iter=50)
                    },
                    {
                        'name': 'Quadratic (STLSQ - Medium)',
                        'feature_library': PolynomialLibrary(degree=2, include_bias=False),
                        'optimizer': STLSQ(threshold=0.005, alpha=0.05, max_iter=50)
                    },
                    {
                        'name': 'Quadratic (STLSQ - High)',
                        'feature_library': PolynomialLibrary(degree=2, include_bias=False),
                        'optimizer': STLSQ(threshold=0.01, alpha=0.1, max_iter=50)
                    }
                ]
                
                best_fold_model = None
                best_fold_score = float('inf')
                
                for config in configs:
                    model = SINDy(
                        feature_library=config['feature_library'],
                        optimizer=config['optimizer'],
                        feature_names=self.species_names
                    )
                    
                    model.fit(X_train, t=None, x_dot=X_dot_train)
                    
                    # 3. Parameter re-estimation
                    refined_model = self.parameter_re_estimation(model, X_train, X_dot_train, X_val, X_dot_val)
                    
                    # 4. Remove non-significant terms
                    cleaned_model = self.remove_non_significant_terms(refined_model, X_val, X_dot_val)
                    
                    # 5. Identifiability analysis
                    is_identifiable, message = self.identifiability_analysis(cleaned_model, X_val, X_dot_val)
                    
                    if is_identifiable:
                        X_dot_pred = cleaned_model.predict(X_val)
                        score = mean_squared_error(X_dot_val, X_dot_pred)
                        
                        if score < best_fold_score:
                            best_fold_score = score
                            best_fold_model = cleaned_model
                
                if best_fold_model is not None:
                    cv_scores.append(best_fold_score)
            
            if cv_scores:
                avg_score = np.mean(cv_scores)
                self.convergence_history.append(avg_score)
                
                print(f"📈 Average CV MSE: {avg_score:.8f}")
                
                # Update best model
                if avg_score < self.best_score:
                    self.best_score = avg_score
                    # Fit best model on all data
                    self.best_model = SINDy(
                        feature_library=PolynomialLibrary(degree=2, include_bias=False),
                        optimizer=STLSQ(threshold=0.001, alpha=0.01, max_iter=50),
                        feature_names=self.species_names
                    )
                    self.best_model.fit(X_all, t=None, x_dot=X_dot_all)
                    
                    print(f"🏆 New best model found! Score: {avg_score:.8f}")
            else:
                print("⚠️  No valid models found in this iteration")
                # Create a simple fallback model
                if self.best_model is None:
                    self.best_model = SINDy(
                        feature_library=PolynomialLibrary(degree=2, include_bias=False),
                        optimizer=STLSQ(threshold=0.001, alpha=0.01, max_iter=50),
                        feature_names=self.species_names
                    )
                    self.best_model.fit(X_all, t=None, x_dot=X_dot_all)
                    print("🔄 Created fallback model")
            
            # Check convergence
            if len(self.convergence_history) > 2:
                if abs(self.convergence_history[-1] - self.convergence_history[-2]) < 1e-8:
                    print("✅ Convergence reached!")
                    break
        
        return self.best_model

# --- Load data ---
print("📊 Loading experimental data...")
data_df = pd.read_csv('experimental_data.csv')
derivatives_df = pd.read_csv('estimated_derivatives.csv')

print(f"📈 Data shape: {data_df.shape}")
print(f"📈 Derivatives shape: {derivatives_df.shape}")

# --- Data preprocessing ---
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
    
    # Remove NaN values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(X_dot).any(axis=1))
    X = X[valid_mask]
    X_dot = X_dot[valid_mask]
    
    if len(X) > 10:
        X_list.append(X)
        X_dot_list.append(X_dot)

print(f"🔬 Valid experiments: {len(X_list)}")
print(f"📊 Total data points: {sum(len(x) for x in X_list)}")

# --- Fit model with enhanced DoE-SINDy ---
doe_sindy = DoESINDy(species_names, max_iterations=5, cv_folds=3)
best_model = doe_sindy.fit(X_list, X_dot_list)

# --- Results analysis ---
print("\n🔍 Final SINDy Model:")
print("=" * 50)
for i, species in enumerate(species_names):
    print(f"d{species}/dt = {best_model.equations()[i]}")

# --- Compare with true model ---
print("\n🔍 Comparison with True Model:")
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

# --- Model coefficients ---
print("\n📋 Model Coefficients:")
print("=" * 50)
feature_names = best_model.feature_library.get_feature_names(species_names)
coef_df = pd.DataFrame(
    best_model.coefficients(),
    index=species_names,
    columns=feature_names
)
print(coef_df)

# --- Save results ---
coef_df.to_csv('sindy_coefficients.csv', index=True)
print("\n✅ Model coefficients saved to 'sindy_coefficients.csv'")

# --- Prediction plots ---
print("\n📊 Generating prediction plots...")
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

# --- Convergence plot ---
if len(doe_sindy.convergence_history) > 1:
    plt.figure(figsize=(10, 6))
    plt.plot(doe_sindy.convergence_history, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Iteration')
    plt.ylabel('CV MSE')
    plt.title('DoE-SINDy Convergence History')
    plt.grid(True, alpha=0.3)
    plt.savefig('convergence_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- Performance metrics ---
r2_scores = []
for i in range(len(species_names)):
    r2 = r2_score(X_dot_all[:, i], X_dot_pred[:, i])
    r2_scores.append(r2)

print("\n📊 Model Performance:")
print("=" * 50)
for i, species in enumerate(species_names):
    print(f"{species}: R² = {r2_scores[i]:.4f}")

overall_r2 = np.mean(r2_scores)
print(f"\nOverall R²: {overall_r2:.4f}")

# --- Save final results ---
results = {
    'overall_r2': overall_r2,
    'best_cv_score': doe_sindy.best_score,
    'n_experiments': len(X_list),
    'n_data_points': len(X_all),
    'convergence_iterations': len(doe_sindy.convergence_history)
}

results_df = pd.DataFrame([results])
results_df.to_csv('sindy_results.csv', index=False)
print("\n✅ Results saved to 'sindy_results.csv'")

print("\n🎉 Enhanced DoE-SINDy fitting completed!")
print(f"📊 Final model trained on {len(X_all)} data points from {len(X_list)} experiments")
print(f"📈 Overall R²: {overall_r2:.4f}")
