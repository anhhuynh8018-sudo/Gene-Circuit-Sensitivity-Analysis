import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from MalCoA_ODE import MalCoA_switch
import sys
sys.path.append('/mnt/project')

# Fixed parameters
alpha3 = 0.05
alpha4 = 0.8
beta1 = 0.5
beta2 = 2.0
m = 4
n = 4
p = 4
q = 4
r = 4
u = 4
K1 = 2
K2 = 5
K3 = 2
K4 = 2
K5 = 0.5
K6 = 0.5
k1 = 0.5
k2 = 0.6
k3 = 2
k4 = 2
X0 = 0
S0 = 45
Y_XS = 0.6
Y_PS1 = 0.4
Y_PS2 = 1.8
mu_max = 2.2
K_S = 0.75
K_m = 0.5


def test_bistability_D_alpha(D_val, alpha1_val, alpha2_val, verbose=False):
    """
    Test for bistability at given parameter values by checking if different
    initial conditions converge to different steady states.
    
    Returns:
        bistable: bool
        steady_states: list of final MalCoA concentrations
    """
    params = np.array([D_val, alpha1_val, alpha2_val])
    
    # Test multiple initial conditions
    # Low MalCoA state
    ic_low = [1, 1, 1, 1, 0, 0.1, 1, 1, 1]
    # High MalCoA state  
    ic_high = [1, 1, 1, 1, 0, 10.0, 1, 1, 1]
    # Medium state
    ic_mid = [1, 1, 1, 1, 0, 5.0, 1, 1, 1]
    
    initial_conditions = [ic_low, ic_mid, ic_high]
    steady_states = []
    
    tspan = (0, 200)
    t_eval = np.linspace(0, 200, 1000)
    
    for ic in initial_conditions:
        try:
            sol = solve_ivp(
                MalCoA_switch,
                tspan,
                ic,
                args=(params,),
                t_eval=t_eval,
                method='BDF',
                rtol=1e-8,
                atol=1e-10
            )
            
            if sol.success:
                # Take final MalCoA value
                final_malcoa = sol.y[5, -1]
                steady_states.append(final_malcoa)
                
                if verbose:
                    print(f"  IC MalCoA={ic[5]:.1f} -> final={final_malcoa:.4f}")
            else:
                if verbose:
                    print(f"  IC MalCoA={ic[5]:.1f} -> FAILED")
                return False, []
                
        except Exception as e:
            if verbose:
                print(f"  IC MalCoA={ic[5]:.1f} -> ERROR: {e}")
            return False, []
    
    # Check if steady states differ significantly
    steady_states = np.array(steady_states)
    max_diff = np.max(steady_states) - np.min(steady_states)
    
    # Bistable if different ICs lead to states differing by > 5%
    bistable = max_diff > 0.05 * np.mean(steady_states)
    
    if verbose and bistable:
        print(f"  BISTABLE! States: {steady_states}")
    
    return bistable, steady_states


def scan_parameter_space():
    """
    Scan (D, alpha1) space for bistability with fixed alpha2
    """
    print("Scanning parameter space for bistability...")
    print("=" * 60)
    
    # Parameter grids
    D_values = np.linspace(0.01, 0.3, 20)
    alpha1_values = np.linspace(0.1, 1.0, 20)
    alpha2_fixed = 0.5  # middle of range
    
    bistability_map = np.zeros((len(alpha1_values), len(D_values)))
    
    for i, alpha1 in enumerate(alpha1_values):
        for j, D in enumerate(D_values):
            is_bistable, states = test_bistability_D_alpha(D, alpha1, alpha2_fixed)
            bistability_map[i, j] = 1 if is_bistable else 0
            
        print(f"Progress: {i+1}/{len(alpha1_values)} rows completed")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(bistability_map, 
                   extent=[D_values[0], D_values[-1], 
                          alpha1_values[0], alpha1_values[-1]],
                   origin='lower',
                   aspect='auto',
                   cmap='RdYlGn',
                   vmin=0,
                   vmax=1)
    
    ax.set_xlabel('Dilution Rate D (1/hr)', fontsize=14)
    ax.set_ylabel('FapR Production Rate α₁', fontsize=14)
    ax.set_title(f'Bistability Map (α₂ = {alpha2_fixed})', fontsize=16)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Bistable (1) vs Monostable (0)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/bistability_map.png', dpi=300, bbox_inches='tight')
    print(f"\nBistability map saved!")
    print(f"Bistable parameter combinations: {np.sum(bistability_map)}/{bistability_map.size}")
    
    return D_values, alpha1_values, bistability_map


def demonstrate_hysteresis(D_val=0.15, alpha1_val=0.8, alpha2_val=0.5):
    """
    Demonstrate hysteresis by slowly varying a parameter (e.g., beta1)
    """
    print(f"\nDemonstrating hysteresis at D={D_val}, α₁={alpha1_val}, α₂={alpha2_val}")
    print("=" * 60)
    
    # We'll vary beta1 (FAS basal production) to show hysteresis
    beta1_values_up = np.linspace(0.1, 3.0, 30)
    beta1_values_down = np.linspace(3.0, 0.1, 30)
    
    malcoa_up = []
    malcoa_down = []
    
    params = np.array([D_val, alpha1_val, alpha2_val])
    
    # Start from low state
    y_current = [1, 1, 1, 1, 0, 0.1, 1, 1, 1]
    
    print("Sweeping beta1 UP (0.1 -> 3.0)...")
    for beta1_val in beta1_values_up:
        # Temporarily modify beta1
        import MalCoA_ODE
        MalCoA_ODE.beta1 = beta1_val
        
        sol = solve_ivp(
            MalCoA_switch,
            (0, 100),
            y_current,
            args=(params,),
            method='BDF',
            rtol=1e-8,
            atol=1e-10
        )
        
        if sol.success:
            y_current = sol.y[:, -1]  # Use final state as next IC
            malcoa_up.append(sol.y[5, -1])
        else:
            malcoa_up.append(np.nan)
    
    print("Sweeping beta1 DOWN (3.0 -> 0.1)...")
    # Start from high state
    y_current = [1, 1, 1, 1, 0, 10.0, 1, 1, 1]
    
    for beta1_val in beta1_values_down:
        MalCoA_ODE.beta1 = beta1_val
        
        sol = solve_ivp(
            MalCoA_switch,
            (0, 100),
            y_current,
            args=(params,),
            method='BDF',
            rtol=1e-8,
            atol=1e-10
        )
        
        if sol.success:
            y_current = sol.y[:, -1]
            malcoa_down.append(sol.y[5, -1])
        else:
            malcoa_down.append(np.nan)
    
    # Reset beta1
    MalCoA_ODE.beta1 = 0.5
    
    # Plot hysteresis curve
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(beta1_values_up, malcoa_up, 'b-o', label='Increasing β₁', markersize=4)
    ax.plot(beta1_values_down, malcoa_down, 'r-s', label='Decreasing β₁', markersize=4)
    
    ax.set_xlabel('FAS Basal Production Rate β₁', fontsize=14)
    ax.set_ylabel('Steady-State MalCoA Concentration', fontsize=14)
    ax.set_title('Hysteresis Demonstration', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/hysteresis_curve.png', dpi=300, bbox_inches='tight')
    print("Hysteresis curve saved!")


def test_specific_point():
    """
    Test a specific parameter combination in detail
    """
    print("\nTesting specific parameter point in detail...")
    print("=" * 60)
    
    # Test middle of your parameter ranges
    D = 0.15
    alpha1 = 0.55
    alpha2 = 0.55
    
    print(f"Parameters: D={D}, α₁={alpha1}, α₂={alpha2}")
    
    is_bistable, states = test_bistability_D_alpha(D, alpha1, alpha2, verbose=True)
    
    if is_bistable:
        print(f"\n✓ BISTABLE at this point!")
        print(f"  Steady states differ by: {np.max(states) - np.min(states):.4f}")
    else:
        print(f"\n✗ NOT bistable at this point")
        print(f"  All ICs converge to similar state: {np.mean(states):.4f} ± {np.std(states):.4f}")


if __name__ == "__main__":
    # Test a specific point first
    test_specific_point()
    
    # Scan parameter space
    D_vals, alpha1_vals, bistab_map = scan_parameter_space()
    
    # If bistability found, demonstrate hysteresis
    if np.sum(bistab_map) > 0:
        print("\n" + "=" * 60)
        print("Bistability detected! Demonstrating hysteresis...")
        demonstrate_hysteresis()
    else:
        print("\nNo bistability detected in scanned parameter range.")
        print("Try varying different parameters (beta1, beta2, Hill coefficients)")

    plt.show()
