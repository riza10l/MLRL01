import numpy as np
import os
from monte_carlo.simulator import MonteCarloSimulator

def test_mc_logic():
    print("[TEST] Initializing Monte Carlo Test...")
    
    # 1. Create dummy daily returns (similar to what a real agent would produce)
    # 252 days, mean 0.05% return, 1% daily vol
    np.random.seed(42)
    dummy_returns = np.random.normal(0.0005, 0.01, 252)
    
    # 2. Setup Simulator
    mc = MonteCarloSimulator(n_simulations=100) # 100 sims for speed
    save_dir = "results/test_mc"
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. Run Return Perturbation
    print("[TEST] Running Return Perturbation...")
    result = mc.run_return_perturbation(dummy_returns, noise_std=0.001)
    
    # 4. Generate Report
    report = mc.generate_report(result)
    print("\n" + "="*40)
    print("      MONTE CARLO TEST REPORT")
    print("="*40)
    print(f"Mean Return:   {report['mean_return']:+.2%}")
    print(f"Prob Positive: {report['prob_positive']:.1%}")
    print(f"Prob Ruin 10%: {report['prob_ruin_10pct']:.1%}")
    print(f"Worst MaxDD:   {report['worst_max_dd']:.2%}")
    print("="*40)
    
    # 5. Save Plots
    print(f"\n[TEST] Saving plots to {save_dir}...")
    mc.plot_all(result, save_dir=save_dir)
    print("[TEST] Success! Monte Carlo engine is healthy.")

if __name__ == "__main__":
    test_mc_logic()