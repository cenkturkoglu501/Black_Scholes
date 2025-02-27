import numpy as np
import pandas as pd

def simulate_portfolio_paths(portfolio_weights, historical_returns, num_simulations, T, steps):
    
#    Simulate portfolio value paths over a time horizon T using geometric Brownian motion.
    
#    Parameters:
#    - portfolio_weights: array-like, shape (n_assets,)
#    - historical_returns: DataFrame or ndarray of shape (n_periods, n_assets)
#    - num_simulations: number of Monte Carlo simulation paths
#    - T: time horizon in years
#    - steps: number of discrete time steps in T (e.g. 252 for daily over one year)
    
#    Returns:
#    - paths: NumPy array of shape (num_simulations, steps+1) with simulated portfolio values
#    - dt: time step size
#    - portfolio_drift: estimated drift of the portfolio
#    - portfolio_volatility: estimated volatility of the portfolio
    
    # Ensure historical returns is a NumPy array
    if isinstance(historical_returns, pd.DataFrame):
        returns = historical_returns.values
    else:
        returns = historical_returns
        
    # Estimate asset-level drift and covariance from historical returns
    mu_assets = np.mean(returns, axis=0)
    cov_assets = np.cov(returns.T)
    
    # Calculate portfolio drift and volatility from asset statistics
    portfolio_drift = np.dot(portfolio_weights, mu_assets)
    portfolio_variance = portfolio_weights @ cov_assets @ portfolio_weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    dt = T / steps
    paths = np.zeros((num_simulations, steps + 1))
    paths[:, 0] = 1.0  # Starting portfolio value (normalized)
    
    # Simulate paths using the geometric Brownian motion formula
    for t in range(1, steps + 1):
        # Generate standard normal increments for all simulation paths
        Z = np.random.normal(0, 1, num_simulations)
        paths[:, t] = paths[:, t-1] * np.exp((portfolio_drift - 0.5 * portfolio_volatility**2) * dt
                                              + portfolio_volatility * np.sqrt(dt) * Z)
    
    return paths, dt, portfolio_drift, portfolio_volatility

def compute_exposure_profiles(paths):
    """
    Compute exposure profiles over time from the simulated portfolio paths.
    
    Returns:
    - EE: Expected Exposure (average positive portfolio value at each time)
    - ENE: Expected Negative Exposure (average of negative portfolio values)
    """
    # Positive exposures (for CVA/FVA/KVA)
    EE = np.maximum(paths, 0).mean(axis=0)
    # Negative exposures (for DVA)
    ENE = np.maximum(-paths, 0).mean(axis=0)
    return EE, ENE

def compute_xva_adjustments(paths, dt, risk_free_rate, hazard_rate_cpty, rec_cpty,
                            hazard_rate_bank, rec_bank, funding_spread, cost_of_capital, capital_factor):
    """
    Compute dynamic XVA adjustments over the time horizon.
    
    The adjustments are computed discretely over time. For each time step:
      - CVA = (1 - rec_cpty) * dPD_cpty * EE, discounted to present
      - DVA = (1 - rec_bank) * dPD_bank * ENE, discounted to present
      - FVA = funding_spread * EE, discounted to present
      - KVA = cost_of_capital * capital_factor * EE, discounted to present
      
    Parameters:
    - paths: simulated portfolio paths (num_simulations x steps+1)
    - dt: time step (in years)
    - risk_free_rate: annual risk-free rate used for discounting
    - hazard_rate_cpty: counterparty’s annual hazard rate
    - rec_cpty: counterparty recovery rate
    - hazard_rate_bank: bank’s (own) annual hazard rate
    - rec_bank: bank’s recovery rate
    - funding_spread: annualized funding spread cost
    - cost_of_capital: annual cost of capital rate
    - capital_factor: fraction of exposure required as capital (regulatory requirement)
    
    Returns:
    - CVA, DVA, FVA, KVA: the respective adjustments as scalars.
    """
    steps = paths.shape[1] - 1
    # Create a time grid for discounting (exclude t=0)
    time_grid = np.linspace(dt, steps * dt, steps)
    discount_factors = np.exp(-risk_free_rate * time_grid)
    
    # Compute exposure profiles at each time step (excluding t=0)
    EE, ENE = compute_exposure_profiles(paths)
    EE = EE[1:]
    ENE = ENE[1:]
    
    # Incremental default probability for each time step (assuming constant hazard rate)
    dPD_cpty = hazard_rate_cpty * dt
    dPD_bank = hazard_rate_bank * dt
    
    # CVA: loss due to counterparty default on positive exposures
    CVA = np.sum(discount_factors * (1 - rec_cpty) * dPD_cpty * EE)
    
    # DVA: benefit due to own default on negative exposures
    DVA = np.sum(discount_factors * (1 - rec_bank) * dPD_bank * ENE)
    
    # FVA: cost of funding the positive exposures
    FVA = np.sum(discount_factors * funding_spread * EE)
    
    # KVA: cost of capital; here assumed to be a proportion of the positive exposure that must be funded
    KVA = np.sum(discount_factors * cost_of_capital * capital_factor * EE)
    
    return CVA, DVA, FVA, KVA

def compute_var_es_from_paths(paths, confidence_level=0.95):
    """
    Compute Value-at-Risk (VaR) and Expected Shortfall (ES) using the terminal portfolio values.
    
    Losses are defined as the negative of the portfolio return over the horizon.
    
    Parameters:
    - paths: simulated portfolio paths (num_simulations x steps+1)
    - confidence_level: e.g., 0.95 for 95% confidence
    
    Returns:
    - VaR: Value-at-Risk
    - ES: Expected Shortfall (average loss beyond VaR)
    """
    terminal_values = paths[:, -1]
    # Portfolio return over the period (initial value is 1)
    portfolio_returns = terminal_values - 1
    losses = -portfolio_returns
    VaR = np.percentile(losses, confidence_level * 100)
    ES = losses[losses >= VaR].mean()
    return VaR, ES

# Example usage:
if __name__ == '__main__':
    np.random.seed(42)
    
    # -------------------------
    # Market and Portfolio Setup
    # -------------------------
    # Simulate historical returns for 3 assets (for drift and volatility estimation)
    n_assets = 3
    n_periods = 1000
    historical_returns = np.random.normal(0.0005, 0.01, (n_periods, n_assets))
    portfolio_weights = np.array([0.4, 0.4, 0.2])
    
    # -------------------------
    # Simulation Parameters
    # -------------------------
    num_simulations = 10000  # Number of Monte Carlo paths
    T = 1.0                  # Time horizon of 1 year
    steps = 252              # Daily steps (approximately)
    
    # -------------------------
    # XVA & Discounting Parameters
    # -------------------------
    risk_free_rate = 0.03    # 3% annual risk-free rate
    hazard_rate_cpty = 0.02  # 2% annual hazard rate for the counterparty
    rec_cpty = 0.4           # 40% recovery rate for the counterparty
    hazard_rate_bank = 0.015 # 1.5% annual hazard rate for the bank itself
    rec_bank = 0.4           # 40% recovery rate for the bank
    funding_spread = 0.005   # 0.5% annual funding spread
    cost_of_capital = 0.08   # 8% annual cost of capital
    capital_factor = 0.1     # Assume 10% of exposure is required as capital
    
    # -------------------------
    # Run Simulations and Compute Metrics
    # -------------------------
    # Simulate the portfolio paths
    paths, dt, portfolio_drift, portfolio_volatility = simulate_portfolio_paths(
        portfolio_weights, historical_returns, num_simulations, T, steps
    )
    
    # Compute dynamic XVA adjustments over the time horizon
    CVA, DVA, FVA, KVA = compute_xva_adjustments(
        paths, dt, risk_free_rate, hazard_rate_cpty, rec_cpty,
        hazard_rate_bank, rec_bank, funding_spread, cost_of_capital, capital_factor
    )
    
    # Compute VaR and Expected Shortfall based on terminal portfolio values
    VaR, ES = compute_var_es_from_paths(paths, confidence_level=0.95)
    
    # -------------------------
    # Output the Results
    # -------------------------
    print("Market Risk Metrics:")
    print("Value-at-Risk (95% confidence, 1-year):", VaR)
    print("Expected Shortfall (1-year):", ES)
    print("\nXVA Adjustments:")
    print("CVA:", CVA)
    print("DVA:", DVA)
    print("FVA:", FVA)
    print("KVA:", KVA)
