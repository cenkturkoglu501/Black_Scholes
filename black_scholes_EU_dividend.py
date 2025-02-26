import numpy as np
import scipy.stats as si
from scipy.optimize import newton

def dividend_black_scholes(S, K, T, r, sigma, q, option_type="call"):
    """
    Computes the Black-Scholes option price for European options with continuous dividend yield.
    
    Parameters:
        S : float
            Spot price of the underlying asset.
        K : float
            Strike price.
        T : float
            Time to maturity in years.
        r : float
            Risk-free interest rate.
        sigma : float
            Volatility of the underlying asset.
        q : float
            Continuous dividend yield.
        option_type : str, optional
            "call" or "put" (default is "call").
    
    Returns:
        price : float
            Option price.
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * np.exp(-q * T) * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * np.exp(-q * T) * si.norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
    return price

def dividend_black_scholes_greeks(S, K, T, r, sigma, q, option_type="call"):
    """
    Computes the Greeks for European options with continuous dividends.
    
    Returns a dictionary with Delta, Gamma, Vega, Theta (per day), and Rho.
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        delta = np.exp(-q * T) * si.norm.cdf(d1)
        theta = (- (S * sigma * np.exp(-q * T) * si.norm.pdf(d1)) / (2 * np.sqrt(T))
                 + q * S * np.exp(-q * T) * si.norm.cdf(d1)
                 - r * K * np.exp(-r * T) * si.norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * si.norm.cdf(d2) / 100
    elif option_type == "put":
        delta = -np.exp(-q * T) * si.norm.cdf(-d1)
        theta = (- (S * sigma * np.exp(-q * T) * si.norm.pdf(d1)) / (2 * np.sqrt(T))
                 - q * S * np.exp(-q * T) * si.norm.cdf(-d1)
                 + r * K * np.exp(-r * T) * si.norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2) / 100
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    gamma = (np.exp(-q * T) * si.norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * si.norm.pdf(d1) * np.sqrt(T) / 100
    
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

def dividend_implied_volatility(S, K, T, r, q, market_price, option_type="call"):
    """
    Computes the implied volatility for a dividend-paying European option using the Newton-Raphson method.
    """
    def objective_function(sigma):
        return dividend_black_scholes(S, K, T, r, sigma, q, option_type) - market_price
    return newton(objective_function, x0=0.2, tol=1e-6)

def dividend_monte_carlo_option_price(S, K, T, r, sigma, q, num_simulations=100000, option_type="call"):
    """
    Estimates the option price via Monte Carlo simulation for a dividend-paying European option.
    The drift is adjusted for the dividend yield.
    """
    np.random.seed(42)  # For reproducibility
    Z = np.random.standard_normal(num_simulations)
    S_T = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return np.exp(-r * T) * np.mean(payoffs)

# Example usage:
S, K, T, r, sigma, q = 100, 100, 1, 0.05, 0.2, 0.03  # Spot price, Strike, Time to expiry, Risk-free rate, Volatility, Dividend yield
print("Dividend-adjusted European Call Price:", dividend_black_scholes(S, K, T, r, sigma, q, "call"))
print("Dividend-adjusted Greeks:", dividend_black_scholes_greeks(S, K, T, r, sigma, q, "call"))
print("Dividend-adjusted Implied Volatility:", dividend_implied_volatility(S, K, T, r, q, 10, "call"))
print("Dividend-adjusted Monte Carlo Call Price:", dividend_monte_carlo_option_price(S, K, T, r, sigma, q, option_type="call"))
