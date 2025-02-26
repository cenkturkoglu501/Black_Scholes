import numpy as np
import scipy.stats as si
from scipy.optimize import newton
from math import exp

# This program helps calculate the following:
# Black-Scholes Pricing for European and American call and put options. To switch between European and American options, please change the "style" in the definition of black_scholes function (You can write "european" or "american"). To switch between call and put options, change "option_type" in the same definition (You can write "call" or "put").
# Greeks Calculation (Delta, Gamma, Vega, Theta, Rho)
# Implied Volatility Solver using the Newton-Raphson method
# Monte Carlo Simulation to estimate the option price

def black_scholes(S, K, T, r, sigma, option_type="call", style="european"):
    """Computes the Black-Scholes option price. Supports European and American options."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if style == "european":
        if option_type == "call":
            price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        elif option_type == "put":
            price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
    elif style == "american":
        if option_type == "call":
            price = max(S - K, S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2))
        elif option_type == "put":
            price = max(K - S, K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1))
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
    else:
        raise ValueError("Invalid option style. Use 'european' or 'american'.")
    
    return price

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """Computes Black-Scholes Greeks."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = si.norm.cdf(d1) if option_type == "call" else -si.norm.cdf(-d1)
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)) / 365
    rho = K * T * np.exp(-r * T) * si.norm.cdf(d2 if option_type == "call" else -d2) / 100
    
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

def implied_volatility(S, K, T, r, market_price, option_type="call"):
    """Finds implied volatility using Newton-Raphson method."""
    def objective_function(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - market_price
    
    return newton(objective_function, x0=0.2, tol=1e-6)

def monte_carlo_option_price(S, K, T, r, sigma, num_simulations=100000, option_type="call"):
    """Monte Carlo simulation for Black-Scholes option pricing."""
    np.random.seed(42)  # For reproducibility
    Z = np.random.standard_normal(num_simulations)
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return np.exp(-r * T) * np.mean(payoffs)

# Example usage:
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2  # Spot price, Strike, Time to expiry, Risk-free rate, Volatility
print("Black-Scholes Call Price (European):", black_scholes(S, K, T, r, sigma, "call", "european"))
print("Black-Scholes Call Price (American):", black_scholes(S, K, T, r, sigma, "call", "american"))
print("Greeks:", black_scholes_greeks(S, K, T, r, sigma, "call"))
print("Implied Volatility:", implied_volatility(S, K, T, r, 10, "call"))
print("Monte Carlo Call Price:", monte_carlo_option_price(S, K, T, r, sigma))
