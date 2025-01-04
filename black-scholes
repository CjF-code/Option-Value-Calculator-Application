import numpy as np
import pandas as pd
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache


def blackScholes(r, S, K, T, sigma, type="c"):
    """
    Black-Schole option price calulation function for both a call and a put.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type == "c":
        price = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
    elif type == "p":
        price = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)
    return price


def delta_calc(r, S, K, T, sigma, type="c"):
    """
    Delta measures the rate of change in the theoretical option value with respect to the underlying asset's price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if type == "c":
        delta = norm.cdf(d1, 0, 1)
    elif type == "p":
        delta = -norm.cdf(-d1, 0, 1)
    return delta


def gamma_calc(r, S, K, T, sigma):
    """
    Gamma measures the rate of change in the delta with respect to changes in the underlying price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    gamma = norm.pdf(d1, 0, 1) / (S * sigma * np.sqrt(T))
    return gamma


def vega_calc(r, S, K, T, sigma):
    """
    Vega measures sensitivity to volatility. Vega is the derivative of the option value with respect to volatility of the underlying asset.
    We multiply by 0.01 so show the sensivity to a 1% change in underlying volatility rather than a 100% change involatility of the underlying.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    vega = S * norm.pdf(d1, 0, 1) * np.sqrt(T)
    return vega * 0.01


def theta_calc(r, S, K, T, sigma, type="c"):
    """
    Theta measures the sesitivity of the value of the derivative to the passage of time - time decay.
    We divide by 365.25 to measure the sensitivty to the passage of 1 day rather then 1 year.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == "c":
        theta = -(S * norm.pdf(d1, 0, 1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(
            -r * T
        ) * norm.cdf(d2, 0, 1)
    elif type == "p":
        theta = -(S * norm.pdf(d1, 0, 1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(
            -r * T
        ) * norm.cdf(-d2, 0, 1)
    return theta / 365


def rho_calc(r, S, K, T, sigma, type="c"):
    """
    Rho measures sensitivity to changes in the interest rate.
    We multiply by 0.01 to measure the sensitivty to a 1% change in the interest rate rather than 100% change.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type == "c":
        rho_calc = K * T * np.exp(-r * T) * norm.cdf(d2, 0, 1)
    elif type == "p":
        rho_calc = -K * T * np.exp(-r * T) * norm.cdf(-d2, 0, 1)
    return rho_calc * 0.01


def call_payoff(spot, strike, premium, position_type="long"):
    """
    Calculate the payoff of a call option.
    """
    if position_type == "long":
        return np.maximum(spot - strike, 0) - premium
    else:  # short position
        return premium - np.maximum(spot - strike, 0)


def put_payoff(spot, strike, premium, position_type="long"):
    """
    Calculate the payoff of a put option.
    """
    if position_type == "long":
        return np.maximum(strike - spot, 0) - premium
    else:  # short position
        return premium - np.maximum(strike - spot, 0)


def create_heatmap(option_type, x_axis_var, y_axis_var, option_obj, prices, axis_dict):
    """
    Create a heatmap visualization for option prices, to be outputed into a streamlit environment.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        prices,
        xticklabels=np.round(getattr(option_obj, f"{axis_dict[x_axis_var]}_range"), 2),
        yticklabels=np.round(getattr(option_obj, f"{axis_dict[y_axis_var]}_range"), 2),
        cmap="viridis",
        cbar_kws={"label": f"{option_type} Option Price"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
    )

    plt.xlabel(f"{x_axis_var} ({axis_dict[x_axis_var]})")
    plt.ylabel(f"{y_axis_var} ({axis_dict[y_axis_var]})")
    plt.title(
        f"{option_type} Option Prices Heatmap showing +/- 5% change in chosen variables"
    )

    return st.pyplot(plt)


def get_SandP_500_tickers():
    """
    Obtains a list of S&P 500 tickers
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    sp500_table = pd.read_html(url, header=0)[0]  # Get the first table
    tickers = sp500_table["Symbol"].tolist()

    # Some tickers may have a dot in them, replace with dash to match yfinance
    tickers = [ticker.replace(".", "-") for ticker in tickers]
    return tickers


def get_spot_price(ticker):
    """
    Obtains current training price of chosen ticker
    """
    stock = yf.Ticker(ticker)
    spot_price = stock.fast_info["lastPrice"]
    return spot_price


def get_implied_volatility(ticker, target_days):
    """
    Obtains implied volatility for a stock ticker using yfinance, using the closest available volatility to the requested target
    days, and return the actual days that was used
    """
    stock = yf.Ticker(ticker)

    # Get all available option expiration dates
    expiration_dates = stock.options

    # Convert expiration dates to days from now
    today = datetime.now()
    days_to_expiry = [
        (datetime.strptime(date, "%Y-%m-%d") - today).days for date in expiration_dates
    ]

    # Find closest expiration to target_days
    closest_idx = min(
        range(len(days_to_expiry)), key=lambda i: abs(days_to_expiry[i] - target_days)
    )
    closest_date = expiration_dates[closest_idx]
    actual_days = days_to_expiry[closest_idx]

    # Get options chain for closest expiration
    chain = stock.option_chain(closest_date)

    # Calculate weighted average implied volatility for this specific term
    calls_iv = chain.calls["impliedVolatility"].mean()
    puts_iv = chain.puts["impliedVolatility"].mean()
    implied_volatility = (calls_iv + puts_iv) / 2

    return implied_volatility, actual_days


def live_available_options(ticker):
    """
    Returns the spot price, a list of available terms (days to maturity),
    and a dictionary where the key is the term and the value is a list of available strike prices for that term.
    """
    stock = _get_cached_ticker(ticker)

    # Obtain history data for chosen ticker, then current price
    current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    history_data = stock.history(period="1d")
    spot_price = history_data["Close"].iloc[-1]

    # Get all available option expiration dates, define strikes dict and available terms
    available_expirations = stock.options
    strikes_by_term = {}
    available_terms = set()

    # Obtain the expirations dates where options were traded
    for expiry in available_expirations:
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
        days_to_expiry = (expiry_date - current_date).days

        # Obtain strikes where both a call and a put have a valid price
        options_chain = stock.option_chain(expiry)
        common_strikes = sorted(
            list(
                set(options_chain.calls["strike"]).intersection(
                    set(options_chain.puts["strike"])
                )
            )
        )

        # Create a dictionary for strikes where they are matched to the durations of the option
        if common_strikes:
            strikes_by_term[days_to_expiry] = common_strikes
            available_terms.add(days_to_expiry)

    available_terms_list = sorted(list(available_terms))

    return spot_price, available_terms_list, strikes_by_term


def live_option_price_calc(ticker, strike_price, days_to_expiry):
    """
    Obtains a live spot price, as well as a call and put option price for a given stock ticker based on the provided expiry data and strike price.
    """
    # Get cached ticker data
    stock = _get_cached_ticker(ticker)

    # Calculate target expiration date
    current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    target_date = current_date + timedelta(days=days_to_expiry)

    # Find closest expiration date
    available_expirations = stock.options
    closest_expiry = min(
        available_expirations,
        key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - target_date).days),
    )

    # Get options chain for the closest expiry
    chain = stock.option_chain(closest_expiry)

    # Get call price
    call_price = None
    specific_call = chain.calls[chain.calls["strike"] == strike_price]
    if not specific_call.empty:
        # Prefer last trade price if available, otherwise use bid-ask midpoint
        if specific_call["volume"].iloc[0] > 0:
            call_price = specific_call["lastPrice"].iloc[0]
        else:
            bid = specific_call["bid"].iloc[0]
            ask = specific_call["ask"].iloc[0]
            call_price = (bid + ask) / 2 if bid > 0 and ask > 0 else None

    # Get put price
    put_price = None
    specific_put = chain.puts[chain.puts["strike"] == strike_price]
    if not specific_put.empty:
        # Prefer last trade price if available, otherwise use bid-ask midpoint
        if specific_put["volume"].iloc[0] > 0:
            put_price = specific_put["lastPrice"].iloc[0]
        else:
            bid = specific_put["bid"].iloc[0]
            ask = specific_put["ask"].iloc[0]
            put_price = (bid + ask) / 2 if bid > 0 and ask > 0 else None

    actual_expiration_date = datetime.strptime(closest_expiry, "%Y-%m-%d").date()

    return actual_expiration_date, call_price, put_price


@lru_cache(maxsize=100)
def _get_cached_ticker(ticker):
    """Cache ticker objects to improve performance for repeated calls."""
    return yf.Ticker(ticker)


# Option class used for easy calculation based on user inputs
class Option:
    def __init__(self, r, S, K, T, sigma, type):
        self.rfr = r
        self.underlying_price = S
        self.strike = K
        self.expiry = T
        self.volatility = sigma
        self.type = type

        self.option_price = blackScholes(r, S, K, T, sigma, type)

        self.delta = delta_calc(r, S, K, T, sigma, type)
        self.gamma = gamma_calc(r, S, K, T, sigma)
        self.vega = vega_calc(r, S, K, T, sigma)
        self.theta = theta_calc(r, S, K, T, sigma, type)
        self.rho = rho_calc(r, S, K, T, sigma, type)

        self.S_range = list(np.round(np.arange(S * 0.95, S * 1.0500001, S * 0.01), 2))
        self.K_range = list(np.round(np.arange(K * 0.95, K * 1.0500001, K * 0.01), 2))
        self.T_range = list(np.round(np.arange(T * 0.95, T * 1.0500001, T * 0.01), 2))
        self.sigma_range = list(
            np.round(np.arange(sigma * 0.95, sigma * 1.0500001, sigma * 0.01), 4)
        )
