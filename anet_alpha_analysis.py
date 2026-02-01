"""
Arista Networks (ANET) Alpha Analysis
NNLS regression against MAG7 + Cisco + SPY
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats


TICKERS = {
    'ANET': 'Arista Networks',
    'META': 'Meta',
    'AAPL': 'Apple',
    'GOOGL': 'Google',
    'AMZN': 'Amazon',
    'NVDA': 'Nvidia',
    'MSFT': 'Microsoft',
    'TSLA': 'Tesla',
    'CSCO': 'Cisco',
    'SPY': 'S&P 500 ETF'
}

FACTORS = ['SPY', 'CSCO', 'META', 'AAPL', 'GOOGL', 'AMZN', 'NVDA', 'MSFT', 'TSLA']


def download_stock_data(tickers: list, period: str = "5y") -> pd.DataFrame:
    """Download adjusted close prices for given tickers."""
    print(f"Downloading data for: {', '.join(tickers)}")
    data = yf.download(tickers, period=period, auto_adjust=True)['Close']
    print(f"Downloaded {len(data)} trading days of data")
    print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    return data


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from price data."""
    return prices.pct_change().dropna()


def run_nnls_regression(y: pd.Series, X: pd.DataFrame) -> dict:
    """
    Run non-negative least squares regression.
    Factor loadings constrained >= 0, intercept unconstrained.
    """
    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data.iloc[:, 0].values
    X_clean = data.iloc[:, 1:].values
    n, k = X_clean.shape
    factor_names = X.columns.tolist()
    
    def objective(params):
        alpha = params[0]
        betas = params[1:]
        y_pred = alpha + X_clean @ betas
        return np.sum((y_clean - y_pred) ** 2)
    
    # Initial guess from OLS
    X_with_const = np.column_stack([np.ones(n), X_clean])
    ols_coefs = np.linalg.lstsq(X_with_const, y_clean, rcond=None)[0]
    initial_guess = ols_coefs.copy()
    initial_guess[1:] = np.maximum(initial_guess[1:], 0.001)
    
    # Bounds: intercept unbounded, betas >= 0
    bounds = [(None, None)] + [(0, None)] * k
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
    
    intercept = result.x[0]
    betas = result.x[1:]
    
    # Calculate stats
    y_pred = intercept + X_clean @ betas
    residuals = y_clean - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    
    # Standard errors
    dof = n - k - 1
    sigma2 = ss_res / dof
    X_with_const = np.column_stack([np.ones(n), X_clean])
    try:
        var_coef = sigma2 * np.linalg.inv(X_with_const.T @ X_with_const)
        se = np.sqrt(np.diag(var_coef))
        se_intercept = se[0]
        se_betas = se[1:]
    except np.linalg.LinAlgError:
        se_intercept = np.nan
        se_betas = np.full(k, np.nan)
    
    t_intercept = intercept / se_intercept if se_intercept != 0 else np.nan
    p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), dof)) if not np.isnan(t_intercept) else np.nan
    
    return {
        'intercept': intercept,
        'betas': dict(zip(factor_names, betas)),
        'se_intercept': se_intercept,
        'se_betas': dict(zip(factor_names, se_betas)),
        't_intercept': t_intercept,
        'p_intercept': p_intercept,
        'r_squared': r_squared,
        'n_obs': n,
        'y_clean': y_clean,
        'X_clean': X_clean,
        'factor_names': factor_names
    }


def calculate_variance_decomposition(result: dict, y: pd.Series, X: pd.DataFrame) -> dict:
    """Calculate SHAP-style variance decomposition."""
    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]
    
    var_y = y_clean.var()
    contributions = {}
    vol_contributions = {}
    
    for factor in X_clean.columns:
        beta = result['betas'].get(factor, 0)
        cov_xy = y_clean.cov(X_clean[factor])
        contributions[factor] = beta * cov_xy / var_y
        vol_contributions[factor] = np.sqrt((beta ** 2) * X_clean[factor].var()) * np.sqrt(252)
    
    return {
        'r2_contributions': contributions,
        'total_r2': sum(contributions.values()),
        'vol_contributions': vol_contributions
    }


def print_summary_stats(returns: pd.DataFrame):
    """Print summary statistics for returns."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (Annualized)")
    print("="*80)
    
    annual_returns = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_returns / annual_vol
    
    stats_df = pd.DataFrame({
        'Annual Return': annual_returns,
        'Annual Vol': annual_vol,
        'Sharpe Ratio': sharpe
    }).sort_values('Annual Return', ascending=False)
    
    print(stats_df.round(4).to_string())
    
    print("\n" + "-"*80)
    print("CORRELATION MATRIX")
    print("-"*80)
    print(returns.corr().round(3).to_string())


def print_results(result: dict, decomp: dict):
    """Print NNLS regression results with variance decomposition."""
    print("\n" + "="*85)
    print("NNLS REGRESSION: ANET ~ SPY + CSCO + MAG7")
    print("Coefficients constrained >= 0, intercept unconstrained")
    print("="*85)
    
    # Key stats
    annual_alpha = result['intercept'] * 252 * 100
    print(f"\nObservations: {result['n_obs']}")
    print(f"R-squared: {result['r_squared']:.4f}")
    print(f"\n*** Annualized Alpha: {annual_alpha:+.2f}% ***")
    print(f"    t-stat: {result['t_intercept']:.3f}, p-value: {result['p_intercept']:.4f}")
    
    # Coefficients table
    print("\n" + "-"*85)
    print("COEFFICIENTS")
    print("-"*85)
    print(f"{'Factor':<10} {'Beta':>10} {'Std Err':>12} {'t-stat':>10} {'Constrained':>12}")
    print("-"*60)
    
    for factor in FACTORS:
        beta = result['betas'][factor]
        se = result['se_betas'][factor]
        t = beta / se if se != 0 and not np.isnan(se) else np.nan
        constrained = "Yes (=0)" if beta == 0 else ""
        print(f"{factor:<10} {beta:>10.4f} {se:>12.6f} {t:>10.3f} {constrained:>12}")
    
    # Variance decomposition
    print("\n" + "-"*85)
    print("VARIANCE DECOMPOSITION (SHAP-style)")
    print("-"*85)
    print(f"{'Factor':<10} {'Beta':>10} {'R² Contrib':>12} {'% of R²':>10} {'Vol Contrib':>14}")
    print("-"*60)
    
    r2_contribs = decomp['r2_contributions']
    vol_contribs = decomp['vol_contributions']
    total_r2 = decomp['total_r2']
    
    sorted_factors = sorted(FACTORS, key=lambda x: abs(r2_contribs[x]), reverse=True)
    
    for factor in sorted_factors:
        beta = result['betas'][factor]
        r2_contrib = r2_contribs[factor]
        pct_of_r2 = (r2_contrib / total_r2 * 100) if total_r2 != 0 else 0
        vol_contrib = vol_contribs[factor]
        print(f"{factor:<10} {beta:>10.4f} {r2_contrib:>12.4f} {pct_of_r2:>9.1f}% {vol_contrib:>13.2f}%")
    
    print("-"*60)
    print(f"{'TOTAL':<10} {'':<10} {total_r2:>12.4f} {'100.0':>9}%")
    print(f"{'Residual':<10} {'':<10} {1-total_r2:>12.4f}")
    print("="*85)


def main():
    # Download data
    tickers = list(TICKERS.keys())
    prices = download_stock_data(tickers, period="5y")
    
    # Calculate daily returns
    returns = calculate_returns(prices)
    
    # Print summary stats
    print_summary_stats(returns)
    
    # Run NNLS regression
    y = returns['ANET']
    X = returns[FACTORS]
    result = run_nnls_regression(y, X)
    
    # Calculate variance decomposition
    decomp = calculate_variance_decomposition(result, y, X)
    
    # Print results
    print_results(result, decomp)
    
    # Save data
    returns.to_csv('anet_returns.csv')
    print("\n✓ Returns saved to 'anet_returns.csv'")
    
    return returns, result, decomp


if __name__ == "__main__":
    returns, result, decomp = main()
