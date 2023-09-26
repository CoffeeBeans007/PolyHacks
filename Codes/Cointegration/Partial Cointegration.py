import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import scipy.optimize as opt

def generate_pci(N, alpha, beta, rho,  sigma_M, sigma_R, sigma_X):
    '''
    Generate two partially cointegrated time series of length N with given parameters, such that:
        - X_2,t = alpha + beta * X_1,t + W_t
        - W_t = M_t + R_t
        - M_t = rho * M_t-1 + eps(M_t)
        - R_t = R_t-1 + eps(R_t)
        - eps(M_t) ∼ NID(0, sigma_M)
        - eps(R_t) ∼ NID(0, sigma_R)

    Parameters:
    N (int): Length of the time series to generate.
    alpha (float): intercept
    beta (float): cointegration coefficient.
    rho (float): AR(1) coefficient / mean reversion coefficient.
    sigma_M (float): white noise variance of mean reverting component .
    sigma_R (float): white noise variance of random walk component.
    sigma_X (float): white noise variance of X1.

    Returns:
    tuple: A tuple containing two cointegrated time series, X1 and X2, which are numpy.ndarray.
    '''

    ret_X1 = sigma_X * np.random.randn(N)
    X1 = 100 * np.exp(np.cumsum(ret_X1))  # generate X1
    W = generate_par(N, rho, sigma_M, sigma_R)  # generate PAR residual
    X2 = alpha + beta * X1 + W  # compute X2

    return X1, X2

def generate_par(N, rho, sigma_M, sigma_R):
    '''
    Generate PAR sample of length N with parameters rho, sigma_M, sigma_R

    Parameters:
    N (int): Length of the time series to generate.
    rho (float): AR(1) coefficient / mean reversion coefficient.
    sigma_M (float): white noise variance of mean reverting component .
    sigma_R (float): white noise variance of random walk component.

    Returns:
    numpy.ndarray: A partial autoregressive time series
    '''
    from statsmodels.tsa.arima_process import arma_generate_sample

    ar_M = [1, -rho]
    ar_R = [1, -1]
    ma = [1]

    M = arma_generate_sample(ar_M, ma, N, scale=sigma_M)
    R = arma_generate_sample(ar_R, ma, N, scale=sigma_R)

    return M + R

def fit_pci(X1, X2, tol=0.001):
    '''
    Fit partial cointegrated model to time series X1 and X2 such that:
        - X_2,t = alpha + beta * X_1,t + W_t
        - W_t = M_t + R_t
        - M_t = rho * M_t-1 + eps(M_t)
        - R_t = R_t-1 + eps(R_t)
        - eps(M_t) ∼ NID(0, sigma_M)
        - eps(R_t) ∼ NID(0, sigma_R)

    Parameters:
    X1 (numpy.ndarray): time series
    X2 (numpy.ndarray): time series, supposedly partially cointegrated with X1

    Returns:
    alpha (float): estimated value for alpha
    beta (float): estimated value for beta
    rho (float): estimated AR(1) coefficient / mean reversion coefficient.
    sigma_M (float): estimated white noise variance of mean reverting component .
    sigma_R (float): estimated white noise variance of random walk component.
    '''

    # calculate initial guess for beta
    results = fit_ols_on_diff(X1, X2)
    beta_i = results

    # calculate initial guess for alpha
    alpha_i = X2[0] - beta_i * X1[0]

    # calculate residuals W and initial guesses for rho, sigma_M, sigma_R
    W = X2 - alpha_i - beta_i * X1
    params_i = fit_mle(W)

    # perform optimization depending on the mode
    def f_to_min_pci(parameters, X1, X2):
        alpha ,beta ,rho ,sigma_M ,sigma_R = parameters
        W = X2 - beta *X1 - alpha
        return -calc_log_like(W, rho, sigma_M, sigma_R)

    x_i = (alpha_i, beta_i, *params_i) # initial guess
    res = opt.minimize(f_to_min_pci, x_i, args=(X1, X2), tol=tol)
    alpha ,beta ,rho ,sigma_M ,sigma_R = res.x
    ll = -res.fun

    return alpha, beta, rho, sigma_M, sigma_R, ll

def fit_mle(X, tol=0.001):
    '''
    fit model using Maximum Likelihood estimation. This is used on the residuals W_t of the linear regression of X1 on X2.

    Parameters:
    X (numpy.ndarray): A partial suspected autoregressive time series

    Returns:
    rho (float): estimated value of rho
    sigma_M (float): estimated value of sigma_M
    sigma_R (float): estimated value of sigma_R
    '''

    estimates = []
    lls = []

    # distribution for random guesses
    rnd_rho = stats.uniform(loc=-1, scale=2)

    # get initial guesses using lagged variances
    x_i = lagvar_estimate_par(X)
    # set boundaries
    bounds = ((-1, 1), (0, np.inf), (0, np.inf))
    # function to use in minimization
    f_to_min = f_to_min_par
    # function to generate random initial values
    std = np.std(np.diff(X))
    rnd_sigma = stats.norm(loc=std, scale=std / 2)

    def gen_x0():
        return rnd_rho.rvs(), rnd_sigma.rvs(), rnd_sigma.rvs()

    # minimize
    res = opt.minimize(f_to_min, (x_i), args=(X), bounds=bounds, tol=tol)

    if res.success:
        estimates.append(res.x)
        lls.append(-res.fun)  # save log-likelihood

    # repeat minimization with different (random) initial values
    n_att = 0
    while len(lls) < 10 and n_att < 100:
        n_att += 1
        x0 = gen_x0()
        res = opt.minimize(f_to_min, (x0), args=(X), bounds=bounds, tol=tol)
        if res.success:
            estimates.append(res.x)
            lls.append(-res.fun)  # save log-likelihood

    try:
        argmax = np.argmax(lls)  # index of the biggest likelihood
        return estimates[argmax]
    except:
        # print('Estimation failed!')
        return len(x0) * [np.nan]  # return nans

def lagvar_estimate_par(X):
    '''
    estimate parameters of partial AR model using lagged variances. used for inital estimation of parameters

    Parameters
    X (numpy.ndarray): A partial autoregressive time series

    Returns:
    rho_lv (float): estimated value for rho
    sigma_M_lv (float): estimated value for sigma_M
    sigma_R_lv (float): estimated value for sigma_R
    '''

    # calculate lagged variances
    v1 = np.var(X[1:] - X[:-1])
    v2 = np.var(X[2:] - X[:-2])
    v3 = np.var(X[3:] - X[:-3])

    # rho
    rho_lv = -(v1 - 2 * v2 + v3) / (2 * v1 - v2)

    # sigma_M
    if (rho_lv + 1) / (rho_lv - 1) * (v2 - 2 * v1) > 0:
        sigma_M_lv = np.sqrt(1 / 2 * (rho_lv + 1) / (rho_lv - 1) * (v2 - 2 * v1))
    else:
        sigma_M_lv = 0

    # sigma_R
    if v2 > 2 * sigma_M_lv ** 2:
        sigma_R_lv = np.sqrt(1 / 2 * (v2 - 2 * sigma_M_lv ** 2))
    else:
        sigma_R_lv = 0

    return rho_lv, sigma_M_lv, sigma_R_lv

def kalman_estimate(X, rho, sigma_M, sigma_R):
    '''
    Calculate estimates of mean-reverting and random walk components.

    Parameters:
    X (numpy.ndarray): A partial autoregressive time series
    rho (float): AR(1) coefficient / mean reversion coefficient.
    sigma_M (float): white noise variance of mean reverting component .
    sigma_R (float): white noise variance of random walk component.

    Returns:
    M (numpy.ndarray): An estimate of the mean reverting component of our time series
    R (numpy.ndarray): An estimate of the random walk component of our time series
    eps (numpy.ndarray): Prediction errors for each time step
    '''

    # create arrays for storing both components and prediction errors
    M = np.zeros(len(X))
    R = np.zeros(len(X))
    eps = np.zeros(len(X))

    # set initial state
    if sigma_R == 0:
        M[0] = X[0]
        R[0] = 0
    else:
        M[0] = 0
        R[0] = X[0]

    # calculate Kalman gain
    if sigma_M == 0:
        K_M = 0
        K_R = 1
    elif sigma_R == 0:
        K_M = 1
        K_R = 0
    else:
        sqr = np.sqrt((1 + rho) ** 2 * sigma_R ** 2 + 4 * sigma_M ** 2)
        K_M = 2 * sigma_M ** 2 / (sigma_R * (sqr + rho * sigma_R + sigma_R) + 2 * sigma_M ** 2)
        K_R = 2 * sigma_R / (sqr - rho * sigma_R + sigma_R)

    # calculate estimates
    for i in range(1, len(X)):
        xhat = rho * M[i - 1] + R[i - 1]
        eps[i] = X[i] - xhat
        M[i] = rho * M[i - 1] + eps[i] * K_M
        R[i] = R[i - 1] + eps[i] * K_R

    return M, R, eps

def calc_log_like(X, rho, sigma_M, sigma_R):
    '''
    Compute log likelihood function

    Parameters:
    X (numpy.ndarray): A partial autoregressive time series
    rho (float): AR(1) coefficient / mean reversion coefficient.
    sigma_M (float): white noise variance of mean reverting component .
    sigma_R (float): white noise variance of random walk component.

    Returns:
    ll (float): Value of the log likelihood, a measure of goodness of fit for our model
    '''

    N = len(X)
    _, _, eps = kalman_estimate(X, rho, sigma_M, sigma_R)
    ll = -(N - 1) / 2 * np.log(2 * np.pi * (sigma_M ** 2 + sigma_R ** 2)) - 1 / (
                2 * (sigma_M ** 2 + sigma_R ** 2)) * np.sum(eps[1:] ** 2)

    return ll

def f_to_min_par(parameters, X):
    '''
    Define the function to minimize
    '''
    rho, sigma_M, sigma_R = parameters
    return -calc_log_like(X, rho, sigma_M, sigma_R)

def fit_ols_on_diff(X1, X2):
    '''
    Fits an OLS model on the first differences of time series X1 and X2

    Parameters:
    X1 (numpy.ndarray): A time-series
    X2 (numpy.ndarray): A time-series

    Returns:
    results.params[0]: returns the Beta value of our OLS fit
    '''
    ret_X1 = np.diff(X1)
    ret_X2 = np.diff(X2)

    results = sm.OLS(ret_X2, ret_X1).fit()

    return results.params[0]





