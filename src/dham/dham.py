"""
DHAM skin permeability analysis.

Estimates the skin permeation coefficient (kP) of a small molecule from GROMACS umbrella
sampling (US) simulations of that molecule crossing a model stratum corneum membrane.

Pipeline (see dham() for the driver):
  1. Read the US pullx trajectories and log files (umbrella centres, force constants, temperature).
  2. Discard the equilibration portion of each trajectory (autocut).
  3. Choose the Markov model bin number and lagtime from the implied timescales (mm_tuning).
  4. Build an unbiased Markov model with the Dynamic Histogram Analysis Method (DHAM)
     and take its stationary distribution as the free energy profile (PMF).
  5. Estimate position-dependent diffusion D(z) from the Kramers-Moyal expansion (km_diffusion).
  6. Combine PMF and D(z) in the inhomogeneous solubility-diffusion model to give kP (calc_perm),
     optionally adding a shunt-route (sweat glands / hair follicles) contribution.
  7. Estimate uncertainties by repeating the analysis on four consecutive blocks of the data.

Conventions used throughout:
  - z is in Angstrom and taken as |z|, i.e. the membrane is assumed symmetric about z = 0.
    Final profiles are mirrored to cover -z ... +z.
  - Bias constants and Markov model energies are in kT; reported PMFs are in kJ/mol.
  - Lagtimes are in trajectory frames (multiply by the pullx output interval for time).
"""

# Required package imports
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, ArpackError
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
import scipy.signal as sig
from matplotlib import cm, colors
from numba import njit, prange

# Standard Plot Settings
plt.rcParams["figure.figsize"] = (11.7,8.3)
STANDARD_SIZE=24
SMALL_SIZE=18
plt.rc('font', size=STANDARD_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=STANDARD_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=STANDARD_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


R = 0.008314 # universal gas constant (kJ mol^-1 K^-1)

# Number of consecutive blocks the data is split into for error estimates
N_ERROR_BLOCKS = 4

def top_eigs(mm, k=3):
    """
    Return the k largest-magnitude eigenvalues (and eigenvectors) of the transposed Markov matrix.

    The left eigenvectors of mm (right eigenvectors of mm^T) give state populations; the largest
    eigenvalue (1) gives the stationary distribution and the second largest gives the slowest
    relaxation timescale.

    Uses the sparse ARPACK solver, which only computes the few eigenvalues needed and is much
    faster than a full decomposition. Falls back to the dense solver if ARPACK fails to converge.

    Parameters
    ----------
    mm : ndarray, shape (M, M)
        Row-stochastic Markov transition probability matrix.
    k : int
        Number of eigenvalues to compute.

    Returns
    -------
    eigenvalues : ndarray (complex)
    eigenvectors : ndarray (complex), one eigenvector per column
    """
    try:
        return eigs(csr_matrix(np.transpose(mm)), k=k, which='LM')
    except ArpackError:
        return eig(np.transpose(mm))

def smooth(profile):
    """
    Apply a light moving-average filter to a 1D profile.

    Each point is replaced by the mean of a window of 2 * (1% of the profile length) points,
    and the edges are padded with the nearest fully-smoothed value.
    Profiles shorter than 100 points are returned unsmoothed (the window would be empty).

    Parameters
    ----------
    profile : 1D array

    Returns
    -------
    smoothed : 1D array, same length as profile
    """

    smoothing_interval = int(0.01 * len(profile))

    if smoothing_interval < 1:
        return np.array(profile, dtype=np.float64)

    smoothed = np.zeros(len(profile))

    for i in range(smoothing_interval, len(profile)):
        smoothed[i] = np.mean(profile[i-smoothing_interval:i+smoothing_interval])

    # Fill start/end edges with the nearest valid values
    smoothed[:smoothing_interval] = smoothed[smoothing_interval]
    smoothed[-smoothing_interval:] = smoothed[-smoothing_interval]

    return smoothed

def zero_min(profile):
    """
    Shift a PMF so its minimum is zero.

    Works on 1D profiles and on 2D arrays of profiles (each row shifted independently).
    """
    return profile - np.min(profile, axis=-1, keepdims=True)

def mirror(profile):
    """
    Mirror a profile defined on |z| (0 -> z_max) to cover the whole membrane (-z_max -> z_max).

    Works on 1D profiles and on 2D arrays of profiles (one profile per row).
    """
    return np.concatenate((profile[..., ::-1], profile), axis=-1)

def autocut(trajectories, thresh=0.01):
    """
    Determines the longest decorrelation time across multiple trajectories.

    For each trajectory:
      - subtract the mean (center it)
      - compute the autocorrelation (via signal correlation)
      - find the first lag where autocorrelation falls <= thresh

    Returns the maximum of these "cut" indices across trajectories so that
    subsequent processing discards enough initial data to ensure decorrelation.

    Parameters:
    -----------
    trajectories : list of np.array
        Each element is a time series (trajectory) to analyze.
    thresh : float
        Threshold for determining decorrelation (default 0.01).

    Returns:
    --------
    trajectory_cut : int
        Maximum index where autocorrelation falls below threshold across all trajectories.
    """

    trajectory_cut = 0   # Initialize maximum decorrelation index

    for traj in tqdm(trajectories, desc='Computing equilibration cut-off...'):
        # Center the trajectory by subtracting the mean
        tm = traj - np.mean(traj)

        # Compute autocorrelation of the centered trajectory
        autocorr = sig.correlate(tm, tm)

        # Normalize so that autocorrelation at lag 0 is 1
        autocorr = autocorr / (np.var(tm) * len(tm))

        # Take only the non-negative lags
        autocorr = autocorr[autocorr.size // 2:]

        # Find indices where autocorrelation falls below the threshold
        mask = (autocorr <= thresh)
        cut_indices = np.where(mask)[0][0]  # First index below threshold

        # Update the maximum decorrelation index if this trajectory is longer
        if cut_indices >= trajectory_cut:
            trajectory_cut = cut_indices

    # Return the largest decorrelation index across all trajectories
    return trajectory_cut

def implied_timescale(d_traj, nbins, lag, bin_edges, uc, bk):
    """
    Build a DHAM Markov model at the given lagtime and return log10 of its slowest implied timescale.

    The implied relaxation timescale is t = -lag / ln|lambda_2|, where lambda_2 is the second
    largest eigenvalue (by magnitude) of the Markov matrix. A well-chosen model gives a timescale
    that no longer changes with lagtime.

    Parameters
    ----------
    d_traj : ndarray (ntraj, nsteps)
        Discretized trajectories.
    nbins : int
        Number of discrete states.
    lag : int
        Lagtime (frames).
    bin_edges, uc, bk : arrays
        Bin edges, umbrella centers and bias constants (see gen_mm_numba).

    Returns
    -------
    log_its : float
        log10 of the implied timescale (frames).
    """

    # Generate transition matrix and Markov model
    transition_matrix, transition_sums = gcm(d_traj, nbins, lag)
    mm = gen_mm_numba(transition_matrix, transition_sums, bin_edges, uc, bk)

    # Extract matrix eigenvalues
    eigenvalues, _ = top_eigs(mm)

    # Sort eigenvalues by magnitude, descending
    sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]

    # The second largest eigenvalue (by magnitude)
    lambda2 = sorted_eigs[1]

    return np.log10(-lag/(np.log(lambda2)))

def mm_tuning(u_data, uc, bk, result_path):
    """
    Tune Markov model discretization (number of bins) and lagtime.

    Approach:
      - For a set of trial bin counts and trial lagtimes, compute the
        Implied Relaxation Timescales (ITS) from the second eigenvalue.
      - For each trial bin count, take the maximum ITS across lagtimes to
        produce a curve of ITS_max vs bin_count.
      - Fit that curve and find where curvature becomes small -> ideal bin number.
      - With ideal bin number fixed, compute ITS vs lagtime and fit an exponential-like
        function, differentiate twice and find where the second derivative
        falls below a threshold -> ideal lagtime.

    Parameters
    ----------
    u_data : 2D array (ntraj, nsteps)
        Equilibrated trajectory data (reaction coordinate) used for discretization.
    uc : 1D array
        Umbrella centers (used in DHAM weighting).
    bk : 1D array
        Biasing constants (one per umbrella).
    result_path : Path
        Directory to save plots and outputs.

    Returns
    -------
    ideal_bin_no : int
        Chosen number of discrete states/bins.
    ideal_lagtime : int
        Chosen lagtime (timesteps).

    """

    # Fit function for lag vs ITS (exponential-like)
    def lag_fitting(x, A, B, C):
        return np.log10(np.exp(A*x**B)+C)


    # Fit function for ITS_max vs bin number (tanh convergence)
    def dis_fitting(x, A, B, C):
        return np.tanh(A*x+B) + C

    # Define the trial bin numbers in the range 100-1000
    trial_bins = np.arange(100, 1001, 100)

    # Define the trial lagtimes (timesteps) in the range 10-4760
    trial_lags = np.arange(10, 5010, 250)

    # Find min and max of trajectory for binning
    trj_min = np.min(u_data)
    trj_max = np.max(u_data)

    # ---------- determine ideal bin number ----------

    # Generate an empty list to contain the maxima of the ITS for each bin
    its_m = []

    # Loop over bins
    for b in tqdm(trial_bins, desc='Tuning MM Hyperparameters...'):

        # Create bin edges and discretize trajectory
        bin_edges = np.linspace(trj_min, trj_max, b)
        d_traj = np.digitize(u_data, bin_edges) - 1

        # ITS curve over all trial lagtimes at this bin number, keeping only its maximum
        its = [implied_timescale(d_traj, b, l, bin_edges, uc, bk) for l in trial_lags]
        its_m.append(np.max(its))

    its_m = np.array(its_m)

    # Fit the ITS_max vs bin-count curve
    p0 = [1E-3, 0, 10]
    popt, pcon = curve_fit(dis_fitting, trial_bins, its_m, p0=p0)

    # Generate a smooth curve from the fitted line
    # (note: this extends beyond the largest trial bin number, so values above 1000 are extrapolated)
    fit_x = np.arange(100, 1501, 1)
    fit_y = dis_fitting(fit_x, *popt)

    # Calculate the curvature of this line and the point (after peak curvature) at which
    # this falls below 1% of its maximum. Fall back to the largest bin number if it never does.
    dx = np.gradient(fit_y, fit_x)
    d2x = np.abs(np.gradient(dx, fit_x))
    peak = np.argmax(d2x)
    below = np.where(d2x[peak:] <= 0.01 * d2x[peak])[0]
    ideal_bin_no = fit_x[peak + below[0]] if below.size else fit_x[-1]

    # =======================
    # Plot discretization results
    # =======================
    fig, ax = plt.subplots()
    ax.plot(trial_bins, its_m, lw=3.0)
    ax.axvline(ideal_bin_no, lw=3.0, linestyle='--', color='black', label='Ideal bin number')
    ax.set_xlabel(r'$Number \: of \: Bins$')
    ax.set_ylabel(r'max log($\tau_{relax}$) / $\rm{} Timesteps$')
    ax.legend()
    plt.tight_layout()
    fig.savefig(result_path / 'Discretization.png', dpi=300)

    # =======================
    # Save discretization data to file
    # =======================
    data = np.column_stack((trial_bins, its_m)).astype(np.float64)

    full_header = '@    title "Markov Model Discretization optimization"\n'
    full_header += '@    xaxis  label "n Bins"\n'
    full_header += '@    yaxis  label "Implied Timescale Maxima"\n'
    full_header += '@    s0 legend "Implied Timescale Maxima" '

    np.savetxt(
        result_path / "Discretization_Optimization.xvg",
        data,
        fmt="%.2f",
        delimiter="\t",
        header=full_header,
        comments=''
    )

    # ---------- determine ideal lagtime using ideal_bin_no ----------

    # Calculate bin edges using idealised bin number and discretize umbrella sampling trajectories
    bin_edges = np.linspace(trj_min, trj_max, ideal_bin_no)
    d_traj = np.digitize(u_data, bin_edges) - 1

    # ITS curve over all trial lagtimes at the ideal bin number
    its_a = np.array([implied_timescale(d_traj, ideal_bin_no, l, bin_edges, uc, bk) for l in trial_lags])

    # Fit the data to an exponential curve
    p0 = [1, 0.1, -1]
    popt, pcon = curve_fit(lag_fitting, trial_lags, its_a, maxfev=100000,p0=p0)

    # Evaluate the fitted curve on a fine grid of lagtimes
    sx = np.arange(10, 5000)
    sy = lag_fitting(sx, *popt)

    # Take the second derivative of the smooth fitted curve
    dx = np.gradient(sy, sx)
    d2x = np.abs(np.gradient(dx, sx))

    # The ideal lagtime is taken as the point at which the second derivative falls below 4E-8
    converged = np.where(d2x<=4E-8)[0]
    if converged.size == 0:
        raise ValueError("Implied timescales did not level off within the trial lagtimes (10-5000 frames); "
                         "cannot choose a Markov lagtime. Check sampling or extend trial_lags in mm_tuning.")
    ideal_lagtime = sx[converged[0]]

    # =======================
    # Plot lagtime results
    # =======================
    fig, ax = plt.subplots()
    ax.plot(trial_lags, its_a, lw=3.0, label='Implied Relaxation Timescales')
    ax.axvline(ideal_lagtime, linestyle='--', color='black', lw=3.0, label=r'$\tau_{M}$')
    ax.set_xlabel(r'$Lagtime$ / $\rm{} Timesteps$')
    ax.set_ylabel(r'log($\tau_{relax}$) / $\rm{} Timesteps$')
    ax.legend()
    plt.tight_layout()
    fig.savefig(result_path / "Lagtime_Optimization.png", dpi=300)

    # =======================
    # Save lagtime data to file
    # =======================
    data = np.column_stack((trial_lags, its_a)).astype(np.float64)

    full_header = '@    title "Markov Model Lagtime Optimization"\n'
    full_header += '@    xaxis  label "Lagtime (timesteps)"\n'
    full_header += '@    yaxis  label "Implied Timescale (timesteps)"\n'
    full_header += '@    s0 legend "Implied Timescale" '

    np.savetxt(
        result_path / "Lagtime_Optimization.xvg",
        data,
        fmt="%.2f",
        delimiter="\t",
        header=full_header,
        comments=''
    )
    # Return optimized bin and lagtime numbers
    return ideal_bin_no, ideal_lagtime

def read_log(filepath):
    """
    Reads a log file from an umbrella sampling simulation and extracts:
    - ensemble temperature
    - umbrella center (init)
    - biasing constant (k)

    If the log contains more than one pull coordinate, the values for the last one are returned.

    Parameters:
        filepath (str): Path to the log file.

    Returns:
        temp (float): Ensemble temperature (K).
        uc (float): Umbrella center value (nm).
        k (float): Biasing constant (kJ mol^-1 nm^-2).

    Raises:
        ValueError: If any of the required values cannot be found in the log.
    """

    # Initialize variables with default "not found" values
    temp = -1      # Temperature, will remain -1 if not found
    uc = None      # Umbrella center, will remain None if not found
    k = -1         # Biasing constant, will remain -1 if not found

    # Open the log file for reading
    with open(filepath, 'r') as file:
        # Loop over each line in the file
        for l in file:
            sl = l.split('=')  # Split line into key and value at '='

            # Remove leading/trailing whitespace and check the key
            if sl[0].strip() == 'ensemble-temperature':
                temp = float(sl[1].strip())  # Convert value to float
            elif sl[0].strip() == 'init':
                uc = float(sl[1].strip())    # Convert value to float
            elif sl[0].strip() == 'k':
                k = float(sl[1].strip())     # Convert value to float

    # Check if temperature was found, raise error if not
    if temp == -1:
        raise ValueError(f"Temperature cannot be found in log {filepath}, please check for malformed logfile!")

    # Check if umbrella center or biasing constant were found, raise error if not
    if uc is None or k == -1:
        raise ValueError(f"Umbrella center / Biasing Constant cannot be found in log {filepath}, please check this is an umbrella sampling logfile!")

    # Return the extracted values
    return temp, uc, k

def read_px(filename):

    """
    Read a pullx / px file extracting the coordinate column.

    The code assumes the first 17 header lines are metadata, and that the
    coordinate of interest is the second column. Adjust skiprows/index if format differs.

    Parameters
    -----------
    filename (str): Path to the pullx file.

    Returns
    -------
    data : 1D numpy array of coordinate values (nm)

    """

    data = np.loadtxt(filename, skiprows=17)[:,1]

    return data

def read_ts(filename):

    """
    Read a pullx / px file and return the time between frames.

    Parameters
    -----------
    filename (str): Path to the pullx file.

    Returns
    -------
    ts : Time between pullx frames (ps)

    """

    # Only the first two time values are needed
    t = np.loadtxt(filename, skiprows=17, max_rows=2)[:,0]

    ts = t[1] - t[0]

    return ts

def calc_perm(pmf, diffusion, bins, temp):

    """
    Calculate permeability from PMF and position-dependent diffusion using the
    inhomogeneous solubility-diffusion model:

        1 / kP = integral of exp(G(z) / RT) / D(z) dz

    Parameters
    -----------
    pmf (1D array): 1D array of PMF energies (kJ/mol)
    diffusion (1D array): 1D array of position dependent diffusion coefficients (cm^2/s)
    bins (1D array): 1D array of discretized bins (Angstrom)
    temp (float): Simulation temperature (Kelvin)

    Returns
    -------
    permeability (float) : Overall kP value (cm/s)

    """

    # Convert bins from Angstrom to cm
    bins = bins*10**-8

    # Calculate the local resistance exp(G/RT) / D
    pmf_diff = np.exp(pmf/(R*temp))/diffusion

    # Total resistance is the integral across the membrane; permeability is its inverse
    permeability = 1 / trapezoid(pmf_diff, bins)

    return permeability

def convergence(data, uc, bk, bins, lag, temp, eqc, results_dir):
    """
    Plot progressive PMFs from increasing portions of the trajectory to inspect convergence.

    Parameters
    ----------
    data : ndarray (ntraj, nsteps)
        Raw trajectory coordinate(s).
    uc, bk : arrays
        Umbrella centers and biasing constants.
    bins : array
        Bin edges used for discretization.
    lag : int
        Chosen lagtime for Markov model.
    temp : float
        Temperature (K).
    eqc : int
        Equilibration cut (number of frames to discard).
    results_dir : Path
        Directory to save convergence plot.

    Produces a single figure (Convergence.png) showing PMF estimates from
    increasing amounts of data (colormap indicates timestep used).
    """
    # Discretize the trajectories
    d_data = np.digitize(data, bins) - 1
    d_data[d_data<=-1] = 0 # Underflow guard

    t_len = d_data.shape[1] # Get the length of the trajectories


    # Only keep bins that overlap the umbrella center range (valid sampling)
    valid_mask = (bins >= np.min(uc)) & (bins <= np.max(uc))
    valid_bins = bins[valid_mask]

    # Construct mirrored full_bins for symmetric plotting: (-z...z)
    full_bins = np.concatenate((-valid_bins[::-1], valid_bins))

    # PMFs are computed from the equilibration cut up to every 10000th frame
    t_values = list(range(eqc+10000, t_len, 10000))

    # Create colormap
    cmap = cm.jet
    norm = colors.Normalize(vmin=min(t_values), vmax=max(t_values))
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()

    for t in tqdm(t_values, desc='Calculating convergence...'):

        # Use trajectory data from eqc to t
        c_run = d_data[:, eqc:t]

        # Build transition counts and DHAM Markov model
        transition_matrix, transition_sums = gcm(c_run, len(bins), lag)
        mm = gen_mm_numba(transition_matrix, transition_sums, bins, uc, bk)

        # Convert to PMF (kJ/mol), restrict to valid bins, set the minimum to zero, then mirror
        pmf = mm2pmf(mm) * temp * R
        full_pmf = mirror(zero_min(pmf[valid_mask]))

        # Plot this (lightly smoothed) PMF curve colored by timestep
        ax.plot(full_bins, smooth(full_pmf), color=scalar_map.to_rgba(t), lw=3.0)

    cbar = plt.colorbar(scalar_map, ax=ax)
    cbar.set_label("Timestep")
    ax.set_xlabel(r'$z$ / $\rm{} \AA$')
    ax.set_ylabel(r'$\Delta{G}$ / $\rm{} kJ\,mol^{-1}$')

    fig.savefig(results_dir / "Convergence.png", dpi=300)

def mm2pmf(mm):
    """
    Calculate PMF from Markov model

    The stationary distribution is the eigenvector of mm^T with eigenvalue 1;
    the PMF is then -ln(p), shifted so its minimum (over all bins) is zero.
    Callers that restrict the PMF to a sub-range should re-reference it with zero_min.

    Parameters
    -----------
    mm (2D array): 2D Markov transition probability matrix

    Returns
    -------
    PMF (1D array) : 1D array of PMF values (kT)

    """

    # Compute the leading eigenvalues and eigenvectors of the transposed matrix
    eigval, eigvec = top_eigs(mm)

    # Select the eigenvector corresponding to the largest eigenvalue
    prob = eigvec[:, np.where(eigval == np.max(eigval))[0][0]]

    # Normalize eigenvector so it sums to 1 → valid probability distribution
    # (this also removes the arbitrary sign/phase the eigensolver returns)
    prob = prob / np.sum(prob)

    # Convert probability distribution to potential of mean force (PMF)
    pmf = -np.log(prob.real.astype(np.float64) + 1e-12)

    # Shift PMF so the minimum energy level is zero (relative scale)
    pmf -= np.min(pmf)

    return pmf

@njit(parallel=True)
def gen_mm_numba(transition_matrix, transition_total, bin_edges, uc, bias):
    """
    Converts and reweights transition count matrices into a single unbiased Markov transition
    probability matrix using DHAM:

        M_ij = N_ij / sum_k [ n_i^k * exp(-(u_j^k - u_i^k) / 2) ]

    where N_ij is the total number of i -> j transitions, n_i^k the number of transitions out of
    state i in simulation k, and u^k the bias potential (kT) of simulation k. Rows are then
    normalized so each sums to one.

    Parameters
    ----------
    transition_matrix : ndarray
        Matrix that contains the transition counts
    transition_total : ndarray
        Array of the total number of transitions from each state in each simulation
    bin_edges : ndarray
        Array containing the bin edges used for discretization
    uc : ndarray
        Array of the bias potential centers used
    bias : ndarray
        Array of bias force constants used in each simulation (kT / Angstrom^2)


    Returns
    -------
    mm : ndarray
        The re-weighted markov transition matrix
    """

    M = transition_matrix.shape[0]
    K = transition_total.shape[0]

    # Bin centers (the bias is evaluated at the center of each bin)
    mm = np.zeros((M, M))
    bin_spacing = bin_edges[1] - bin_edges[0]
    centers = np.empty(M)
    for m in range(M):
        centers[m] = bin_edges[m] + bin_spacing / 2.0

    # Precompute the harmonic bias u[k,m] = 0.5 * k_k * (z_m - uc_k)^2 for every simulation and bin
    u = np.empty((K, M))
    for k in range(K):
        for m in range(M):
            diff = uc[k] - centers[m]
            u[k, m] = 0.5 * bias[k] * diff * diff

    # build denom incrementally (no huge KxMxM array); rows are independent so run in parallel
    for i in prange(M):
        for j in range(M):
            if transition_matrix[i, j] > 0.0:
                denom = 0.0
                for k in range(K):
                    if transition_total[k, i] > 0.0:
                        denom += transition_total[k, i] * np.exp((u[k, i] - u[k, j]) / 2.0)
                if denom > 0.0:
                    mm[i, j] = transition_matrix[i, j] / denom
                else:
                    mm[i, j] = 0.0
            else:
                mm[i, j] = 0.0

    # normalize rows safely
    for i in prange(M):
        rowsum = 0.0
        for j in range(M):
            rowsum += mm[i, j]
        if rowsum > 0.0:
            for j in range(M):
                mm[i, j] /= rowsum
        else:
            # leave row as zeros (could optionally set self-transition = 1)
            pass

    return mm

@njit
def gcm(disc_data, nbins, lagt):
    """
    Compute transition count matrices from discretized trajectories.

    Parameters
    ----------
    disc_data : ndarray, shape (ntraj, nsteps)
        Array of discretized trajectories.
    nbins : int
        Number of discrete states (bins).
    lagt : int
        Lag time between current and next state when counting transitions.

    Returns
    -------
    transition_matrix : ndarray, shape (nbins, nbins)
        Global transition count matrix (summed over all trajectories).
    transition_total : ndarray, shape (ntraj, nbins)
        For each trajectory, the number of transitions from each state.
    """

    # Number of trajectories and number of time steps per trajectory
    ntraj, nsteps = disc_data.shape

    # Global transition matrix (sum over all trajectories)
    transition_matrix = np.zeros((nbins, nbins), dtype=np.int64)

    # Per-trajectory totals: how many transitions leave each state
    transition_total = np.zeros((ntraj, nbins), dtype=np.int64)

    # Accumulate both directly, rather than building an ntraj x nbins x nbins
    # per-trajectory count array and summing it afterwards
    for k in range(ntraj):
        # Loop over time steps, starting from lag time
        for i in range(lagt, nsteps):

            # State at time t
            current_state = disc_data[k, i-lagt]

            # State at time t + lag
            transition_state = disc_data[k, i]

            # Increment the global transition count and this trajectory's outgoing total
            transition_matrix[current_state, transition_state] += 1
            transition_total[k, current_state] += 1

    return transition_matrix, transition_total

def calc_c2(mm, bins):
    """
    Second Kramers-Moyal coefficient (half the variance of the displacement) for every state.

    With cn = sum_j (x_i - x_j)^n * M_ij, returns (c2 - c1^2) / 2 for each state i.
    Its gradient with respect to lagtime gives the local diffusion coefficient.

    Parameters
    ----------
    mm : ndarray, shape (M, M)
        Markov transition probability matrix at a given lagtime.
    bins : ndarray, shape (M,)
        Bin positions (Angstrom).

    Returns
    -------
    c2 : ndarray, shape (M,)
        Half the displacement variance from each state (Angstrom^2).
    """

    # Vectorized over all (i, j) pairs: dz[i, j] = bins[i] - bins[j]
    dz = bins[:, None] - bins[None, :]

    # Kramers-Moyal variance: c2 - c1^2, where cn = sum_j (x_i - x_j)^n * M_ij
    c2 = np.sum(dz**2 * mm, axis=1) - np.sum(dz * mm, axis=1)**2

    return c2/2

def km_diffusion(dtraj, lagtime, bins, uc, bk, ts):

    """
    Calculates the position dependent diffusion coefficient using the Kramers Moyal expansion as described:
    Position-Dependent Diffusion from Biased Simulations and Markov State Model Analysis François Sicard, Vladimir Koskin, Alessia Annibale, and Edina Rosta Journal of Chemical Theory and Computation 2021 17 (4), 2022-2033 DOI: 10.1021/acs.jctc.0c01151

    A Markov model is built at every lagtime from the Markov lagtime up to 30000 frames
    (in steps of 50), the second Kramers-Moyal coefficient is computed for each state, and
    D(z) is the slope of that coefficient against lagtime.

    Parameters
    ----------
    dtraj : ndarray
        Array containing the discretized umbrella sampling trajectories
    lagtime : int
        Markov timescale as determined from the implied relaxation timescale (frames)
    bins : ndarray
        Bin edges used to discretize trajectories
    uc : ndarray
        Array of umbrella centers used in umbrella sampling
    bk : ndarray
        Array containing bias constants used in each umbrella sampling trajectory
    ts : float
        Time between umbrella sampling trajectory frames (s)

    Returns
    -------
    diffusion_coefficients : ndarray
        Diffusion coefficient for each bin (cm^2/s)
    """

    # Lagtimes (frames) at which the Kramers-Moyal coefficient is evaluated
    lagtimes = np.arange(lagtime, 30001, 50)

    km_c2m = []

    for lt in tqdm(lagtimes, desc='Calculating Kramers-Moyal coefficients...'):

        transition_matrix, transition_sums = gcm(dtraj, len(bins), lt) # Construct transition matrix at current lagtime

        mm = gen_mm_numba(transition_matrix, transition_sums, bins, uc, bk) # Construct the resulting mm

        km_c2m.append(calc_c2(mm, bins))

    # Shape (n_lagtimes, n_bins)
    km_c2m = np.array(km_c2m)

    # Perform a linear regression to extract the gradient of each c2 (all states fitted at once)
    slopes = np.polyfit(lagtimes, km_c2m, 1)[0]
    diffusion_coefficients = np.abs(slopes)  # slope of c2 vs lagtime (Angstrom^2 / frame)

    # Convert Angstrom^2 / frame -> cm^2 / s
    diffusion_coefficients = (diffusion_coefficients * 1E-16) / ts

    return diffusion_coefficients

def pmf_error(dtraj, lagtime, bins, uc, bk,temp):
    """
    Estimate the PMF uncertainty by computing a PMF from each of N_ERROR_BLOCKS consecutive,
    non-overlapping blocks of the trajectories and taking the standard deviation between them.

    Parameters
    ----------
    dtraj : ndarray (ntraj, nsteps)
        Discretized (equilibrated) trajectories.
    lagtime : int
        Markov lagtime (frames).
    bins, uc, bk : arrays
        Bin edges, umbrella centers and bias constants.
    temp : float
        Temperature (K).

    Returns
    -------
    error : ndarray (n_bins,)
        Standard deviation of the block PMFs in each bin (kJ/mol).
    pmfs : ndarray (N_ERROR_BLOCKS, n_bins)
        The individual block PMFs (kJ/mol).
    """

    n_bins = len(bins)

    pmfs = []

    # Split the trajectories into consecutive, non-overlapping blocks
    for block in np.array_split(dtraj, N_ERROR_BLOCKS, axis=1):

        # DHAM Markov model and PMF (kJ/mol) for this block
        tm, ts = gcm(block, n_bins, lagtime)
        mm = gen_mm_numba(tm, ts, bins, uc, bk)
        pmfs.append(mm2pmf(mm)*temp*R)

    pmfs = np.vstack(pmfs)

    error = np.std(pmfs, axis=0)

    return(error, pmfs)

def diffusion_error(dtraj, lag, bins, uc, bk, ts):
    """
    Estimate the diffusion uncertainty by computing D(z) from each of N_ERROR_BLOCKS consecutive,
    non-overlapping blocks of the trajectories and taking the standard deviation between them.

    Each block must be considerably longer than the 30000-frame maximum lagtime used in km_diffusion.

    Parameters
    ----------
    dtraj : ndarray (ntraj, nsteps)
        Discretized (equilibrated) trajectories.
    lag : int
        Markov lagtime (frames).
    bins, uc, bk : arrays
        Bin edges, umbrella centers and bias constants.
    ts : float
        Time between trajectory frames (s).

    Returns
    -------
    error : ndarray (n_bins,)
        Standard deviation of the block diffusion coefficients in each bin (cm^2/s).
    diffs : ndarray (N_ERROR_BLOCKS, n_bins)
        The individual block diffusion profiles (cm^2/s).
    """

    # Split the trajectories into consecutive, non-overlapping blocks
    blocks = np.array_split(dtraj, N_ERROR_BLOCKS, axis=1)

    diffs = np.vstack([km_diffusion(block, lag, bins, uc, bk, ts) for block in blocks])

    error = np.std(diffs, axis=0)

    return (error, diffs)

def dham(il, ix, watdiff, watdiff_er):

    """
    Compute skin permeation coefficient (kP) from umbrella sampling simulation data.
    From umbrella sampling simulations of small molecules within a model stratum corneum membrane,
    we calculate a kP value for transport through the stratum corneum.

    If the diffusion coefficient for the small molecule in water is also provided, we estimate the kP
    associated with travelling through the shunt routes (sweat glands and hair follicles) using a simple
    mathematical model from:
    Mitragotri S. (2003). Modeling skin permeability to hydrophilic and hydrophobic solutes based on four permeation pathways. Journal of controlled release : official journal of the Controlled Release Society, 86(1), 69–92. https://doi.org/10.1016/s0168-3659(02)00321-8

    This model has been modified to treat sweat glands and hair follicles as separate routes, scaling the aqueous
    diffusion coefficient in each by the ratio of the viscosity of water to that of the fluid filling the route
    (sweat or sebum). Area fractions and the sweat gland path length are taken from Mitragotri (2003); the other values from:

    Sweat viscosity:
    Emrich, H., Stoll, E., Friolet, B. et al. Sweat Composition in Relation to Rate of Sweating in Patients with Cystic Fibrosis of the Pancreas. Pediatr Res 2, 464–478 (1968). https://doi.org/10.1203/00006450-196811000-00004

    Sebum viscosity:
    Butcher, E. O. and Coonin, A. The Physical Properties of Human Sebum. Journal of Investigative Dermatology, 12(4), 249–254 (1949). https://doi.org/10.1038/jid.1949.37

    Water viscosity (303.15 K):
    Coe, J. and Godfrey, T. Viscosity of Water. Journal of Applied Physics, 15, 625–626 (1944). https://doi.org/10.1063/1.1707481

    Hair follicle (infundibulum) path length:
    Kasting, G. B., Miller, M. A., LaCount, T. D. and Jaworska, J. A composite model for the transport of hydrophilic and lipophilic compounds across the skin: steady-state behavior. Journal of Pharmaceutical Sciences, 108(1), 337–349 (2019).

    All results are written to ./DHAM_Results.

    Parameters
    ----------
    il : string
        Name of .dat file containing the ordered names of GROMACS umbrella sampling log files
    ix : string
        Name of .dat file containing the ordered names of GROMACS umbrella sampling pullx files
    watdiff : float or None
        The diffusion coefficient of the sampled small molecule in pure water (cm^2 / s)
    watdiff_er : float or None
        The uncertainty in watdiff (cm^2 / s)
    """

    # Make directory to store results
    results_dir = Path("DHAM_Results")
    results_dir.mkdir(exist_ok=True)

    # =======================
    # Read simulation data
    # =======================

    with open(il, 'r') as f:
        log_files = [line.rstrip() for line in f]

    with open(ix, 'r') as f:
        px_files = [line.rstrip() for line in f]

    # Extract simulation trajectories, shape (n_windows, n_frames)
    u_data = np.array([read_px(f) for f in tqdm(px_files, desc='Extracting trajectories ...')], dtype=np.float64)

    # Take the absolute of the reaction coordinate and convert nm to angstrom
    u_data = np.abs(u_data) * 10

    # Extract timestep
    ts = read_ts(px_files[0])

    # Convert timestep from ps to seconds
    ts = ts/(10**12)

    # Load in the US log files to extract umbrella centers, biasing potentials and the temperature
    log_data = np.array([read_log(f) for f in tqdm(log_files, desc='Extracting umbrella centers, biasing constants and temperature...')])

    # Assume all simulations are the same temperature (AS THEY SHOULD BE) so only extract one
    temp = log_data[0,0]

    # Extract umbrella centers
    uc = log_data[:, 1]

    # Take the absolute of the umbrella centers and convert nm to angstrom
    uc = np.abs(uc) * 10

    # Extract biasing potentials
    bk = log_data[:, 2]

    # Convert bias constants from kJ mol^-1 nm^-2 to kT Angstrom^-2
    bk = bk/(temp*R*100)

    # =======================
    # Equilibration and Markov model tuning
    # =======================

    # Calculate how much of the initial trajectory should be discarded
    equilibration_point = autocut(u_data)

    # Remove equilibration portion from data
    equilibrated = u_data[:, equilibration_point:]

    # Determine the ideal discretization and lagtime
    n_bins, lag =  mm_tuning(equilibrated, uc, bk, results_dir)

    # Extract minimum and maximum of the trajectories
    trj_min = np.min(equilibrated)
    trj_max = np.max(equilibrated)

    # Calculate bins
    bins = np.linspace(trj_min, trj_max, n_bins)

    # Discretize trajectories
    discretized_trajectory = np.digitize(equilibrated, bins) - 1

    # Underflow guard
    discretized_trajectory[discretized_trajectory<=-1] = 0

    # =======================
    # Sampling checks
    # =======================

    # Populate umbrella sampling histograms (frames in each bin, all windows combined)
    HCount = np.bincount(discretized_trajectory.ravel(), minlength=len(bins)).astype(np.float64)

    # Plot histograms
    fig, ax = plt.subplots()
    ax.fill_between(bins, HCount, np.zeros(len(bins)))
    ax.set_xlabel(r'$z$ / $\rm{}\AA$')
    ax.set_ylabel(r'$Histogram \: Count$')

    # Save histograms
    fig.savefig(results_dir / "Histograms.png", dpi=300)

    # Compute Convergence
    convergence(u_data, uc, bk, bins, lag, temp, equilibration_point, results_dir)

    # =======================
    # Free energy profile
    # =======================

    # Using Optimized Markov parameters, construct MM and extract free energy estimate

    transition_matrix, transition_sums = gcm(discretized_trajectory, n_bins, lag) # Construct transition matrix at the Markov lagtime

    mm = gen_mm_numba(transition_matrix, transition_sums, bins, uc, bk) # Construct the resulting mm

    pmf = mm2pmf(mm) * temp * R # Extract free energy estimate and convert to kJ/mol

    # Calculate a pmf for four consecutive blocks of the data (used for the error estimate)
    _, pepmfs = pmf_error(discretized_trajectory,lag,bins,uc,bk,temp)

    # Limit PMF to within the range of the min and max umbrella centers to remove sampling limit errors
    valid_indices = (bins >= np.min(uc)) & (bins <= np.max(uc))

    # Reference each PMF so its minimum within the valid range is zero
    # (mm2pmf zeroes the minimum over all bins, which may lie outside this range)
    valid_pmf = zero_min(pmf[valid_indices])
    valid_pe_pmfs = zero_min(pepmfs[:, valid_indices])

    # PMF error: standard deviation between the re-referenced block PMFs
    valid_pmf_error = np.std(valid_pe_pmfs, axis=0)

    # Extract bins within valid range and mirror everything to cover the whole membrane
    valid_bins = bins[valid_indices]
    full_pmf = mirror(valid_pmf)
    full_pmf_error = mirror(valid_pmf_error)
    full_pe_pmfs = mirror(valid_pe_pmfs)

    # Mirror bins (negative z on the left)
    full_bin = np.concatenate((-valid_bins[::-1], valid_bins))

    # =======================
    # Diffusion profile
    # =======================

    # Calculate position dependent diffusion coefficients
    diffusion = km_diffusion(discretized_trajectory, lag, bins, uc, bk, ts)

    diffusion_er, pediffs = diffusion_error(discretized_trajectory, lag, bins, uc, bk, ts)

    # Restrict to the valid range and mirror, as for the PMF
    full_diffusion = mirror(diffusion[valid_indices])
    full_diff_er = mirror(diffusion_er[valid_indices])
    full_pe_diffs = mirror(pediffs[:, valid_indices])

    # =======================
    # Permeability
    # =======================

    permeability = calc_perm(full_pmf, full_diffusion, full_bin, temp)

    # Permeability error: standard deviation of the kP values from each block's PMF and D(z)
    block_perms = [calc_perm(full_pe_pmfs[b], full_pe_diffs[b], full_bin, temp) for b in range(N_ERROR_BLOCKS)]
    perm_er = np.std(block_perms)

    if watdiff:

        # Shunt-route permeability (Mitragotri 2003): kP = epsilon * D_shunt / L,
        # modified to explicitly calculate separate contributions from sweat glands and hair follicles,
        # with D_shunt = D_Water * (V_Water / V_fluid) correcting for the viscosity of the fluid filling each route:
        #
        # kP_Shunt = kP_Sweat + kP_Hair
        #          = ((epsilon_Sweat * D_Water * V_Water)/(L_Sweat * V_Sweat)) + ((epsilon_Hair * D_Water * V_Water)/(L_Hair * V_Hair))
        #
        # D_Water       = provided small molecule aqueous diffusion coefficient (cm^2/s)
        # V_Water       = 0.78 mPa s, water at 303.15 K (Coe & Godfrey 1944)
        # epsilon_Sweat = 1E-4, sweat gland area fraction (Mitragotri 2003)
        # L_Sweat       = 500E-4 cm, sweat gland path length (Mitragotri 2003)
        # V_Sweat       = 0.92 mPa s, sweat (Emrich et al. 1968)
        # epsilon_Hair  = 2E-3, hair follicle area fraction (Mitragotri 2003)
        # L_Hair        = 410E-4 cm, follicular infundibulum path length (Kasting et al. 2019)
        # V_Hair        = 85.97 mPa s, sebum (Butcher & Coonin 1949)
        #
        # The uncertainty in D_Water is propagated through the same expression.

        kP_shunt = ((watdiff*1E-4*0.78)/(0.92*500E-4)) + ((watdiff*2E-3*0.78)/(85.97*410E-4))
        kP_shunt_er = ((watdiff_er*1E-4*0.78)/(0.92*500E-4)) + ((watdiff_er*2E-3*0.78)/(85.97*410E-4))

        # Stratum corneum and shunt routes act in parallel, so their permeabilities add
        permeability_shunt_correction = permeability + kP_shunt


        with open(results_dir / "kp.txt", "w") as f:
            f.write(f"kP_StCor (cm/s) = {permeability} +/- {perm_er}\n")
            f.write(f"log(kP_StCor) (cm/s) = {np.log10(permeability)} +/- {np.log10(perm_er)} \n")

            f.write(f"kP_Shunt (cm/s) = {kP_shunt} +/- {kP_shunt_er} \n")
            f.write(f"log(kP_Shunt) (cm/s) = {np.log10(kP_shunt)} +/- {np.log10(kP_shunt_er)} \n")

            f.write(f"kP_Total (cm/s) = {permeability_shunt_correction} +/- {perm_er+kP_shunt_er} \n")
            f.write(f"log(kP_Total) (cm/s) = {np.log10(permeability_shunt_correction)} +/- {np.log10(perm_er+kP_shunt_er)}\n")

    else:

        with open(results_dir / "kp.txt", "w") as f:
            f.write(f"kP_StCor (cm/s) = {permeability} +/- {perm_er} \n")
            f.write(f"log(kP_StCor) (cm/s) = {np.log10(permeability)} +/- {np.log10(perm_er)}\n")

    # =======================
    # Plot and save PMF
    # =======================

    # Add a short smoothing filter to the free energy profile to smooth any bumps arising naturally through DHAM
    # (smoothing is for the plot only; PMF.xvg and kP use the unsmoothed PMF)
    smooth_pmf = smooth(full_pmf)

    fig, ax = plt.subplots()

    ax.plot(full_bin, smooth_pmf, lw=3.0, label='PMF')
    ax.fill_between(full_bin, smooth_pmf - full_pmf_error, smooth_pmf + full_pmf_error, color='red', alpha=0.2, label='Error')


    ax.set_xlabel(r'$z$ / $\rm{} \AA$')
    ax.set_ylabel(r'$\Delta{G}$ / $\rm{} kJ\,mol^{-1}$')

    ax.legend()

    plt.savefig(results_dir / "PMF.png", dpi = 300)

    data = np.column_stack((full_bin, full_pmf, full_pmf_error))
    data = data.astype(np.float64)

    # === Prepare Header for Output File ===
    full_header = '@    title "Free Energy"\n'
    full_header += '@    xaxis  label "z / A"\n'
    full_header += '@    yaxis  label "Free Energy / kJ mol^-1"\n'
    full_header += '@    s0 legend "Free Energy"\n'
    full_header += '@    s1 legend "PMF Error"\n'

    np.savetxt(
    results_dir / "PMF.xvg",
    data,
    fmt="%.2f",
    delimiter="\t",
    header=full_header,
    comments=''
    )

    # =======================
    # Plot and save diffusion
    # =======================

    fig, ax = plt.subplots()

    ax.plot(full_bin, full_diffusion, lw=3.0, label='Diffusion')

    ax.fill_between(full_bin, full_diffusion-full_diff_er, full_diffusion+full_diff_er, color='red', alpha=0.2, label='Error')

    ax.set_xlabel(r'$z$ / $\rm{} \AA$')
    ax.set_ylabel(r'$Diffusion$ / $\rm{} cm^{2} \; s^{-1}$')

    ax.legend()

    plt.savefig(results_dir / "Smoothed_Diffusion.png", dpi = 300)

    data = np.column_stack((full_bin, full_diffusion, full_diff_er))

    data = data.astype(np.float64)

    # === Prepare Header for Output File ===
    full_header = '@    title "Diffusion"' + '\n'
    full_header += '@    xaxis  label "z / A"' + '\n'
    full_header += '@    yaxis  label "Diffusion / cm^2 s^-1"' + '\n'
    full_header += '@    s0 legend "Diffusion"' + '\n'
    full_header += '@    s1 legend "Diffusion Error" '

    # === Save Results to File ===
    np.savetxt(
        results_dir / "Diffusion.xvg",  # Output file path
        data,                        # 3 columns: z, diffusion, error
        fmt="%.2e",                  # Scientific notation, 2 decimal places
        delimiter="\t",              # Tab-separated values
        header=full_header,          # Header describing the file
        comments=''                  # No comment character before header lines
    )

    return

if __name__ == '__main__':

    import argparse

    whoami = """
Compute the skin permeation coefficient (kP) of a small molecule from GROMACS umbrella sampling
simulations across a model stratum corneum membrane.

The free energy profile is obtained with the Dynamic Histogram Analysis Method (DHAM), position-
dependent diffusion from the Kramers-Moyal expansion of the DHAM Markov model, and kP from the
inhomogeneous solubility-diffusion model. If the molecule's diffusion coefficient in water is given,
a shunt-route (sweat gland / hair follicle) permeability is also estimated and added.

Results (plots, .xvg data and kp.txt) are written to ./DHAM_Results.

Example:
    python dham.py -il logs.dat -ix pullx.dat -watdiff 1E-5 1E-6
"""
    parser = argparse.ArgumentParser(description = whoami, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-il', type=str, nargs=1, required= True, help=' .dat file containing ordered names of GROMACS log files')
    parser.add_argument('-ix', type=str, nargs=1, required= True, help=' .dat file containing ordered names of GROMACS pullx files')
    parser.add_argument('-watdiff', type=float, nargs=2, metavar=('DIFFUSION', 'UNCERTAINTY'), default=None, help=('Provide the diffusion coefficient of your small molecule in water ''(cm^2/s, scientific notation e.g. 1E-6) followed by its uncertainty.'))

    args = parser.parse_args()

    if args.watdiff is not None:
        diffusion, uncertainty = args.watdiff
    else:
        diffusion = None
        uncertainty = None

    dham(
        args.il[0],
        args.ix[0],
        diffusion,
        uncertainty
    )
