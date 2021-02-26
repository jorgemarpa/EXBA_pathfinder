""" Collection of utility functions"""

import numpy as np
import functools

from scipy import sparse
from patsy import dmatrix
from tqdm.notebook import tqdm
import pyia
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.timeseries import BoxLeastSquares


@functools.lru_cache()
def get_gaia_sources(ras, decs, rads, magnitude_limit=18, epoch=2020, gaia="dr2"):
    """
    Will find gaia sources using a TAP query, accounting for proper motions.

    Inputs have be hashable, e.g. tuples

    Parameters
    ----------
    ras : tuple
        Tuple with right ascension coordinates to be queried
        shape nsources
    ras : tuple
        Tuple with declination coordinates to be queried
        shape nsources
    ras : tuple
        Tuple with radius query
        shape nsources
    magnitude_limit : int
        Limiting magnitued for query

    Returns
    -------
    Pandas DatFrame with number of result sources (rows) and Gaia columns

    """
    wheres = [
        f"""1=CONTAINS(
                  POINT('ICRS',ra,dec),
                  CIRCLE('ICRS',{ra},{dec},{rad}))"""
        for ra, dec, rad in zip(ras, decs, rads)
    ]

    where = """\n\tOR """.join(wheres)
    gd = pyia.GaiaData.from_query(
        f"""SELECT solution_id, designation, source_id, random_index, ref_epoch,
        coord1(prop) AS ra, ra_error, coord2(prop) AS dec, dec_error, parallax,
        parallax_error, parallax_over_error, pmra, pmra_error, pmdec, pmdec_error,
        ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr,
        dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr,
        pmra_pmdec_corr, astrometric_n_obs_al, astrometric_n_obs_ac,
        astrometric_n_good_obs_al, astrometric_n_bad_obs_al, astrometric_gof_al,
        astrometric_chi2_al, astrometric_excess_noise, astrometric_excess_noise_sig,
        astrometric_params_solved, astrometric_primary_flag, astrometric_weight_al,
        astrometric_pseudo_colour, astrometric_pseudo_colour_error,
        mean_varpi_factor_al, astrometric_matched_observations,
        visibility_periods_used, astrometric_sigma5d_max, frame_rotator_object_type,
        matched_observations, duplicated_source, phot_g_n_obs, phot_g_mean_flux,
        phot_g_mean_flux_error, phot_g_mean_flux_over_error, phot_g_mean_mag,
        phot_bp_n_obs, phot_bp_mean_flux, phot_bp_mean_flux_error,
        phot_bp_mean_flux_over_error, phot_bp_mean_mag, phot_rp_n_obs,
        phot_rp_mean_flux, phot_rp_mean_flux_error, phot_rp_mean_flux_over_error,
        phot_rp_mean_mag, phot_bp_rp_excess_factor, phot_proc_mode, bp_rp, bp_g, g_rp,
        radial_velocity, radial_velocity_error, rv_nb_transits, rv_template_teff,
        rv_template_logg, rv_template_fe_h, phot_variable_flag, l, b, ecl_lon, ecl_lat,
        priam_flags, teff_val, teff_percentile_lower, teff_percentile_upper, a_g_val,
        a_g_percentile_lower, a_g_percentile_upper, e_bp_min_rp_val,
        e_bp_min_rp_percentile_lower, e_bp_min_rp_percentile_upper, flame_flags,
        radius_val, radius_percentile_lower, radius_percentile_upper, lum_val,
        lum_percentile_lower, lum_percentile_upper, datalink_url, epoch_photometry_url,
        ra as ra_gaia, dec as dec_gaia FROM (
 SELECT *,
 EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, 0, ref_epoch, {epoch}) AS prop
 FROM gaiadr2.gaia_source
 WHERE {where}
)  AS subquery
WHERE phot_g_mean_mag<={magnitude_limit}

"""
    )
    return gd.data.to_pandas()


def _make_A(phi, r, cut_r=5):
    """ Make spline design matrix in polar coordinates """
    phi_spline = sparse.csr_matrix(wrapped_spline(phi, order=3, nknots=6).T)
    r_knots = np.linspace(0.25 ** 0.5, 5 ** 0.5, 8) ** 2
    r_spline = sparse.csr_matrix(
        np.asarray(
            dmatrix(
                "bs(x, knots=knots, degree=3, include_intercept=True)",
                {"x": list(r), "knots": r_knots},
            )
        )
    )
    X = sparse.hstack(
        [phi_spline.multiply(r_spline[:, idx]) for idx in range(r_spline.shape[1])],
        format="csr",
    )
    cut = np.arange(phi_spline.shape[1] * 1, phi_spline.shape[1] * cut_r)
    a = list(set(np.arange(X.shape[1])) - set(cut))
    X1 = sparse.hstack(
        [X[:, a], r_spline[:, 1:cut_r], sparse.csr_matrix(np.ones(X.shape[0])).T],
        format="csr",
    )
    return X1


def make_A_edges(r, f, type="cuadratic"):
    if type == "linear":
        A = np.vstack([r ** 0, r, f]).T
    elif type == "cuadratic":
        A = np.vstack([r ** 0, r, r ** 2, f]).T
    elif type == "cubic":
        A = np.vstack([r ** 0, r, r ** 2, r ** 3, f]).T
    elif type == "exp":
        A = np.vstack([r ** 0, np.exp(-r), f]).T
    elif type == "inverse":
        A = np.vstack([r ** 0, 1 / r, f]).T
    else:
        print("Wrong desing matrix basis type")
    return A


def wrapped_spline(input_vector, order=2, nknots=10):
    """
    Creates a vector of folded-spline basis according to the input data. This is meant
    to be used to build the basis vectors for periodic data, like the angle in polar
    coordinates.

    Parameters
    ----------
    input_vector : numpy.ndarray
        Input data to create basis, angle values MUST BE BETWEEN -PI and PI.
    order : int
        Order of the spline basis
    nknots : int
         Number of knots for the splines

    Returns
    -------
    folded_basis : numpy.ndarray
        Array of folded-spline basis
    """

    if not ((input_vector > -np.pi) & (input_vector < np.pi)).all():
        raise ValueError("Must be between -pi and pi")
    x = np.copy(input_vector)
    x1 = np.hstack([x, x + np.pi * 2])
    nt = (nknots * 2) + 1

    t = np.linspace(-np.pi, 3 * np.pi, nt)
    dt = np.median(np.diff(t))
    # Zeroth order basis
    basis = np.asarray(
        [
            ((x1 >= t[idx]) & (x1 < t[idx + 1])).astype(float)
            for idx in range(len(t) - 1)
        ]
    )
    # Higher order basis
    for order in np.arange(1, 4):
        basis_1 = []
        for idx in range(len(t) - 1):
            a = ((x1 - t[idx]) / (dt * order)) * basis[idx]

            if ((idx + order + 1)) < (nt - 1):
                b = (-(x1 - t[(idx + order + 1)]) / (dt * order)) * basis[
                    (idx + 1) % (nt - 1)
                ]
            else:
                b = np.zeros(len(x1))
            basis_1.append(a + b)
        basis = np.vstack(basis_1)

    folded_basis = np.copy(basis)[: nt // 2, : len(x)]
    for idx in np.arange(-order, 0):
        folded_basis[idx, :] += np.copy(basis)[nt // 2 + idx, len(x) :]
    return folded_basis


def _bootstrap_max(t, y, dy, pmin, pmax, ffac, random_seed, n_bootstrap=1000):
    """Generate a sequence of bootstrap estimates of the max"""

    rng = np.random.RandomState(random_seed)
    power_max = []
    for _ in range(n_bootstrap):
        s = rng.randint(0, len(y), len(y))  # sample with replacement
        bls_boot = BoxLeastSquares(t, y[s], dy[s])
        result = bls_boot.autopower(
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.33],
            minimum_period=pmin,
            maximum_period=pmax,
            frequency_factor=ffac,
        )
        power_max.append(result.power.max())

    power_max = u.Quantity(power_max)
    power_max.sort()

    return power_max


def fap_bootstrap(Z, pmin, pmax, ffac, t, y, dy, n_bootstraps=1000, random_seed=None):
    """Bootstrap estimate of the false alarm probability"""
    pmax = _bootstrap_max(t, y, dy, pmin, pmax, ffac, random_seed, n_bootstraps)

    return 1 - np.searchsorted(pmax, Z) / len(pmax)


def get_bls_periods(lcs, plot=False, n_boots=100):

    search_period = np.arange(0, 25, 0.2)[1:]
    period_best, period_fap, snr = [], [], []
    pmin, pmax, ffac = 0.5, 20, 10

    for lc in tqdm(lcs, desc="BLS search", leave=False):
        lc = lc.remove_outliers(sigma_lower=1e10, sigma_upper=5)
        periodogram = lc.to_periodogram(
            method="bls",
            minimum_period=pmin,
            maximum_period=pmax,
            frequency_factor=ffac,
        )
        best_fit_period = periodogram.period_at_max_power
        power_snr = periodogram.snr[np.argmax(periodogram.power)]
        period_best.append(best_fit_period)
        snr.append(power_snr)
        if n_boots > 0:
            p_fap = fap_bootstrap(
                periodogram.power.max(),
                pmin,
                pmax,
                ffac,
                lc.time,
                lc.flux,
                lc.flux_err,
                n_bootstraps=n_boots,
                random_seed=99,
            )
        else:
            p_fap = np.nan
        period_fap.append(np.nan)

        if plot:
            fig = plt.figure(figsize=(13, 9))

            plt.subplots_adjust(wspace=0.25, hspace=0.25)

            sub1 = fig.add_subplot(2, 2, 1)
            sub1.axvline(
                best_fit_period.value,
                color="r",
                linestyle="--",
                label="Period : %.3f d \nsnr %.4f\nfap : %.3f"
                % (best_fit_period.value, power_snr, p_fap),
            )
            periodogram.plot(ax=sub1)
            sub1.legend()

            sub2 = fig.add_subplot(2, 2, 2)
            lc.fold(
                period=best_fit_period,
                epoch_time=periodogram.transit_time_at_max_power,
            ).errorbar(ax=sub2, alpha=0.8)

            sub3 = fig.add_subplot(2, 2, (3, 4))
            lc.plot(ax=sub3)
            periodogram.get_transit_model().plot(ax=sub3, color="r")
            plt.show()

    return u.Quantity(period_best), np.array(period_fap), np.array(snr)


def solve_linear_model(A, y, y_err=None, prior_mu=None, prior_sigma=None, k=None):
    """
    Solves a linear model with design matrix A and observations y:
        Aw = y
    return the solutions w for the system assuming Gaussian priors.
    Alternatively the observation errors, priors, and a boolean mask for the
    observations (row axis) can be provided.

    Adapted from Luger, Foreman-Mackey & Hogg, 2017
    (https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract)

    Parameters
    ----------
    A : numpy ndarray or scipy sparce csr matrix
        Desging matrix with solution basis
        shape n_observations x n_basis
    y : numpy ndarray
        Observations
        shape n_observations
    y_err : numpy ndarray, optional
        Observation errors
        shape n_observations
    prior_mu : float, optional
        Mean of Gaussian prior values for the weights (w)
    prior_sigma : float, optional
        Standard deviation of Gaussian prior values for the weights (w)
    k : boolean, numpy ndarray, optional
        Mask that sets the observations to be used to solve the system
        shape n_observations

    Returns
    -------
    w : numpy ndarray
        Array with the estimations for the weights
        shape n_basis
    werrs : numpy ndarray
        Array with the error estimations for the weights, returned if y_err is
        provided
        shape n_basis
    """
    if k is None:
        k = np.ones(len(y), dtype=bool)

    if y_err is not None:
        sigma_w_inv = A[k].T.dot(A[k].multiply(1 / y_err[k, None] ** 2))
        B = A[k].T.dot((y[k] / y_err[k] ** 2))
    else:
        sigma_w_inv = A[k].T.dot(A[k])
        B = A[k].T.dot(y[k])

    if type(sigma_w_inv) == sparse.csr_matrix:
        sigma_w_inv = sigma_w_inv.toarray()

    if prior_mu is not None and prior_sigma is not None:
        sigma_w_inv += np.diag(1 / prior_sigma ** 2)
        B += prior_mu / prior_sigma ** 2
    w = np.linalg.solve(sigma_w_inv, B)
    if y_err is not None:
        w_err = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
        return w, w_err
    return w


def clean_aperture_mask(mask):
    def remove_isolated(mask, n=1):
        if n == 0:
            return mask
        true_pos = np.array(np.where(mask))
        mask_c = mask.copy()
        for i, j in true_pos.T:
            if i in [0, mask.shape[0] - 1] or j in [0, mask.shape[1] - 1]:
                up = i + 1 if (i != mask.shape[0] - 1) else 0
                down = i - 1
                left = j - 1
                right = j + 1 if (j != mask.shape[1] - 1) else 0
                if not any([mask[down, j], mask[up, j], mask[i, left], mask[i, right]]):
                    mask_c[i, j] = False
                continue
            if (
                not sum(
                    [
                        mask[i - 1, j],
                        mask[i + 1, j],
                        mask[i, j + 1],
                        mask[i, j - 1],
                        mask[i - 1, j - 1],
                        mask[i + 1, j + 1],
                        mask[i + 1, j - 1],
                        mask[i - 1, j + 1],
                    ]
                )
                > 2
                and mask.sum() > 3
            ):
                mask_c[i, j] = False
        return remove_isolated(mask_c, n - 1)

    for i in range(len(mask)):
        mask[i] = remove_isolated(mask[i], 1)

    return mask


def make_A(phi, r, cut_r=5):
    """ Make spline design matrix in polar coordinates """
    phi_spline = sparse.csr_matrix(wrapped_spline(phi, order=3, nknots=6).T)
    low_knot = np.percentile(r, 3)
    upp_knot = np.percentile(r, 90)
    r_knots = np.linspace(low_knot ** 0.5, upp_knot ** 0.5, 8) ** 2
    r_spline = sparse.csr_matrix(
        np.asarray(
            dmatrix(
                "bs(x, knots=knots, degree=3, include_intercept=True)",
                {"x": list(r), "knots": r_knots},
            )
        )
    )
    X = sparse.hstack(
        [phi_spline.multiply(r_spline[:, idx]) for idx in range(r_spline.shape[1])],
        format="csr",
    )
    cut = np.arange(phi_spline.shape[1] * 1, phi_spline.shape[1] * cut_r)
    a = list(set(np.arange(X.shape[1])) - set(cut))
    X1 = sparse.hstack(
        [X[:, a], r_spline[:, 1:cut_r], sparse.csr_matrix(np.ones(X.shape[0])).T],
        format="csr",
    )
    return X1


def wrapped_spline(input_vector, order=2, nknots=10):
    """
    Creates a vector of folded-spline basis according to the input data. This is meant
    to be used to build the basis vectors for periodic data, like the angle in polar
    coordinates.

    Parameters
    ----------
    input_vector : numpy.ndarray
        Input data to create basis, angle values MUST BE BETWEEN -PI and PI.
    order : int
        Order of the spline basis
    nknots : int
         Number of knots for the splines

    Returns
    -------
    folded_basis : numpy.ndarray
        Array of folded-spline basis
    """

    if not ((input_vector > -np.pi) & (input_vector < np.pi)).all():
        raise ValueError("Must be between -pi and pi")
    x = np.copy(input_vector)
    x1 = np.hstack([x, x + np.pi * 2])
    nt = (nknots * 2) + 1

    t = np.linspace(-np.pi, 3 * np.pi, nt)
    dt = np.median(np.diff(t))
    # Zeroth order basis
    basis = np.asarray(
        [
            ((x1 >= t[idx]) & (x1 < t[idx + 1])).astype(float)
            for idx in range(len(t) - 1)
        ]
    )
    # Higher order basis
    for order in np.arange(1, 4):
        basis_1 = []
        for idx in range(len(t) - 1):
            a = ((x1 - t[idx]) / (dt * order)) * basis[idx]

            if ((idx + order + 1)) < (nt - 1):
                b = (-(x1 - t[(idx + order + 1)]) / (dt * order)) * basis[
                    (idx + 1) % (nt - 1)
                ]
            else:
                b = np.zeros(len(x1))
            basis_1.append(a + b)
        basis = np.vstack(basis_1)

    folded_basis = np.copy(basis)[: nt // 2, : len(x)]
    for idx in np.arange(-order, 0):
        folded_basis[idx, :] += np.copy(basis)[nt // 2 + idx, len(x) :]
    return folded_basis
