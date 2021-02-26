import os, sys, glob
import argparse
import numpy as np
from scipy import sparse
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, match_coordinates_3d
from astropy.stats import sigma_clip, SigmaClip
from astropy.wcs import WCS
import pickle
from photutils import Background2D, MedianBackground, BkgZoomInterpolator

path = os.path.dirname(os.getcwd())

sys.path.append(path)
from exba.utils import get_gaia_sources, make_A, make_A_edges, solve_linear_model

parser = argparse.ArgumentParser(description="AutoEncoder")
parser.add_argument(
    "--quarter",
    dest="quarter",
    type=int,
    default=5,
    help="Which quarter.",
)
parser.add_argument(
    "--channel",
    dest="channel",
    type=int,
    default=1,
    help="List of files to be downloaded",
)
parser.add_argument(
    "--dm-type",
    dest="dm_type",
    type=str,
    default="cuadratic",
    help="Type of basis for desing matrix",
)
parser.add_argument(
    "--plot",
    dest="plot",
    action="store_true",
    default=True,
    help="Make plots.",
)
parser.add_argument(
    "--save",
    dest="save",
    action="store_true",
    default=True,
    help="Save models.",
)
parser.add_argument(
    "--dry-run",
    dest="dry_run",
    action="store_true",
    default=False,
    help="dry run.",
)
args = parser.parse_args()

r_min, r_max = 20, 1044
c_min, c_max = 12, 1112

remove_sat = True
sample_sources = True
N_sample = 500


def model_bkg(data, mask=None):
    """
    BkgZoomInterpolator:
    This class generates full-sized background and background RMS images
    from lower-resolution mesh images using the `~scipy.ndimage.zoom`
    (spline) interpolator.
    """
    model = Background2D(
        data,
        mask=mask,
        box_size=(64, 50),
        filter_size=15,
        exclude_percentile=20,
        sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
        bkg_estimator=MedianBackground(),
        interpolator=BkgZoomInterpolator(order=3),
    )

    return model.background


def do_query(ra_q, dec_q, rad, epoch, quarter, channel):
    file_name = "%s/data/ffi/%i/channel_%i_gaia_xmatch.csv" % (path, quarter, channel)
    if os.path.isfile(file_name):
        print("Loading query from file...")
        print(file_name)
        sources = pd.read_csv(file_name)
    else:
        sources = get_gaia_sources(
            tuple(ra_q),
            tuple(dec_q),
            tuple(rad),
            magnitude_limit=18,
            epoch=epoch,
            gaia="dr2",
        )
        print("Saving query to file...")
        print(file_name)
        sources.to_csv(file_name)
    return sources


def clean_source_list(sources):

    print("Cleaning sources table:")
    # remove bright/faint objects
    sources = sources[
        (sources.phot_g_mean_flux > 1e3) & (sources.phot_g_mean_flux < 1e6)
    ].reset_index(drop=True)
    print(sources.shape)

    # find well separated sources
    s_coords = SkyCoord(sources.ra, sources.dec, unit=("deg"))
    midx, mdist = match_coordinates_3d(s_coords, s_coords, nthneighbor=2)[:2]
    # remove sources closer than 4" = 1 pix
    closest = mdist.arcsec < 8.0
    blocs = np.vstack([midx[closest], np.where(closest)[0]])
    bmags = np.vstack(
        [
            sources.phot_g_mean_mag[midx[closest]],
            sources.phot_g_mean_mag[np.where(closest)[0]],
        ]
    )
    faintest = [blocs[idx][s] for s, idx in enumerate(np.argmax(bmags, axis=0))]
    unresolved = np.in1d(np.arange(len(sources)), faintest)
    del s_coords, midx, mdist, closest, blocs, bmags

    sources = sources[~unresolved].reset_index(drop=True)
    print(sources.shape)

    # find sources inside the image with 10 pix of inward tolerance
    inside = (
        (sources.row > 10)
        & (sources.row < 1014)
        & (sources.col > 10)
        & (sources.col < 1090)
    )
    sources = sources[inside].reset_index(drop=True)
    print(sources.shape)
    print("done!")

    return sources


def _saturated_pixels_mask(flux, column, row, saturation_limit=1.5e5):
    """Finds and removes saturated pixels, including bleed columns."""
    # Which pixels are saturated
    saturated = np.nanpercentile(flux, 99, axis=0)
    saturated = np.where((saturated > saturation_limit).astype(float))[0]

    # Find bad pixels, including allowence for a bleed column.
    bad_pixels = np.vstack(
        [
            np.hstack([column[saturated] + idx for idx in np.arange(-3, 3)]),
            np.hstack([row[saturated] for idx in np.arange(-3, 3)]),
        ]
    ).T
    # Find unique row/column combinations
    bad_pixels = bad_pixels[
        np.unique(["".join(s) for s in bad_pixels.astype(str)], return_index=True)[1]
    ]
    # Build a mask of saturated pixels
    m = np.zeros(len(column), bool)
    for p in bad_pixels:
        m |= (column == p[0]) & (row == p[1])
    return m


def find_psf_edge(r, mean_flux, gf, radius_limit=6, cut=300, dm_type="cuadratic"):

    temp_mask = r < radius_limit
    temp_mask &= temp_mask.sum(axis=0) == 1

    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.log10((temp_mask.astype(float) * mean_flux))
    k = np.isfinite(f[temp_mask])

    A = make_A_edges(r[temp_mask], np.log10(gf[temp_mask]), type=dm_type)

    for count in [0, 1]:
        sigma_w_inv = A[k].T.dot(A[k])
        B = A[k].T.dot(f[temp_mask][k])
        w = np.linalg.solve(sigma_w_inv, B)
        res = np.ma.masked_array(f[temp_mask], ~k) - A.dot(w)
        k &= ~sigma_clip(res, sigma=5).mask

    test_f = np.linspace(
        np.log10(gf.min()),
        np.log10(gf.max()),
        100,
    )
    test_r = np.arange(0, radius_limit, 0.125)
    test_r2, test_f2 = np.meshgrid(test_r, test_f)

    test_A = make_A_edges(test_r2.ravel(), test_f2.ravel(), type=dm_type)
    test_val = test_A.dot(w).reshape(test_r2.shape)

    # find radius where flux > cut
    l = np.zeros(len(test_f)) * np.nan
    for idx in range(len(test_f)):
        loc = np.where(10 ** test_val[idx] < cut)[0]
        if len(loc) > 0:
            l[idx] = test_r[loc[0]]

    ok = np.isfinite(l)
    polifit_results = np.polyfit(test_f[ok], l[ok], 2)
    source_radius_limit = np.polyval(polifit_results, np.log10(gf[:, 0]))
    source_radius_limit[source_radius_limit > radius_limit] = radius_limit
    source_radius_limit[source_radius_limit < 0] = 0

    if args.save:
        to_save = dict(w=w, polifit_results=polifit_results)
        output = "%s/data/ffi/%i/channel_%i_psf_edge_model_%s.pkl" % (
            path,
            args.quarter,
            args.channel,
            dm_type,
        )
        with open(output, "wb") as file:
            pickle.dump(to_save, file)

    if args.plot:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

        ax[0].scatter(r[temp_mask], f[temp_mask], s=0.4, c="k", alpha=0.5, label="Data")
        ax[0].scatter(
            r[temp_mask][k],
            f[temp_mask][k],
            s=0.4,
            c="g",
            alpha=0.5,
            label="Data clipped",
        )
        ax[0].scatter(
            r[temp_mask][k], A[k].dot(w), c="r", s=0.4, alpha=0.7, label="Model"
        )
        ax[0].set(xlabel=("Radius [pix]"), ylabel=("log$_{10}$ Flux"))
        ax[0].legend(frameon=True)

        im = ax[1].pcolormesh(
            test_f2,
            test_r2,
            10 ** test_val,
            vmin=0,
            vmax=500,
            cmap="viridis",
            shading="auto",
        )
        line = np.polyval(np.polyfit(test_f[ok], l[ok], 2), test_f)
        line[line > radius_limit] = radius_limit
        ax[1].plot(test_f, line, color="r", label="Mask threshold")
        ax[1].legend(frameon=True)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Contained PSF Flux [counts]")

        ax[1].set(
            ylabel=("Radius from Source [pix]"),
            xlabel=("log$_{10}$ Source Flux"),
        )
        fig_name = "%s/data/ffi/%i/channel_%i_psf_edge_model_%s.png" % (
            path,
            args.quarter,
            args.channel,
            dm_type,
        )

        plt.savefig(fig_name, format="png")
        plt.clf()

    return source_radius_limit


def build_psf_model(r, phi, mean_flux, flux_estimates, radius):
    warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)

    # assign the flux estimations to be used for mean flux normalization
    source_mask = sparse.csr_matrix(r < radius[:, None])

    # mean flux values using uncontaminated mask and normalized by flux estimations
    mean_f = np.log10(
        source_mask.astype(float).multiply(mean_flux).multiply(1 / flux_estimates).data
    )
    phi_b = source_mask.multiply(phi).data
    r_b = source_mask.multiply(r).data

    # build a design matrix A with b-splines basis in radius and angle axis.
    A = make_A(phi_b.ravel(), r_b.ravel())
    prior_sigma = np.ones(A.shape[1]) * 100
    prior_mu = np.zeros(A.shape[1])
    nan_mask = np.isfinite(mean_f.ravel())

    # we solve for A * psf_w = mean_f
    psf_w = solve_linear_model(
        A,
        mean_f.ravel(),
        k=nan_mask,
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
    )

    # We then build the same design matrix for all pixels with flux
    Ap = make_A(
        source_mask.multiply(phi).data,
        source_mask.multiply(r).data,
    )

    # And create a `mean_model` that has the psf model for all pixels with fluxes
    mean_model = sparse.csr_matrix(r.shape)
    m = 10 ** Ap.dot(psf_w)
    mean_model[source_mask] = m
    mean_model.eliminate_zeros()
    mean_model = mean_model.multiply(1 / mean_model.sum(axis=1))

    if args.save:
        to_save = dict(psf_w=psf_w)
        output = "%s/data/ffi/%i/channel_%i_psf_model.pkl" % (
            path,
            args.quarter,
            args.channel,
        )
        with open(output, "wb") as file:
            pickle.dump(to_save, file)

    if args.plot:
        # Plotting r,phi,meanflux used to build PSF model
        ylim = r_b.max() * 1.1
        vmin, vmax = mean_f.min(), mean_f.max()
        fig, ax = plt.subplots(1, 3, figsize=(14, 3))
        ax[0].set_title("Mean flux")
        cax = ax[0].scatter(
            source_mask.multiply(phi).data,
            source_mask.multiply(r).data,
            c=mean_f,
            marker=".",
            vmin=vmin,
            vmax=vmax,
        )
        ax[0].set_ylim(0, ylim)
        fig.colorbar(cax, ax=ax[0])
        ax[0].set_ylabel(r"$r$ [pixels]")
        ax[0].set_xlabel(r"$\phi$ [rad]")

        ax[1].set_title("Average PSF Model")
        cax = cax = ax[1].scatter(
            source_mask.multiply(phi).data,
            source_mask.multiply(r).data,
            c=np.log10(m),
            marker=".",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(cax, ax=ax[1])
        ax[1].set_xlabel(r"$\phi$ [rad]")

        ax[2].set_title("Average PSF Model (grid)")
        r_test, phi_test = np.meshgrid(
            np.linspace(0 ** 0.5, ylim ** 0.5, 100) ** 2,
            np.linspace(-np.pi + 1e-5, np.pi - 1e-5, 100),
        )
        A_test = make_A(phi_test.ravel(), r_test.ravel())
        model_test = A_test.dot(psf_w)
        model_test = model_test.reshape(phi_test.shape)
        cax = ax[2].pcolormesh(phi_test, r_test, model_test, shading="auto")
        fig.colorbar(cax, ax=ax[2])
        ax[2].set_xlabel(r"$\phi$ [rad]")

        fig_name = "%s/data/ffi/%i/channel_%i_psf_model.png" % (
            path,
            args.quarter,
            args.channel,
        )

        plt.savefig(fig_name, format="png")
        plt.clf()

    return mean_model


def run_code(Q=5, CH=1):

    # Loading image and header data
    fits_list = np.sort(glob.glob("%s/data/ffi/%i/kplr*_ffi-cal.fits" % (path, Q)))
    test_file = fits_list[0]
    print("Working with file: ", test_file)
    hdr = fits.open(test_file)[CH].header
    img = fits.open(test_file)[CH].data
    wcs = WCS(hdr)
    print("Image shape: ", img.shape)

    # get ra,dec coordinates and pixel mesh
    row_2d, col_2d = np.mgrid[: img.shape[0], : img.shape[1]]
    row, col = row_2d.ravel(), col_2d.ravel()
    ra, dec = wcs.all_pix2world(np.vstack([col, row]).T, 0).T
    ra_2d, dec_2d = ra.reshape(img.shape), dec.reshape(img.shape)

    # get coordinates of the center for query
    loc = (img.shape[0] // 2, img.shape[1] // 2)
    ra_q, dec_q = wcs.all_pix2world(np.atleast_2d(loc), 0).T
    rad = [np.hypot(ra - ra.mean(), dec - dec.mean()).max()]

    time = Time(hdr["TSTART"] + 2454833, format="jd")
    print(
        "Will query with this (ra, dec, radius, epoch): ", ra_q, dec_q, rad, time.jyear
    )
    sources = do_query(ra_q, dec_q, rad, time.jyear, Q, CH)
    sources["col"], sources["row"] = wcs.all_world2pix(
        sources.loc[:, ["ra", "dec"]].values, 0.5
    ).T

    # correct col,row columns for gaia sources
    _sources = sources.copy()
    _sources.row -= r_min
    _sources.col -= c_min

    # clean sources
    _sources = clean_source_list(_sources)

    # remove useless Pixels
    _col_2d = col_2d[r_min:r_max, c_min:c_max] - c_min
    _row_2d = row_2d[r_min:r_max, c_min:c_max] - r_min
    _ra_2d = ra_2d[r_min:r_max, c_min:c_max]
    _dec_2d = dec_2d[r_min:r_max, c_min:c_max]
    _flux_2d = img[r_min:r_max, c_min:c_max]

    # background substraction
    _flux_2d = _flux_2d - model_bkg(_flux_2d, mask=None)

    _col = _col_2d.ravel()
    _row = _row_2d.ravel()
    _ra = _ra_2d.ravel()
    _dec = _dec_2d.ravel()
    _flux = _flux_2d.ravel()

    print("New shape is:", _col_2d.shape)
    print("New shape is (ravel):", _col.shape)

    if args.plot:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].pcolor(_flux_2d, vmin=0, vmax=1000)
        ax[0].scatter(
            _sources.col, _sources.row, edgecolor="red", facecolor="None", s=10
        )
        ax[1].pcolor(_flux_2d, vmin=0, vmax=1000)
        ax[1].scatter(
            _sources.col, _sources.row, edgecolor="red", facecolor="None", s=10
        )
        ax[1].set_xlim(500, 600)
        ax[1].set_ylim(500, 600)

        ax[0].set_ylabel("Row pixels")
        ax[0].set_xlabel("Row pixels")
        ax[1].set_xlabel("Row pixels")
        fig_name = "%s/data/ffi/%i/channel_%i_image_gaia_sources.png" % (path, Q, CH)

        plt.savefig(fig_name, format="png")
        plt.clf()

    if remove_sat:
        non_sat_mask = ~_saturated_pixels_mask(
            _flux, _col, _row, saturation_limit=1.5e5
        )

        _col = _col[non_sat_mask]
        _row = _row[non_sat_mask]
        _ra = _ra[non_sat_mask]
        _dec = _dec[non_sat_mask]
        _flux = _flux[non_sat_mask]

        print("New shape is (non-sat): ", _col.shape)

    if sample_sources:
        if args.plot:
            plt.figure(figsize=(6, 4))
            plt.hist(np.log10(_sources.phot_g_mean_flux), bins=200, label="original")

        _sources = _sources.sample(N_sample)
        # _sources = sample_sources(_sources)
        print("New number of sources (subsample): ", _sources.shape)

        if args.plot:
            plt.hist(np.log10(_sources.phot_g_mean_flux), bins=200, label="sample")
            plt.yscale("log")
            plt.xlabel("Gaia flux [log]")
            plt.legend(loc="upper right")

            fig_name = "%s/data/ffi/%i/channel_%i_gaia_flux_dist.png" % (path, Q, CH)

            plt.savefig(fig_name, format="png")
            plt.clf()

    # create dx, dy, gf, r, phi, vectors
    # gaia estimate flux values per pixel to be used as flux priors
    print("Computing dx, dy, gf...")
    dx, dy, gf = np.asarray(
        [
            np.vstack(
                [
                    _col - _sources["col"].iloc[idx],
                    _row - _sources["row"].iloc[idx],
                    np.zeros(len(_col)) + _sources.phot_g_mean_flux.iloc[idx],
                ]
            )
            for idx in range(len(_sources))
        ]
    ).transpose([1, 0, 2])

    # convertion to polar coordinates
    print("to polar coordinates...")
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)

    # compute PSF edge model
    print("Computing PSF edges...")
    radius = find_psf_edge(r, _flux, gf, radius_limit=6, cut=300, dm_type=args.dm_type)

    # compute PSF model
    print("Computing PSF model...")
    psf_model = build_psf_model(r, phi, _flux, gf, radius * 2)


if __name__ == "__main__":
    print("Running PSF models for Q: %i Ch: %i" % (args.quarter, args.channel))
    if args.dry_run:
        print("Dry run mode, exiting...")
        sys.exit()
    run_code(Q=args.quarter, CH=args.channel)