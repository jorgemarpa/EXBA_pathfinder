"""Tools to stitch EXBA images and extract LC of detected Gaia sources"""
import os, glob
import warnings

import numpy as np
import pandas as pd
import _pickle as cPickle
import lightkurve as lk
from lightkurve.correctors import CBVCorrector
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patches
from tqdm.notebook import tqdm
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, match_coordinates_3d
from astropy.stats import sigma_clip

from .utils import get_gaia_sources, get_bls_periods

main_path = os.path.dirname(os.getcwd())


class EXBA(object):
    def __init__(self, channel=53, quarter=5, path=main_path):

        self.quarter = quarter
        self.channel = channel

        # load local TPFs files
        tpfs_paths = np.sort(
            glob.glob(
                "%s/data/EXBA/%s/%s/*_lpd-targ.fits.gz"
                % (main_path, str(channel), str(quarter))
            )
        )
        self.tpfs_files = tpfs_paths

        tpfs = lk.TargetPixelFileCollection(
            [lk.KeplerTargetPixelFile(f) for f in tpfs_paths[:4]]
        )
        self.tpfs = tpfs
        # check for same channels and quarter
        channels = [tpf.get_header()["CHANNEL"] for tpf in tpfs]
        quarters = [tpf.get_header()["QUARTER"] for tpf in tpfs]
        # target_ids = [tpf.get_header()["OBJECT"] for tpf in tpfs]
        # ra_objects = [tpf.get_header()["RA_OBJ"] for tpf in tpfs]
        # dec_objects = [tpf.get_header()["DEC_OBJ"] for tpf in tpfs]

        if len(set(channels)) != 1 and list(set(channels)) != [channel]:
            raise ValueError(
                "All TPFs must be from the same channel %s"
                % ",".join([str(k) for k in channels])
            )

        if len(set(quarters)) != 1 and list(set(quarters)) != [quarter]:
            raise ValueError(
                "All TPFs must be from the same quarter %s"
                % ",".join([str(k) for k in quarters])
            )

        # stich channel's strips and parse TPFs
        (
            self.time,
            self.cadences,
            row,
            col,
            flux,
            flux_err,
            unw,
        ) = self._parse_TPFs_channel(tpfs)
        self.row_2d, self.col_2d, self.flux_2d, self.flux_err_2d, self.unw_2d = (
            row.copy(),
            col.copy(),
            flux.copy(),
            flux_err.copy(),
            unw.copy(),
        )
        self.row, self.col, self.flux, self.flux_err, self.unw = (
            row.ravel(),
            col.ravel(),
            flux.reshape(flux.shape[0], np.product(flux.shape[1:])),
            flux_err.reshape(flux_err.shape[0], np.product(flux_err.shape[1:])),
            unw.ravel(),
        )
        self.ra, self.dec = self._convert_to_wcs(tpfs, self.row, self.col)
        self.ra_2d, self.dec_2d = (
            self.ra.reshape(self.row_2d.shape),
            self.dec.reshape(self.row_2d.shape),
        )

        # search Gaia sources in the sky
        sources = self._get_coord_and_query_gaia(
            self.ra, self.dec, self.unw, self.time[0], magnitude_limit=20
        )
        sources["col"], sources["row"] = tpfs[0].wcs.wcs_world2pix(
            sources.ra, sources.dec, 0.5
        )
        sources["col"] += tpfs[0].column
        sources["row"] += tpfs[0].row
        self.sources, self.bad_sources = self._clean_source_list(
            sources, self.ra, self.dec
        )

        self.dx, self.dy, self.gf = np.asarray(
            [
                np.vstack(
                    [
                        self.col - self.sources["col"][idx],
                        self.row - self.sources["row"][idx],
                        np.zeros(len(self.col)) + self.sources.phot_g_mean_flux[idx],
                    ]
                )
                for idx in range(len(self.sources))
            ]
        ).transpose([1, 0, 2])

    def __repr__(self):
        q_result = ",".join([str(k) for k in list([self.quarter])])
        return "EXBA Patch:\n\t Channel %i, Quarter %s, Gaia sources %i" % (
            self.channel,
            q_result,
            len(self.sources),
        )

    def _parse_TPFs_channel(self, tpfs):
        cadences = np.array([tpf.cadenceno for tpf in tpfs])
        # check if all TPFs has same cadences
        if not np.all(cadences[1:, :] - cadences[-1:, :] == 0):
            raise ValueError("All TPFs must have same time basis")

        # make sure tpfs are sorted by colum direction
        tpfs = lk.TargetPixelFileCollection(
            [tpfs[i] for i in np.argsort([tpf.column for tpf in tpfs])]
        )

        # extract times
        times = tpfs[0].time.jd

        # extract row,column mesh grid
        col, row = np.hstack(
            [
                np.mgrid[
                    tpf.column : tpf.column + tpf.shape[2],
                    tpf.row : tpf.row + tpf.shape[1],
                ]
                for tpf in tpfs
            ]
        )

        # extract flux vales
        flux = np.hstack([tpf.flux.transpose(1, 2, 0) for tpf in tpfs]).transpose(
            2, 0, 1
        )
        flux_err = np.hstack(
            [tpf.flux_err.transpose(1, 2, 0) for tpf in tpfs]
        ).transpose(2, 0, 1)

        # bookkeeping of tpf-pixel
        unw = np.hstack(
            [np.ones(tpf.shape[1:], dtype=np.int) * i for i, tpf in enumerate(tpfs)]
        )

        return times, cadences[0], row.T, col.T, flux, flux_err, unw

    def _convert_to_wcs(self, tpfs, row, col):
        ra, dec = tpfs[0].wcs.wcs_pix2world(
            (col - tpfs[0].column), (row - tpfs[0].row), 0.0
        )

        return ra, dec

    def _get_coord_and_query_gaia(rself, ra, dec, unw, epoch=2020, magnitude_limit=20):
        """
        Calculate ra, dec coordinates and search radius to query Gaia catalog

        Parameters
        ----------
        ra : numpy.ndarray
            Right ascension coordinate of pixels to do Gaia search
        ra : numpy.ndarray
            Declination coordinate of pixels to do Gaia search
        unw : numpy.ndarray
            TPF index of each pixel
        epoch : float
            Epoch of obervation in Julian Days of ra, dec coordinates,
            will be used to propagate proper motions in Gaia.

        Returns
        -------
        sources : pandas.DataFrame
            Catalog with query result
        """
        # find the max circle per TPF that contain all pixel data to query Gaia
        ras, decs, rads = [], [], []
        for l in np.unique(unw[0]):
            ra1 = ra[unw[0] == l]
            dec1 = dec[unw[0] == l]
            ras.append(ra1.mean())
            decs.append(dec1.mean())
            rads.append(
                np.hypot(ra1 - ra1.mean(), dec1 - dec1.mean()).max()
                + (u.arcsecond * 6).to(u.deg).value
            )
        # query Gaia with epoch propagation
        sources = get_gaia_sources(
            tuple(ras),
            tuple(decs),
            tuple(rads),
            magnitude_limit=magnitude_limit,
            epoch=Time(epoch, format="jd").jyear,
        )
        return sources

    def _clean_source_list(self, sources, ra, dec):
        # find sources on the image
        inside = np.zeros(len(sources), dtype=bool)
        # max distance in arcsec from image edge to source ra, dec
        off = 6.0 / 3600
        for k in range(len(sources)):
            raok = (sources["ra"][k] > ra - off) & (sources["ra"][k] < ra + off)
            decok = (sources["dec"][k] > dec - off) & (sources["dec"][k] < dec + off)
            inside[k] = (raok & decok).any()
        del raok, decok

        # find well separated sources
        s_coords = SkyCoord(sources.ra, sources.dec, unit=("deg"))
        midx, mdist = match_coordinates_3d(s_coords, s_coords, nthneighbor=2)[:2]
        # remove sources closer than 4" = 1 pix
        closest = mdist.arcsec < 2.0
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

        # Keep track of sources that we removed
        sources.loc[:, "clean_flag"] = 0
        sources.loc[~inside, "clean_flag"] += 2 ** 0  # outside TPF
        sources.loc[unresolved, "clean_flag"] += 2 ** 1  # close contaminant

        # combine 2 source masks
        clean = sources.clean_flag == 0
        removed_sources = sources[~clean].reset_index(drop=True)
        sources = sources[clean].reset_index(drop=True)

        return sources, removed_sources

    def simple_aperture_phot(self, space="pix-sq"):
        if space == "world":
            aper = (u.arcsecond * 2 * 4).to(u.deg).value  # aperture radii in deg
            aperture_mask = [
                np.hypot(self.ra - s.ra, self.dec - s.dec) < aper
                for _, s in self.sources.iterrows()
            ]
        elif space == "pix-cir":
            aper = 1.7
            aperture_mask = [
                np.hypot(self.col - s.col, self.row - s.row) < aper
                for _, s in self.sources.iterrows()
            ]
        elif space == "pix-sq":
            aper = [1.5, 1.5]
            aperture_mask = [
                (np.abs(self.col - np.floor(s.col)) < aper[1])
                & (np.abs(self.row - np.floor(s.row)) < aper[0])
                for _, s in self.sources.iterrows()
            ]
        elif space == "pix-auto":
            if not hasattr(self, "radius"):
                raise AttributeError(
                    "No PSF radius computed, please run `_find_psf_edge()` first"
                )
            # create circular aperture mask using PSF edges
            aperture_mask = [
                np.hypot(self.col - s.col, self.row - s.row) <= r
                for r, (_, s) in zip(self.radius, self.sources.iterrows())
            ]
            # create a background mask
            sq_ap = [
                (np.abs(self.col - np.floor(s.col)) < 6)
                & (np.abs(self.row - np.floor(s.row)) < 6)
                for _, s in self.sources.iterrows()
            ]
            bkg_mask = np.asarray(sq_ap).sum(axis=0) == 0
            self.bkg_mask = bkg_mask.reshape(self.col_2d.shape)
            # compute a SNR image
            mean_flux = self.flux.value.mean(axis=0)
            bkg_std = mean_flux[bkg_mask].std()
            snr_img = np.abs(mean_flux / bkg_std)
            self.snr_img = snr_img.reshape(self.col_2d.shape)
            # combine circular aperture with SNR > 5 mask
            aperture_mask &= snr_img > 3

        aperture_mask = np.asarray(aperture_mask)
        sap = np.zeros((self.sources.shape[0], self.flux.shape[0]))
        sap_e = np.zeros((self.sources.shape[0], self.flux.shape[0]))

        for tdx in tqdm(range(len(self.flux)), desc="Simple SAP flux", leave=False):
            sap[:, tdx] = [self.flux[tdx][mask].value.sum() for mask in aperture_mask]
            sap_e[:, tdx] = [
                np.power(self.flux_err[tdx][mask].value, 2).sum() ** 0.5
                for mask in aperture_mask
            ]

        # check for light curves with zero flux
        # zero_mask = np.all((sap == 0), axis=1)
        # sap = sap[~zero_mask]
        # sap_e = sap_e[~zero_mask]
        # aperture_mask = aperture_mask[~zero_mask]
        # self.bad_sources = pd.concat(
        #     [self.bad_sources, self.sources[zero_mask]]
        # ).reset_index(drop=True)
        # self.sources = self.sources[~zero_mask].reset_index(drop=True)

        self.aperture_mask = aperture_mask.reshape(
            self.sources.shape[0], self.flux_2d.shape[1], self.flux_2d.shape[2]
        )
        self.sap_lcs = lk.LightCurveCollection(
            [
                lk.KeplerLightCurve(
                    time=self.time,
                    cadenceno=self.cadences,
                    flux=sap[i],
                    flux_err=sap_e[i],
                    time_format="bkjd",
                    flux_unit="electron/s",
                    targetid=self.sources.designation[i],
                    label=self.sources.designation[i],
                    mission="Kepler",
                    quarter=int(self.quarter),
                    channel=int(self.channel),
                    ra=self.sources.ra[i],
                    dec=self.sources.dec[i],
                ).remove_outliers(sigma=5)
                for i in range(len(sap))
            ]
        )
        return

    def _find_psf_edge(self, radius_limit=6, cut=300, plot=True):

        mean_flux = np.nanmean(self.flux, axis=0).value
        r = np.hypot(self.dx, self.dy)

        temp_mask = (r < radius_limit) & (self.gf < 1e6) & (self.gf > 1e4)
        temp_mask &= temp_mask.sum(axis=0) == 1

        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.log10((temp_mask.astype(float) * mean_flux))
        weights = (
            (self.flux_err ** 0.5).sum(axis=0) ** 0.5 / self.flux.shape[0]
        ) * temp_mask
        A = np.vstack(
            [
                r[temp_mask] ** 0,
                r[temp_mask],
                r[temp_mask] ** 2,
                np.log10(self.gf[temp_mask]),
                # np.log10(self.gf[temp_mask]) ** 2,
            ]
        ).T
        k = np.isfinite(f[temp_mask])
        for count in [0]:
            sigma_w_inv = A[k].T.dot(A[k] / weights[temp_mask][k, None] ** 2)
            B = A[k].T.dot(f[temp_mask][k] / weights[temp_mask][k] ** 2)
            w = np.linalg.solve(sigma_w_inv, B)
            res = np.ma.masked_array(f[temp_mask], ~k) - A.dot(w)
            k &= ~sigma_clip(res, sigma=3).mask

        test_f = np.linspace(
            np.log10(self.gf.min()),
            np.log10(self.gf.max()),
            100,
        )
        test_r = np.arange(0, radius_limit, 0.25)
        test_r2, test_f2 = np.meshgrid(test_r, test_f)

        test_val = (
            np.vstack(
                [
                    test_r2.ravel() ** 0,
                    test_r2.ravel(),
                    test_r2.ravel() ** 2,
                    test_f2.ravel(),
                    # test_f2.ravel() ** 2,
                ]
            )
            .T.dot(w)
            .reshape(test_r2.shape)
        )

        # find radius where flux > cut
        l = np.zeros(len(test_f)) * np.nan
        for idx in range(len(test_f)):
            loc = np.where(10 ** test_val[idx] < cut)[0]
            if len(loc) > 0:
                l[idx] = test_r[loc[0]]
        ok = np.isfinite(l)
        source_radius_limit = np.polyval(
            np.polyfit(test_f[ok], l[ok], 2), np.log10(self.gf[:, 0])
        )
        source_radius_limit[source_radius_limit > radius_limit] = radius_limit
        self.radius = source_radius_limit

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

            ax[0].scatter(r[temp_mask][k], f[temp_mask][k], s=0.4, c="k", label="Data")
            ax[0].scatter(r[temp_mask][k], A[k].dot(w), c="r", s=0.4, label="Model")
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
            plt.show()
        return

    def apply_CBV(self, do_under=False, ignore_warnings=True):
        if ignore_warnings:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=lk.utils.LightkurveWarning)

        # Select which CBVs to use in the correction
        cbv_type = ["SingleScale"]
        # Select which CBV indices to use
        # Use the first 8 SingleScale and all Spike CBVS
        cbv_indices = [np.arange(1, 9)]

        over_fit_m = []
        under_fit_m = []
        corrected_lcs = []

        for lc in tqdm(self.sap_lcs, desc="Applying CBVs to LCs", leave=False):
            lc = lc[lc.flux_err > 0]
            cbvcor = CBVCorrector(lc)
            cbvcor.correct_gaussian_prior(
                cbv_type=cbv_type, cbv_indices=cbv_indices, alpha=1e-4
            )
            over_fit_m.append(cbvcor.over_fitting_metric())
            if do_under:
                under_fit_m.append(cbvcor.under_fitting_metric())
            corrected_lcs.append(cbvcor.corrected_lc)

        self.corrected_lcs = lk.LightCurveCollection(corrected_lcs)
        self.over_fitting_metrics = np.array(over_fit_m)
        if do_under:
            self.under_fitting_metrics = np.array(under_fit_m)
        return

    def store_data(self):
        out_path = os.path.dirname(self.tpfs_files[0])
        self.sources.to_csv("%s/gaia_dr2_xmatch.csv" % (out_path))
        lc_out_name = "sap_lcs"
        if hasattr(self, "corrected_lcs"):
            lc_to_save = self.corrected_lcs
            lc_out_name += "_CBVcorrected"
        else:
            lc_to_save = self.sap_lcs
        with open("%s/%s.pkl" % (out_path, lc_out_name), "wb") as f:
            cPickle.dump(lc_to_save, f)
        return

    def check_for_rolling_band(self):
        """
        method to check if the exba patch suffers from rolling time-variable background
        noise, aka rolling band
        """
        # create a background mask, use self.aperture_mask.sum(axis=0) == 0

        # compute background light curves

        # inspect for low-freq variability, if so, then create model lc for bkg

        return

    def do_bls_search(self, plot=False, test_lcs=None, n_boots=100):

        if not hasattr(self, "corrected_lcs"):
            raise AttributeError(
                "No CBV corrected light curves computed, run apply_CBV first"
            )
        if test_lcs:
            search_here = list(test_lcs) if len(test_lcs) == 1 else test_lcs
        else:
            search_here = self.corrected_lcs

        period_best, period_fap, power_snr = get_bls_periods(
            search_here, plot=plot, n_boots=n_boots
        )

        if test_lcs:
            return period_best, period_fap, power_snr
        else:
            self.periods = period_best
            self.periods_fap = period_fap
            self.periods_snr = power_snr
        return

    def plot_image(self, space="pixels", sources=True, **kwargs):
        if space == "pixels":
            x = self.col_2d
            y = self.row_2d
            sx = self.sources.col
            sy = self.sources.row
            xlabel = "Pixel Column Number"
            xlabel = "Pixel Row Number"
        elif space == "wcs":
            x = self.ra_2d
            y = self.dec_2d
            sx = self.sources.ra
            sy = self.sources.dec
            xlabel = "R.A."
            xlabel = "Decl."

        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        fig.suptitle("EXBA block | Q: %i | Ch: %i" % (self.quarter, self.channel))
        pc = ax.pcolormesh(
            x,
            y,
            self.flux_2d[0],
            shading="auto",
            norm=colors.SymLogNorm(linthresh=50, vmin=3, vmax=5000, base=10),
        )
        if sources:
            ax.scatter(
                sx,
                sy,
                s=20,
                facecolors="none",
                marker="o",
                edgecolors="r",
                label="Gaia Sources",
            )
        ax.set_xlabel("R.A. [deg]")
        ax.set_ylabel("Dec [deg]")
        fig.colorbar(pc, label=r"Flux ($e^{-}s^{-1}$)")
        ax.set_aspect("equal", adjustable="box")

        return ax

    def plot_lightcurves(self, object_id=None, step=10):

        if object_id is None:
            lcs = self.corrected_lcs if hasattr(self, "corrected_lcs") else self.sap_lcs
            sources = self.sources
            aperture_mask = self.aperture_mask
            step = step
        else:
            idx = np.in1d(self.sources.designation, np.array(object_id))
            sources = self.sources[idx]
            lcs = (
                self.corrected_lcs[idx]
                if hasattr(self, "corrected_lcs")
                else self.sap_lcs[idx]
            )
            aperture_mask = self.aperture_mask[idx]
            step = 1

        for s in range(0, len(sources), step):
            if aperture_mask[s].sum() == 0:
                print("Warning: zero pixels in aperture mask.")
                continue

            fig, ax = plt.subplots(
                1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [4, 1]}
            )

            lcs[s].plot(label=lcs[s].targetid, ax=ax[0])

            fig.suptitle(
                "EXBA block | Q: %i | Ch: %i | Source %i"
                % (self.quarter, self.channel, s)
            )
            pc = ax[1].pcolor(
                self.flux_2d[0],
                shading="auto",
                norm=colors.SymLogNorm(linthresh=50, vmin=3, vmax=5000, base=10),
            )
            ax[1].scatter(
                sources.col.iloc[s] - self.col.min(),
                sources.row.iloc[s] - self.row.min(),
                s=25,
                facecolors="r",
                marker="o",
                edgecolors="r",
            )
            ax[1].set_xlabel("Pixels")
            ax[1].set_ylabel("Pixels")
            fig.colorbar(pc, label=r"Flux ($e^{-}s^{-1}$)")
            ax[1].set_aspect("equal", adjustable="box")

            for i in range(self.ra_2d.shape[0]):
                for j in range(self.ra_2d.shape[1]):
                    if aperture_mask[s, i, j]:
                        rect = patches.Rectangle(
                            xy=(j, i),
                            width=1,
                            height=1,
                            color="red",
                            fill=False,
                            hatch="",
                        )
                        ax[1].add_patch(rect)
            zoom = np.argwhere(aperture_mask[s] == True)
            ax[1].set_ylim(
                np.maximum(0, zoom[0, 0] - 5),
                np.minimum(zoom[-1, 0] + 5, self.ra_2d.shape[0]),
            )
            ax[1].set_xlim(
                np.maximum(0, zoom[0, -1] - 5),
                np.minimum(zoom[-1, -1] + 5, self.ra_2d.shape[1]),
            )

            plt.show()


class EXBACollection(EXBA):
    def __init__(self, EXBAs):

        # check if each element of exba_quarters are EXBA objects
        # if not all([isinstance(exba, EXBA) for exba in EXBAs]):
        #     raise AssertionError("All elements of the list must be EXBA objects")

        self.channel = EXBAs[0].channel
        self.quarter = [exba.quarter for exba in EXBAs]

        # check that gaia sources are in all quarters
        gids = [exba.sources.designation.tolist() for exba in EXBAs]
        unique_gids = np.unique([item for sublist in gids for item in sublist])

        # create matris with index position to link sources across quarters
        # this asume that sources aren't in the same position in the DF, sources
        # can disapear (fall out the ccd), not all sources show up in all quarters.
        pm = np.empty((len(unique_gids), len(EXBAs)), dtype=np.int) * np.nan
        for q in range(len(EXBAs)):
            mask1 = np.in1d(EXBAs[q].sources.designation.values, unique_gids)
            mask2 = np.in1d(unique_gids, EXBAs[q].sources.designation.values)
            pm[mask2, q] = np.arange(len(EXBAs[q].sources.designation.values))[mask1]

        # rearange sources as list of lk Collection containing all quarters per source
        sources = []
        for i, gid in enumerate(unique_gids):
            aux = lk.LightCurveCollection(
                [
                    EXBAs[q].sap_lcs[int(pos)]
                    for q, pos in enumerate(pm[i])
                    if np.isfinite(pos)
                ]
            )
            sources.append(lk.LightCurveCollection(aux))
        self.source_lcs = sources

    def __repr__(self):
        q_result = ",".join([str(k) for k in list([self.quarter])])
        return "EXBA Patch:\n\t Channel %i, Quarter %s, Gaia sources %i" % (
            self.channel,
            q_result,
            len(self.source_lcs),
        )

    def stitch_quarters(self):

        # lk.LightCurveCollection.stitch() normalize by default all lcs before stitching
        if hasattr(self, "source_lcs"):
            self.stitched_lcs = lk.LightCurveCollection(
                [lc.stitch() for lc in self.source_lcs]
            )
        if hasattr(self, "corrected_lcs"):
            self.stitched_corrected_lcs = lk.LightCurveCollection(
                [lc.stitch() for lc in self.corrected_lcs]
            )

        return

    def apply_CBV(self, ignore_warnings=True, do_under=False):

        if ignore_warnings:
            warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Select which CBVs to use in the correction
        cbv_type = ["SingleScale"]
        # Select which CBV indices to use
        # Use the first 8 SingleScale and all Spike CBVS
        cbv_indices = [np.arange(1, 9)]

        over_fit_m = []
        under_fit_m = []
        corrected_lcs = []

        for source_lc in tqdm(self.source_lcs, desc="Gaia sources"):
            tmp, tmp_over, tmp_under = [], [], []
            for i, lc in enumerate(source_lc):
                lc = lc[lc.flux_err > 0]
                cbvcor = CBVCorrector(lc)
                cbvcor.correct_gaussian_prior(
                    cbv_type=cbv_type, cbv_indices=cbv_indices, alpha=1e-4
                )
                tmp_over.append(cbvcor.over_fitting_metric())
                if do_under:
                    tmp_under.append(cbvcor.under_fitting_metric())
                tmp.append(cbvcor.corrected_lc)
            tmp = lk.LightCurveCollection(tmp)
            corrected_lcs.append(tmp)
            over_fit_m.append(np.array(tmp_over))
            if do_under:
                under_fit_m.append(np.array(tmp_under))
        self.corrected_lcs = corrected_lcs
        self.over_fitting_metrics = np.asarray(over_fit_m, dtype=object)
        if do_under:
            self.under_fitting_metrics = np.asarray(under_fit_m, dtype=object)
        return


class EXBALightCurveCollection:
    def __init__(self, lcs, metadata):

        # check if each element of exba_quarters are EXBA objects
        # if not all([isinstance(exba, EXBA) for exba in EXBAs]):
        #     raise AssertionError("All elements of the list must be EXBA objects")

        self.channel = np.unique([lc[0].channel for lc in lcs])
        self.quarter = np.unique([lc[0].quarter for lc in lcs])

        # check that gaia sources are in all quarters
        gids = [df.designation.tolist() for df in metadata]
        unique_gids = np.unique([item for sublist in gids for item in sublist])

        # create matris with index position to link sources across quarters
        # this asume that sources aren't in the same position in the DF, sources
        # can disapear (fall out the ccd), not all sources show up in all quarters.
        pm = np.empty((len(unique_gids), len(lcs)), dtype=np.int) * np.nan
        for q in range(len(lcs)):
            mask1 = np.in1d(metadata[q].designation.values, unique_gids)
            mask2 = np.in1d(unique_gids, metadata[q].designation.values)
            pm[mask2, q] = np.arange(len(metadata[q].designation.values))[mask1]

        # rearange sources as list of lk Collection containing all quarters per source
        sources = []
        for i, gid in enumerate(unique_gids):
            aux = lk.LightCurveCollection(
                [lcs[q][int(pos)] for q, pos in enumerate(pm[i]) if np.isfinite(pos)]
            )
            sources.append(lk.LightCurveCollection(aux))
        self.lcs = sources
        self.metadata = (
            pd.concat(metadata, axis=0, join="outer")
            .drop_duplicates(["designation"], ignore_index=True)
            .drop(["Unnamed: 0", "col", "row"], axis=1)
        )

    def __repr__(self):
        ch_result = ",".join([str(k) for k in list([self.channel])])
        q_result = ",".join([str(k) for k in list([self.quarter])])
        return (
            "Light Curves from: \n\tChannel %s \n\tQuarter %s \n\tGaia sources %i"
            % (
                ch_result,
                q_result,
                len(self.lcs),
            )
        )

    def stitch_quarters(self):

        # lk.LightCurveCollection.stitch() normalize by default all lcs before stitching
        if hasattr(self, "source_lcs"):
            self.stitched_lcs = lk.LightCurveCollection(
                [lc.stitch() for lc in self.source_lcs]
            )

        return

    def do_bls_search(self, plot=False, n_boots=100, fap_tresh=0.1):

        self.metadata["has_planet"] = False
        self.metadata["N_periodic_quarters"] = 0
        self.metadata["Period"] = None
        self.metadata["Period_snr"] = None
        self.metadata["Period_fap"] = None

        for lc_long in tqdm(self.lcs, desc="Gaia sources"):

            periods, faps, snrs = get_bls_periods(lc_long, plot=plot, n_boots=n_boots)
            print(periods, faps, snrs)
            # check for significant periodicity detection in at least one quarter
            if np.isfinite(faps).all():
                p_mask = faps < fap_tresh
            else:
                p_mask = snrs > 50
            if p_mask.sum() > 0:
                # check that periods are similar, within a tolerance
                # this assumes that there's one only one period
                # have to fix this to make it work for multiple true periods detected
                # or if BLS detected ane of the armonics, not necesary yet, when need it
                # use np.round(periods) and np.unique() to check for same periods and
                # harmonics within tolerance.
                good_periods = periods[p_mask]
                good_faps = faps[p_mask]
                good_snrs = snrs[p_mask]
                all_close = (
                    np.array(
                        [np.isclose(p, good_periods, atol=0.1) for p in good_periods]
                    ).sum(axis=0)
                    > 1
                )
                if all_close.sum() > 1:
                    idx = np.where(self.metadata.designation == lc_long[0].label)[0]
                    self.metadata["has_planet"].iloc[idx] = True
                    self.metadata["N_periodic_quarters"].iloc[idx] = all_close.sum()
                    self.metadata["Period"].iloc[idx] = good_periods[all_close].mean()
                    self.metadata["Period_fap"].iloc[idx] = good_faps[all_close].mean()
                    self.metadata["Period_snr"].iloc[idx] = good_snrs[all_close].mean()
                # check if periods are harmonics
            break

    @staticmethod
    def from_stored_lcs(channels, quarters):

        # load gaia catalogs and lcs
        metadata, lcs, nodata = [], [], []
        for ch in channels:
            for q in quarters:
                df_path = "%s/data/EXBA/%i/%i/gaia_dr2_xmatch.csv" % (main_path, ch, q)
                lc_path = "%s/data/EXBA/%i/%i/sap_lcs_CBVcorrected.pkl" % (
                    main_path,
                    ch,
                    q,
                )
                if not os.path.isfile(df_path) or not os.path.isfile(lc_path):
                    print(
                        "WARNING: channel %i quarter %i have no storaged files"
                        % (ch, q)
                    )
                    nodata.append([ch, q])
                    continue
                df = pd.read_csv(df_path)
                lc = cPickle.load(
                    open(
                        lc_path,
                        "rb",
                    )
                )
                metadata.append(df)
                lcs.append(lc)

        return EXBALightCurveCollection(lcs, metadata)
