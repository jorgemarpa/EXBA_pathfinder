"""Tools to stitch EXBA images and extract LC of detected Gaia sources"""
import os, glob

import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm.notebook import tqdm
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, match_coordinates_3d
from astropy.stats import sigma_clip

from .utils import get_gaia_sources

main_path = os.path.dirname(os.getcwd())


class EXBA(object):
    def __init__(self, channel=53, quarter=5, path=main_path):

        self.quarter = quarter
        self.channel = channel

        # load local TPFs files
        tpfs_paths = np.sort(
            glob.glob("%s/data/EXBA/%s/%s/*_lpd-targ.fits.gz" %
                      (main_path, channel, quarter))
        )

        tpfs = lk.TargetPixelFileCollection(
            [lk.KeplerTargetPixelFile(f) for f in tpfs_paths[:4]]
        )
        print(tpfs)
        # check for same channels and quarter
        channels = [tpf.get_header()["CHANNEL"] for tpf in tpfs]
        quarters = [tpf.get_header()["QUARTER"] for tpf in tpfs]
        target_ids = [tpf.get_header()["OBJECT"] for tpf in tpfs]
        ra_objects = [tpf.get_header()["RA_OBJ"] for tpf in tpfs]
        dec_objects = [tpf.get_header()["DEC_OBJ"] for tpf in tpfs]

        if len(set(channels)) != 1 and list(set(channels)) != [channel]:
            raise ValueError("All TPFs must be from the same channel %i" % channel)

        if quarter != "all":
            if list(set(quarters)) != [quarter]:
                raise ValueError("TPFs are from the wrong quarter")
        else:
            if list(set(quarters)) != [5, 9, 13, 17]:
                raise ValueError(
                    "Not all quarters are available " + list(set(quarters))
                )

        # stich channel's strips and parse TPFs
        self.time, row, col, flux, flux_err, unw = self._parse_TPFs_channel(tpfs)
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
        self.ra_2d, self.dec_2d = (self.ra.reshape(self.row_2d.shape),
                                   self.dec.reshape(self.row_2d.shape))

        # search Gaia sources in the sky
        sources = self._get_coord_and_query_gaia(
            self.ra, self.dec, self.unw, self.time[0], magnitude_limit=25
        )
        sources['col'], sources['row'] = tpfs[0].wcs.wcs_world2pix(sources.ra,
                                                                   sources.dec, 0.5)
        sources['col'] += tpfs[0].column
        sources['row'] += tpfs[0].row
        self.sources, self.bad_sources = self._clean_source_list(sources,
                                                            self.ra, self.dec)


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
        times = tpfs[0].astropy_time.jd

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

        return times, row.T, col.T, flux, flux_err, unw

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
        sources.loc[:, "clean_flag"].iloc[~inside] += 2 ** 0  # outside TPF
        sources.loc[:, "clean_flag"].iloc[unresolved] += 2 ** 1  # close contaminant

        # combine 2 source masks
        clean = sources.clean_flag == 0
        removed_sources = sources[~clean].reset_index(drop=True)
        sources = sources[clean].reset_index(drop=True)

        return sources, removed_sources


    def simple_aperture_phot(self, space='pix-sq'):
        if space == 'world':
            aper = (u.arcsecond * 2 * 4).to(u.deg).value # aperture radii in deg
            aperture_mask = [np.hypot(self.ra - s.ra, self.dec - s.dec) < aper
                             for _, s in self.sources.iterrows()]
        elif space == 'pix-cir':
            aper = 1.7
            aperture_mask = [np.hypot(self.col - s.col, self.row - s.row) < aper
                             for _, s in self.sources.iterrows()]
        elif space == 'pix-sq':
            aper = [1.5, 1.5]
            aperture_mask = [(np.abs(self.col - np.floor(s.col)) < aper[1]) &
                             (np.abs(self.row - np.floor(s.row)) < aper[0])
                             for _, s in self.sources.iterrows()]

        sap = np.zeros((self.sources.shape[0], self.flux.shape[0]))
        sap_e = np.zeros((self.sources.shape[0], self.flux.shape[0]))

        for tdx in tqdm(range(len(self.flux)), desc='Simple SAP flux'):
            sap[:, tdx] = [self.flux[tdx][mask].value.sum() for mask in aperture_mask]
            sap_e[:, tdx] = [np.power(self.flux_err[tdx][mask].value, 2).sum() ** 0.5
                             for mask in aperture_mask]

        self.aperture_mask = np.asarray(aperture_mask).reshape(self.sources.shape[0],
                                                               self.flux_2d.shape[1],
                                                               self.flux_2d.shape[2])
        self.sap_lcs = [lk.LightCurve(time=self.time, flux=sap[i], flux_err=sap_e[i],
                          time_format='bkjd', flux_unit='electron/s',
                          targetid=self.sources.designation[i]).remove_outliers(sigma=3)
                         for i in range(len(sap))]
        return

    def plot_image(self, space="pixels", sources=True, **kwargs):
        if space == "pixels":
            x = self.col_2d
            y = self.row_2d
            sx = self.sources.col
            sy = self.sources.row
            xlabel = 'Pixel Column Number'
            xlabel = 'Pixel Row Number'
        elif space == "wcs":
            x = self.ra_2d
            y = self.dec_2d
            sx = self.sources.ra
            sy = self.sources.dec
            xlabel = 'R.A.'
            xlabel = 'Decl.'

        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        fig.suptitle(
            "EXBA block | Q: %i | Ch: %i"
            % (self.quarter, self.channel)
        )
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

    def plot_lightcurves(self, apertures=None):

        for s in range(0, len(sources), 5):
            if self.aperture_mask[s].sum() == 0:
                print("Warning: zero pixels in aperture mask.")
                continue
            fig, ax = plt.subplots(1, 2, figsize=(15,4),
                                   gridspec_kw={'width_ratios': [4, 1]})

            self.exba_lcs[s].plot(label=self.exba_lcs[s].targetid, ax=ax[0])

            fig.suptitle('EXBA block | Q: %i | Ch: %i' %
                         (self.quarter, self.channel))
            pc = ax[1].pcolor(self.flux_2d[0], shading='auto',
                               norm=colors.SymLogNorm(linthresh=50, vmin=3,
                                                      vmax=5000, base=10))
            ax[1].scatter(self.sources.col[s], self.sources.row[s], s=50,
                          facecolors='none', marker='o', edgecolors='r')
            ax[1].set_xlabel('Pixels')
            ax[1].set_ylabel('Pixels')
            fig.colorbar(pc, label=r"Flux ($e^{-}s^{-1}$)")
            ax[1].set_aspect('equal', adjustable='box')

            for i in range(self.ra_2d.shape[0]):
                for j in range(self.ra_2d.shape[1]):
                    if self.aperture_mask[s, i, j]:
                        rect = patches.Rectangle(
                                        xy=(j, i),
                                        width=1, height=1, color='red',
                                        fill=False, hatch='')
                        ax[1].add_patch(rect)
            zoom = np.argwhere(self.aperture_mask[s] == True)
            ax[1].set_ylim(np.maximum(0, zoom[0,0] - 5),
                           np.minimum(zoom[-1,0] + 5, self.ra_2d.shape[0]))
            ax[1].set_xlim(np.maximum(0, zoom[0,-1] - 5),
                           np.minimum(zoom[-1,-1] + 5, self.ra_2d.shape[1]))

            plt.show()
