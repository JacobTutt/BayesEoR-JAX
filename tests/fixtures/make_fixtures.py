"""Regenerate the committed test fixtures in ``tests/fixtures``.

Run from the repository root with ``python tests/fixtures/make_fixtures.py``. The
fixtures are deterministic, small (a few hundred kB) and describe a tiny drift
scan by a four-antenna array. Only re-run this when the fixture definition
changes; the expected end-to-end values must then be regenerated too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.coordinates import EarthLocation
from astropy import units
from pyuvdata import Telescope, UVData, utils as uvutils
from pyuvdata.analytic_beam import AiryBeam

DATA_DIR = Path(__file__).resolve().parent

#: HERA site, as [latitude (deg), longitude (deg), altitude (m)].
TELESCOPE_LATLONALT = (-30.72152777777791, 21.428305555555557, 1073.0000000093132)
#: Four antennas on a 14.6 m grid: (0,1) and (2,3) are a redundant pair.
ANTENNA_ENU_M = np.array(
    [[0.0, 0.0, 0.0], [14.6, 0.0, 0.0], [7.3, 12.644, 0.0], [21.9, 12.644, 0.0]]
)
FREQS_HZ = 150.0e6 + 100.0e3 * np.arange(4)
JD_START = 2459999.1
N_TIMES = 4
INTEGRATION_S = 10.0
SEED = 20261024


def _telescope() -> Telescope:
    """Return the four-antenna telescope at the HERA site."""
    lat, lon, alt = TELESCOPE_LATLONALT
    location = EarthLocation.from_geodetic(lon * units.deg, lat * units.deg, alt * units.m)
    ecef = uvutils.ECEF_from_ENU(
        ANTENNA_ENU_M, latitude=np.deg2rad(lat), longitude=np.deg2rad(lon), altitude=alt
    )
    centre = np.array([location.x.to_value("m"), location.y.to_value("m"), location.z.to_value("m")])
    return Telescope.new(
        name="TEST4",
        instrument="TEST4",
        location=location,
        antenna_positions={i: ecef[i] - centre for i in range(len(ANTENNA_ENU_M))},
        antenna_diameters=np.full(len(ANTENNA_ENU_M), 14.0),
        x_orientation="east",
        update_from_known=False,
    )


def make_uvh5(path: Path) -> UVData:
    """Write a calibrated, unprojected four-antenna drift scan in Jy, with autos."""
    rng = np.random.default_rng(SEED)
    antpairs = [(i, j) for i in range(4) for j in range(i, 4)]
    times = JD_START + INTEGRATION_S / 86400.0 * np.arange(N_TIMES)
    uvd = UVData.new(
        freq_array=FREQS_HZ,
        polarization_array=["xx", "yy"],
        times=times,
        telescope=_telescope(),
        antpairs=antpairs,
        do_blt_outer=True,
        integration_time=INTEGRATION_S,
        channel_width=100.0e3,
        vis_units="Jy",
        empty=True,
        update_telescope_from_known=False,
    )
    shape = uvd.data_array.shape
    uvd.data_array = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(complex)
    # Autocorrelations are real and positive, as in a real correlator.
    autos = uvd.ant_1_array == uvd.ant_2_array
    uvd.data_array[autos] = np.abs(uvd.data_array[autos]) + 10.0
    uvd.write_uvh5(str(path), clobber=True)
    return uvd


def make_beamfits(path: Path) -> None:
    """Write a coarse az/za Airy electric-field beam covering the fixture band."""
    beam = AiryBeam(diameter=14.0).to_uvbeam(
        freq_array=np.linspace(149.5e6, 151.0e6, 4),
        beam_type="efield",
        axis1_array=np.deg2rad(np.arange(0.0, 360.0, 30.0)),
        axis2_array=np.deg2rad(np.arange(0.0, 91.0, 5.0)),
    )
    beam.write_beamfits(str(path), clobber=True)


def make_preprocessed(uvh5_path: Path, out_dir: Path) -> None:
    """Write the ``.npy`` vector and instrument model for the cross-correlations."""
    from bayeseor_jax.data.vis import preprocess_uvdata

    preprocess_uvdata(
        uvh5_path, save_vis=True, save_model=True, save_dir=out_dir, clobber=True,
    )


def main() -> None:
    """Regenerate every fixture file."""
    uvh5 = DATA_DIR / "tiny.uvh5"
    make_uvh5(uvh5)
    make_beamfits(DATA_DIR / "airy_efield.beamfits")
    make_preprocessed(uvh5, DATA_DIR / "tiny_npy")
    print(f"Fixtures written to {DATA_DIR}")


if __name__ == "__main__":
    main()
