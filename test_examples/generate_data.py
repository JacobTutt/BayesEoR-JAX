"""Generate the small laptop datasets locally from GSM and point-source skyh5 files.

Only selected visibilities and baseline metadata are saved. The external sky
maps are needed to regenerate the examples, not to run the checked-in examples.
"""

import argparse
import json
from pathlib import Path

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers
import h5py
import jax
jax.config.update("jax_enable_x64", True)
import jax_healpy as hp
import numpy as np
from scipy.special import j1

from bayeseor_jax.model.observation.image_geometry import HERA_LOCATION

# Central 15 channels/integrations of the original 39-channel, 47-time observation.
FREQUENCIES = 158398437.5 + np.arange(15) * 97656.25
TIMES = 2459999.097166265 + np.arange(-7, 8) * 21.511369943618774 / 86400
# All baseline groups below 40 m, in the original instrument's ordering.
# The other 15 signed groups are their conjugates, not independent measurements.
BASELINES = np.array([
    [2, 0, 0], [4, 0, 0], [1, -1, 0], [3, -1, 0], [5, -1, 0],
    [1, 1, 0], [3, 1, 0], [5, 1, 0], [2, -2, 0], [4, -2, 0],
    [0, 2, 0], [2, 2, 0], [4, 2, 0], [1, -3, 0], [1, 3, 0],
]) * [7.3, 7.3*np.sqrt(3), 1]
MASK_NSIDE = 64


def observation_frames():
    """Return the topocentric coordinate frame at each integration centre."""
    iers.conf.auto_download = False
    lat, lon, height = HERA_LOCATION
    return AltAz(obstime=Time(TIMES, format="jd"),
                 location=EarthLocation.from_geodetic(lon*u.deg, lat*u.deg, height*u.m))


def build_footprint():
    """Select Nside=64 pixel centres within 5 degrees of zenith at any time.

    Returns
    -------
    numpy.ndarray
        Boolean RING-ordered mask on the inference grid. All higher-resolution
        pixels and point sources inherit the footprint of these coarse pixels;
        no additional angular cut is applied at their own resolution.
    """
    centres = SkyCoord(az=np.zeros(len(TIMES))*u.deg,
                       alt=np.full(len(TIMES), 90)*u.deg, frame=observation_frames()).icrs
    coarse_sky = healpix_sky(MASK_NSIDE)
    return np.any(coarse_sky.cartesian.xyz.value.T @ centres.cartesian.xyz.value
                  >= np.cos(np.deg2rad(5)), axis=1)


def select_directions(sky, footprint):
    """Select directions by their coarse parent pixels and transform to ENU.

    Parameters
    ----------
    sky : astropy.coordinates.SkyCoord
        Fixed ICRS directions of pixels or point sources.
    footprint : numpy.ndarray
        Boolean Nside=64 RING mask from build_footprint.

    Returns
    -------
    indices, lmn : numpy.ndarray
        Selected source indices and topocentric unit vectors in the shared
        coarse-pixel footprint. Celestial identities remain fixed across time.
    """
    footprint = np.asarray(footprint)
    if footprint.shape != (12*MASK_NSIDE**2,) or footprint.dtype != bool:
        raise ValueError("footprint must be a boolean Nside=64 RING mask")
    parent = np.asarray(hp.ang2pix(MASK_NSIDE, np.pi/2-sky.icrs.dec.rad, sky.icrs.ra.rad))
    indices = np.flatnonzero(footprint[parent])
    horizontal = sky[indices][None, :].transform_to(observation_frames()[:, None])
    az, alt = horizontal.az.rad, horizontal.alt.rad
    lmn = np.stack((np.cos(alt)*np.sin(az), np.cos(alt)*np.cos(az), np.sin(alt)), axis=-1)
    return indices, lmn


def healpix_sky(nside):
    """Return ICRS directions for a RING-ordered HEALPix sky."""
    theta, phi = hp.pix2ang(nside, np.arange(12*nside**2))
    return SkyCoord(ra=np.asarray(phi)*u.rad, dec=(np.pi/2-np.asarray(theta))*u.rad)


def integrate(lmn, brightness_sr):
    """Integrate an Airy-weighted sky independently of the inference operators.

    Parameters
    ----------
    lmn : numpy.ndarray
        ENU unit directions, shape (time, source, 3).
    brightness_sr : numpy.ndarray
        Brightness times pixel solid angle, or converted point-source flux,
        in mK sr with shape (frequency, source).

    Returns
    -------
    numpy.ndarray
        Complex visibilities in (time, frequency, baseline) order. Each block
        contains the 15 physical baselines followed by their conjugates.
    """
    result = np.empty((len(lmn), len(FREQUENCIES), 2*len(BASELINES)), dtype=np.complex128)
    for t, directions in enumerate(lmn):
        sin_za = np.sqrt(np.clip(1-directions[:, 2]**2, 0, 1))
        for f, frequency in enumerate(FREQUENCIES):
            argument = np.pi * 14.2 * frequency / 299792458 * sin_za
            voltage = np.ones_like(argument)
            np.divide(2*j1(argument), argument, out=voltage, where=argument != 0)
            beam = voltage**2 * (directions[:, 2] > 0)
            # Baselines are ant2-ant1, matching the positive-phase visibility convention.
            fringe = np.exp(2j*np.pi*frequency/299792458 * (BASELINES @ directions.T))
            visibility = fringe @ (beam * brightness_sr[f])
            result[t, f] = np.r_[visibility, visibility.conj()]
    return result.reshape(-1)


def generate(gsm_path, point_path, output):
    """Write noiseless 21-cm and joint datasets plus shared observation metadata."""
    footprint = build_footprint()
    with h5py.File(gsm_path) as sky:
        assert sky['Header/hpx_frame/frame'][()] == b'icrs'
        assert sky['Header/hpx_order'][()] == b'ring'
        assert sky['Data/stokes'].attrs['unit'] == 'K'
        nside = int(sky['Header/nside'][()])
        indices, directions = select_directions(healpix_sky(nside), footprint)
        stored_pixels = sky['Header/hpx_inds'][:]
        assert np.array_equal(stored_pixels, np.arange(12*nside**2))
        channels = np.array([np.argmin(abs(sky['Header/freq_array'][:]-f)) for f in FREQUENCIES])
        np.testing.assert_allclose(sky['Header/freq_array'][:][channels], FREQUENCIES)
        brightness = np.stack([sky['Data/stokes'][0, int(f), indices] for f in channels])
        foreground = integrate(directions, brightness * 1000 * 4*np.pi/(12*nside**2))
        gsm_history = sky['Header/history'][()].decode()

    with h5py.File(point_path) as sky:
        assert sky['Header/skycoord/frame'][()] == b'icrs'
        assert sky['Data/stokes'].attrs['unit'] == 'Jy'
        coords = SkyCoord(ra=sky['Header/skycoord/ra'][:]*u.deg,
                          dec=sky['Header/skycoord/dec'][:]*u.deg)
        indices, directions = select_directions(coords, footprint)
        flux = sky['Data/stokes'][0, 0, indices]
        reference = sky['Header/reference_frequency'][:][indices]
        alpha = sky['Header/spectral_index'][:][indices]
        flux = flux[None, :] * (FREQUENCIES[:, None]/reference)**alpha
        jy_to_mk_sr = 1e-23 * 299792458**2 / (2*1.380649e-23*FREQUENCIES**2)
        foreground += integrate(directions, flux * jy_to_mk_sr[:, None])
        point_history = sky['Header/history'][()].decode()

    nside_21, seed, sigma_21 = 256, 92381923, 6.4
    indices, directions = select_directions(healpix_sky(nside_21), footprint)
    rng = np.random.default_rng(seed)
    # Draw one full-sky channel at a time so the generation stays small in memory.
    brightness = []
    for _ in FREQUENCIES:
        channel = rng.normal(0, sigma_21, 12*nside_21**2)
        brightness.append(channel[indices] - channel.mean())
    signal = integrate(directions, np.asarray(brightness) * 4*np.pi/(12*nside_21**2))

    for name, data in (("eor", signal), ("eor_fgs", signal + foreground)):
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "visibilities.npy", data)
    instrument = output / "instrument"
    instrument.mkdir(parents=True, exist_ok=True)
    baselines = np.concatenate((BASELINES, -BASELINES))
    np.save(instrument / "uvw_model.npy", np.broadcast_to(baselines, (len(TIMES), len(baselines), 3)))
    np.save(instrument / "redundancy_model.npy", np.ones((len(TIMES), len(baselines))))
    metadata = dict(
        description="Local 10-degree time-union simulation; no horizon/full-sky recovery claim.",
        frequency_hz=FREQUENCIES.tolist(), jd=TIMES.tolist(), field_diameter_deg=10,
        mask_nside=MASK_NSIDE, mask_ring_pixels=np.flatnonzero(footprint).tolist(),
        mask_definition="Select coarse pixel centres within 5 degrees of zenith at any integration; retain all child pixels and point sources inside those coarse pixels.",
        beam="14.2-m Airy power beam", telescope_latlonalt=list(HERA_LOCATION),
        units="mK sr", row_order="time, frequency, baseline; positive then conjugate",
        noiseless=True, gsm_nside=nside, signal_nside=nside_21,
        signal_seed=seed, signal_pixel_std_mk=sigma_21,
        signal_spectrum="Independent pixels/channels: P = sigma_21^2 times comoving voxel volume",
        gsm_history=gsm_history, point_source_history=point_history,
    )
    (instrument / "simulation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved two {signal.size}-element visibility vectors under {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm-sky", required=True, type=Path)
    parser.add_argument("--point-sky", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Destination example directory; existing products are replaced.")
    args = parser.parse_args()
    generate(args.gsm_sky, args.point_sky, args.output)
