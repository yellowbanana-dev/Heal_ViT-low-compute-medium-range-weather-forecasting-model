"""
Zonal Energy Spectrum computation.

Adapted from WeatherBench2's ZonalEnergySpectrum for use with NumPy arrays
on the HEAL-ViT regular lat-lon grid (121 lat x 240 lon).

Reference: WeatherBench2 — https://github.com/google-research/weatherbench2

Given dataset values f[l], l = 0,..., L-1 along a zonal circle of constant
latitude with circumference C (m), the DFT is:
    F[k] = (1/L) * sum_l f[l] exp(-i 2pi k l / L)

The energy spectrum is:
    S[0] = C |F[0]|^2,
    S[k] = 2C |F[k]|^2,  k > 0   (accounts for pos & neg frequencies)

This normalisation ensures Parseval's relation:
    integral |f(x)|^2 dx  ~  (C/L) sum_l |f[l]|^2  =  sum_k S[k].
"""

import numpy as np

EARTH_RADIUS_M = 6_371_229.0  # metres (WB2 default)


def zonal_energy_spectrum(field, lat, lon):
    """Compute the zonal energy spectrum for a spatial field.

    Parameters
    ----------
    field : np.ndarray, shape (..., n_lat, n_lon)
        Input field values.  Last two axes must be (latitude, longitude).
    lat : np.ndarray, shape (n_lat,)
        Latitude values in degrees (descending, 90 -> -90).
    lon : np.ndarray, shape (n_lon,)
        Longitude values in degrees (ascending, 0 -> 360, endpoint=False).

    Returns
    -------
    spectrum : np.ndarray, shape (..., n_lat, n_wavenumber)
        Zonal energy spectrum at each latitude.
        n_wavenumber = n_lon // 2 + 1.
    wavenumbers : np.ndarray, shape (n_wavenumber,)
        Integer zonal wavenumber indices 0 ... n_lon // 2.
    """
    # Circumference of each zonal circle
    circum_equator = 2 * np.pi * EARTH_RADIUS_M
    circum = np.cos(np.deg2rad(lat)) * circum_equator          # (n_lat,)

    # Forward DFT along longitude (last axis), normalised by 1/N
    f_k = np.fft.rfft(field, axis=-1, norm='forward')
    n_wn = f_k.shape[-1]

    # One-sided power; double k > 0 to account for negative frequencies
    power = np.real(f_k * np.conj(f_k))
    multiplier = np.ones(n_wn)
    multiplier[1:] = 2.0
    power = power * multiplier

    # Scale by circumference -> units become [field_unit]^2 * m
    spectrum = power * circum[..., np.newaxis]

    wavenumbers = np.arange(n_wn)
    return spectrum, wavenumbers


def latitude_averaged_spectrum(field, lat, lon):
    """Latitude-weighted (cos phi) average of the zonal energy spectrum.

    Parameters
    ----------
    field : np.ndarray, shape (..., n_lat, n_lon)
    lat, lon : coordinate arrays

    Returns
    -------
    avg_spectrum : np.ndarray, shape (..., n_wavenumber)
    wavenumbers  : np.ndarray, shape (n_wavenumber,)
    """
    spectrum, wavenumbers = zonal_energy_spectrum(field, lat, lon)

    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.sum()

    # Weighted mean over latitude axis (second-to-last)
    avg_spectrum = np.sum(spectrum * weights[..., np.newaxis], axis=-2)

    return avg_spectrum, wavenumbers
