"""
Image perturbations modelling conveyor-belt failure modes.

Four perturbation families, each parameterised by five severity levels:

    motion_blur       fast-moving objects under a finite shutter time
    gaussian_noise    camera sensor noise
    brightness        lighting variation (progressive darkening)
    jpeg_compression  a low-quality or heavily compressed camera feed

Shared by notebook 06 (robustness testing) and notebook 12 (noise feature
analysis) so both apply identical degradation.

Reproducibility
---------------
Gaussian noise is the only stochastic perturbation. It draws from an explicitly
seeded `numpy.random.Generator` rather than the global NumPy random state, so a
given (severity, seed) pair reproduces the same noise field on every run.
Use `make_perturbation` to obtain a fresh, independently seeded callable.
"""

import io

import cv2
import numpy as np
from PIL import Image

RANDOM_STATE = 42

SEVERITIES = [1, 2, 3, 4, 5]

MOTION_BLUR_KERNELS = [3, 7, 11, 15, 21]
GAUSSIAN_NOISE_STDS = [5, 15, 30, 50, 75]
BRIGHTNESS_FACTORS = [0.8, 0.6, 0.45, 0.3, 0.15]
JPEG_QUALITIES = [80, 60, 40, 20, 5]


def apply_motion_blur(image, severity):
    """Horizontal box blur. Severity controls kernel size; higher = more blur."""
    img_array = np.array(image)
    kernel_size = MOTION_BLUR_KERNELS[severity - 1]
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    blurred = cv2.filter2D(img_array, -1, kernel)
    return Image.fromarray(blurred)


def apply_gaussian_noise(image, severity, rng=None):
    """
    Additive Gaussian sensor noise. Severity controls the noise standard deviation.

    `rng` must be a numpy.random.Generator for reproducible output. When omitted
    a generator seeded with RANDOM_STATE is created per call, which keeps single
    calls deterministic but is not suitable for sweeping a whole dataset -- use
    `make_perturbation` for that.
    """
    rng = np.random.default_rng(RANDOM_STATE) if rng is None else rng

    img_array = np.array(image).astype(np.float32)
    noise_std = GAUSSIAN_NOISE_STDS[severity - 1]
    noise = rng.normal(0, noise_std, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_brightness(image, severity):
    """Progressive darkening. Severity controls the brightness multiplier."""
    img_array = np.array(image).astype(np.float32)
    factor = BRIGHTNESS_FACTORS[severity - 1]
    darkened = np.clip(img_array * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(darkened)


def apply_jpeg_compression(image, severity):
    """JPEG re-encoding artefacts. Severity controls the quality setting."""
    quality = JPEG_QUALITIES[severity - 1]
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


PERTURBATIONS = {
    'motion_blur':      apply_motion_blur,
    'gaussian_noise':   apply_gaussian_noise,
    'brightness':       apply_brightness,
    'jpeg_compression': apply_jpeg_compression,
}

PERTURBATION_TITLES = {
    'motion_blur':      'Motion Blur',
    'gaussian_noise':   'Gaussian Noise',
    'brightness':       'Brightness Reduction',
    'jpeg_compression': 'JPEG Compression',
}


def make_perturbation(name, severity, seed=RANDOM_STATE):
    """
    Build a single-argument callable applying `name` at `severity` to a PIL image.

    Stochastic perturbations receive their own Generator seeded from
    (seed, severity), so each severity sweep is reproducible and independent.
    """
    fn = PERTURBATIONS[name]

    if fn is apply_gaussian_noise:
        rng = np.random.default_rng([seed, severity])
        return lambda image: fn(image, severity, rng=rng)

    return lambda image: fn(image, severity)
