"""Evaluation metrics: MSE, PSNR, SSIM, BER, bpp."""
import numpy as np
from skimage.metrics import structural_similarity

def mse(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))

def psnr(a, b, peak=255.0):
    m = mse(a, b)
    return float("inf") if m == 0 else float(10 * np.log10(peak ** 2 / m))

def ssim(a, b):
    return float(structural_similarity(a, b, data_range=255))

def ber(bits_a, bits_b):
    """Bit error rate between two equal-length bit arrays."""
    a = np.asarray(bits_a).ravel(); b = np.asarray(bits_b).ravel()
    assert a.shape == b.shape, "bit arrays must have equal length"
    return float(np.mean(a != b))

def bpp(n_payload_bits, image_shape):
    """Embedding capacity in bits per pixel."""
    return n_payload_bits / (image_shape[0] * image_shape[1])