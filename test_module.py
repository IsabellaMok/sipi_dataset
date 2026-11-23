import numpy as np
from dataset import load_dataset
from iwt import iwt2, iiwt2
import hamming1511 as hm
import metrics as mt

rng = np.random.default_rng(42)
images = load_dataset()   # Baboon, Peppers, Sailboat, Splash, Airplane

print(f"{'Image':<10}{'bits':<8}{'bpp':<8}{'PSNR':<9}{'SSIM':<9}{'BER':<6}{'out-of-range'}")
for name, img in images.items():
    LL, HL, LH, HH = iwt2(img)
    assert np.array_equal(iiwt2(LL, HL, LH, HH), img), f"IWT not lossless on {name}"

    flat = HH.ravel()
    lsb = (flat & 1).astype(np.uint8)
    n_bits = (len(lsb) // 15) * 4
    secret = rng.integers(0, 2, n_bits).astype(np.uint8)

    new_flat = (flat & ~1) | hm.embed(lsb, secret)
    stego_raw = iiwt2(LL, HL, LH, new_flat.reshape(HH.shape))
    oor = int(np.sum((stego_raw < 0) | (stego_raw > 255)))
    stego = np.clip(stego_raw, 0, 255).astype(np.uint8)

    # extraction: transform the stego image again and read the HH LSBs
    _, _, _, HH2 = iwt2(stego)
    got = hm.extract((HH2.ravel() & 1).astype(np.uint8), n_bits)

    print(f"{name:<10}{n_bits:<8}{mt.bpp(n_bits, img.shape):<8.3f}"
          f"{mt.psnr(img, stego):<9.2f}{mt.ssim(img, stego):<9.4f}"
          f"{mt.ber(secret, got):<6.3f}{oor}")