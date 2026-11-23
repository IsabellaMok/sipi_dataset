import numpy as np

# Parity-check matrix H (4x15): column j (1-indexed) = binary representation of j
H = np.array([[(j >> k) & 1 for j in range(1, 16)] for k in range(4)], dtype=np.uint8)

def _syndrome(block):
    return H.dot(block) % 2                                  # 4-bit vector

def _to_int(bits4):
    return int(sum(int(b) << k for k, b in enumerate(bits4)))

def embed_block(cover15, secret4):
    c = cover15.copy().astype(np.uint8)
    d = _to_int((_syndrome(c) + secret4) % 2)                # syndrome XOR secret
    if d != 0:
        c[d - 1] ^= 1                                        # flip the single position d
    return c

def extract_block(stego15):
    return _syndrome(stego15.astype(np.uint8))

def embed(cover_bits, secret_bits):
    cover_bits = np.asarray(cover_bits, np.uint8); secret_bits = np.asarray(secret_bits, np.uint8)
    n_blocks = len(secret_bits) // 4
    assert len(cover_bits) >= 15 * n_blocks, "not enough cover bits"
    out = cover_bits.copy()
    for i in range(n_blocks):
        out[15*i:15*i+15] = embed_block(out[15*i:15*i+15], secret_bits[4*i:4*i+4])
    return out

def extract(stego_bits, n_secret_bits):
    stego_bits = np.asarray(stego_bits, np.uint8)
    return np.concatenate([extract_block(stego_bits[15*i:15*i+15]) for i in range(n_secret_bits // 4)])

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    cover = rng.integers(0, 2, 15 * 1000).astype(np.uint8)
    secret = rng.integers(0, 2, 4 * 1000).astype(np.uint8)
    stego = embed(cover, secret)
    print("secret recovered exactly:", np.array_equal(extract(stego, len(secret)), secret))
    print("changed bits per 15-bit block (max):", int(np.max((cover != stego).reshape(-1, 15).sum(1))))
    print("avg changes per block:", round(float((cover != stego).sum()) / 1000, 3), "(theory 15/16 = 0.9375)")