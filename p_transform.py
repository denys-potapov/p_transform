"""P_Transform."""
from bitarray import bitarray, util
import numpy as np

import random
import hashlib

from functools import cache


def p_transform(a, validate=True):
    """P-transform binary vector a.

    if validate=True assert that: p_transform(p_transform(a)) == a"""
    width = len(a)
    p = p_matrix(width)
    a_np = np.array(list(a))

    b = (a_np @ p) % 2
    b = bitarray(list(b))
    if validate:
        # assert that transfor return correct value
        assert p_transform(b, False) == a

    return b


@cache
def p_matrix(size):
    """Build P_matrix of size.

    Size should be a power of 2."""

    assert (size % 2) == 0 and size >= 4
    if size == 4:
        return np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        )

    p = np.zeros((size, size), dtype=int)
    half = size // 2

    part_1 = p_matrix(half)
    part_2 = filler_1(half)

    p[:half, :half] = part_1
    p[half:, half:] = part_1
    p[:half, half:] = part_2
    p[half:, :half] = part_2

    # assert that matrix is symmetrical transform
    assert np.array_equal(np.matmul(p, p) % 2, np.identity(size))

    return p


def filler_1(half):
    """Helper for P_matrix of size half*2.
    Return a chessboard like pattern. Example for half = 4:
        [0, 0, 1, 1]
        [0, 0, 1, 1]
        [1, 1, 0, 0]
        [1, 1, 0, 0]
    """
    part_2 = np.zeros((half, half), dtype=int)
    quat = half // 2
    part_2[:quat, quat:] = 1
    part_2[quat:, :quat] = 1
    return part_2


def distance(a, b):
    """Hamming distance."""
    return (a ^ b).count(1)


def sha256(a):
    """SHA-256 of input vector."""
    b = bitarray()
    b.frombytes(hashlib.sha256(a.tobytes()).digest())
    return b


def sha256_p(a):
    """SHA-256(P_transform(A)) of input vector."""
    return sha256(p_transform(a))


def calc_avalanche(num_samples, width, fn):
    """Calculate avalanche effect:
    1. Generate random input vector A
    2. Flip one bit in A to get B
    3. Calculate average distance between Fn(A) and Fn(B).
    """
    distances = []
    for i in range(num_samples):
        a = util.urandom(width)

        # flip one bit
        bit_flip = util.zeros(width)
        bit_flip[random.randrange(width)] = 1

        b = a ^ bit_flip
        d = distance(fn(a), fn(b))
        distances.append(d)

    return sum(distances) / len(distances)


if __name__ == "__main__":
    NUM_SAMPLES = 10_000
    avalanche_sha = calc_avalanche(NUM_SAMPLES, 256, sha256)
    print("SHA256 Avalanche", avalanche_sha)

    avalanche_p_transform = calc_avalanche(NUM_SAMPLES, 256, p_transform)
    print("P_transform Avalanche", avalanche_p_transform)

    avalanche_sha_p = calc_avalanche(NUM_SAMPLES, 256, sha256_p)
    print("sha256_p", avalanche_sha_p)
