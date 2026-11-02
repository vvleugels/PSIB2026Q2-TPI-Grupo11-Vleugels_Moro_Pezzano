"""Lossless coding estimators and an educational JPEG-like DCT codec."""
from __future__ import annotations

from collections import Counter
import heapq
from itertools import count

import numpy as np
from scipy.fft import dctn, idctn


LUMINANCE_QUANTIZATION = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.int32)


def huffman_code_table(frequencies: dict) -> dict:
    """Return a deterministic Huffman code mapping from symbols to bit strings."""
    if any(int(freq) <= 0 for freq in frequencies.values()):
        raise ValueError("Huffman frequencies must be positive integers")
    if not frequencies:
        return {}
    serial = count()
    heap = [(int(freq), next(serial), ("leaf", symbol)) for symbol, freq in frequencies.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        return {heap[0][2][1]: "0"}
    while len(heap) > 1:
        left_weight, _, left = heapq.heappop(heap)
        right_weight, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (left_weight + right_weight, next(serial), ("node", left, right)))
    table: dict = {}

    def visit(node, prefix):
        if node[0] == "leaf":
            table[node[1]] = prefix or "0"
        else:
            visit(node[1], prefix + "0")
            visit(node[2], prefix + "1")
    visit(heap[0][2], "")
    return table


def huffman_encode_decode(values) -> tuple[list, dict, int]:
    """Encode symbols into a bit string, decode it, and return table and bit count."""
    symbols = list(values)
    table = huffman_code_table(Counter(symbols))
    bits = "".join(table[symbol] for symbol in symbols)
    reverse = {code: symbol for symbol, code in table.items()}
    decoded, prefix = [], ""
    for bit in bits:
        prefix += bit
        if prefix in reverse:
            decoded.append(reverse[prefix])
            prefix = ""
    if prefix:
        raise ValueError("Huffman bitstream ended mid-code")
    return decoded, table, len(bits)


def entropy_bits_per_pixel(image_u8: np.ndarray) -> float:
    """Shannon entropy of 8-bit pixel values in bits per pixel."""
    image = _validate_u8(image_u8)
    counts = np.bincount(image.ravel(), minlength=256).astype(float)
    probabilities = counts[counts > 0] / image.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def row_rle_size_bytes(image_u8: np.ndarray) -> int:
    """Estimate row-reset RLE size using one value byte and uint16 run length."""
    image = _validate_u8(image_u8)
    runs = 0
    for row in image:
        if row.size:
            runs += 1 + int(np.count_nonzero(row[1:] != row[:-1]))
    return runs * 3


def jpeg_quantization_table(quality: int) -> np.ndarray:
    """Scale the standard luminance table using the IJG quality convention."""
    if isinstance(quality, bool) or not isinstance(quality, (int, np.integer)) or not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be an integer in [1, 100]")
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip((LUMINANCE_QUANTIZATION * scale + 50) // 100, 1, 255).astype(np.int32)


def _zigzag_indices() -> list[tuple[int, int]]:
    return [(i, j) for diagonal in range(15)
            for i, j in sorted(((i, diagonal - i) for i in range(8)
                                if 0 <= diagonal - i < 8), reverse=not bool(diagonal % 2))]


_ZIGZAG = _zigzag_indices()


def jpeg_like_dct(image_u8: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    """Quantize 8x8 DCT blocks, reconstruct, and estimate entropy-coded bits.

    The estimate models zigzag AC run/category and differential DC category
    symbols, Huffman code lengths and coefficient magnitude bits; it excludes
    JPEG headers and byte stuffing. The returned estimate is therefore compared
    with, but is not identical to, the real ``cv2.imencode`` file size.
    """
    image = _validate_u8(image_u8)
    quant = jpeg_quantization_table(quality)
    height, width = image.shape
    pad_h, pad_w = (-height) % 8, (-width) % 8
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge").astype(np.float64)
    reconstructed = np.zeros_like(padded)
    dc_symbols, ac_symbols = [], []
    extra_bits = 0
    previous_dc = 0
    for y in range(0, padded.shape[0], 8):
        for x in range(0, padded.shape[1], 8):
            block = padded[y:y + 8, x:x + 8] - 128
            coeff = dctn(block, type=2, norm="ortho")
            qcoeff = np.rint(coeff / quant).astype(np.int32)
            block_reconstruction = idctn(qcoeff * quant, type=2, norm="ortho") + 128
            reconstructed[y:y + 8, x:x + 8] = block_reconstruction
            sequence = [int(qcoeff[i, j]) for i, j in _ZIGZAG]
            dc = sequence[0]
            difference = dc - previous_dc
            previous_dc = dc
            dc_category = abs(difference).bit_length()
            dc_symbols.append(dc_category)
            extra_bits += dc_category
            run = 0
            for value in sequence[1:]:
                if value == 0:
                    run += 1
                    if run == 16:
                        ac_symbols.append((15, 0))
                        run = 0
                else:
                    category = abs(value).bit_length()
                    ac_symbols.append((run, category))
                    extra_bits += category
                    run = 0
            if run:
                ac_symbols.append((0, 0))  # End of block.
    dc_lengths = {symbol: len(code) for symbol, code in huffman_code_table(Counter(dc_symbols)).items()}
    ac_lengths = {symbol: len(code) for symbol, code in huffman_code_table(Counter(ac_symbols)).items()}
    coded_bits = sum(dc_lengths[value] for value in dc_symbols)
    coded_bits += sum(ac_lengths[value] for value in ac_symbols) + extra_bits
    restored = np.clip(np.rint(reconstructed[:height, :width]), 0, 255).astype(np.uint8)
    return restored, int(coded_bits)


def _validate_u8(image: np.ndarray) -> np.ndarray:
    result = np.asarray(image)
    if result.ndim != 2 or result.dtype != np.uint8:
        raise ValueError("image must be a 2D uint8 array")
    if result.size == 0:
        raise ValueError("image must not be empty")
    return result
