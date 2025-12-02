
from typing import List, Tuple, Dict, Any
import math
from array import array

# -----------------------------
# PForDelta (Patched Frame of Reference) with delta coding
# -----------------------------
# This is a simple, readable reference implementation aimed at educational use.
# It splits a sequence into fixed-size blocks (default 128), chooses a bit-width b
# that fits most values, bit-packs those "inliers", and stores the few "outliers"
# (larger values) as patches (position, value). Before that, we apply delta coding
# to make gaps smaller (critical for postings and positions).
#
# The format per block (Python dict for clarity):
# {
#   "first": int,             # base (for delta decoding)
#   "b": int,                 # bit width for inliers
#   "packed": array('I'),     # 32-bit chunks with bit-packed values
#   "n": int,                 # number of values in this block
#   "patches": List[Tuple[int,int]]  # (index_in_block, original_delta_value)
# }
#
# The whole stream is a list of blocks.


def _delta_encode(nums: List[int]) -> List[int]:
    if not nums:
        return []
    deltas = [nums[0]]
    for i in range(1, len(nums)):
        deltas.append(nums[i] - nums[i-1])
    return deltas


def _delta_decode(deltas: List[int]) -> List[int]:
    if not deltas:
        return []
    out = [deltas[0]]
    for i in range(1, len(deltas)):
        out.append(out[-1] + deltas[i])
    return out


def _choose_b(values: List[int], max_outliers_ratio: float = 0.1) -> Tuple[int, List[int]]:
    """Pick bit-width b so that at least (1 - max_outliers_ratio) of values fit.
    Returns b and the list of outlier indices.
    """
    if not values:
        return 0, []
    # Try b from 0..32 and pick minimal b that leaves <= ratio outliers
    for b in range(0, 33):
        limit = (1 << b) - 1 if b > 0 else 0
        outliers = [i for i, v in enumerate(values) if v > limit]
        if len(outliers) <= max(1, int(len(values) * max_outliers_ratio)):
            return b, outliers
    # Fallback: 32 bits (no outliers)
    return 32, []


def _pack_bits(values: List[int], b: int) -> array:
    """Pack list of non-negative ints into 32-bit words with b bits per value."""
    if b == 0:
        return array('I')  # nothing to pack
    bitstream = 0
    bitlen = 0
    out = array('I')
    for v in values:
        bitstream = (bitstream << b) | (v & ((1 << b) - 1))
        bitlen += b
        while bitlen >= 32:
            # Flush the top 32 bits
            shift = bitlen - 32
            word = (bitstream >> shift) & 0xFFFFFFFF
            out.append(word)
            bitlen -= 32
            bitstream &= (1 << shift) - 1 if shift > 0 else 0
    if bitlen > 0:
        out.append((bitstream << (32 - bitlen)) & 0xFFFFFFFF)
    return out


def _unpack_bits(words: array, n: int, b: int) -> List[int]:
    """Unpack n values (each b bits) from 32-bit words."""
    if b == 0:
        return [0] * n
    vals = []
    bitbuf = 0
    bitlen = 0
    it = iter(words)
    try:
        word = next(it)
    except StopIteration:
        word = None
    word_bits_left = 32 if word is not None else 0

    for _ in range(n):
        v = 0
        bits_needed = b
        while bits_needed > 0:
            if word is None:
                raise ValueError("Not enough data to unpack")
            take = min(bits_needed, word_bits_left)
            v = (v << take) | ((word >> (32 - take)) & ((1 << take) - 1))
            word = ((word << take) & 0xFFFFFFFF) if take < 32 else 0
            word_bits_left -= take
            if word_bits_left == 0:
                try:
                    word = next(it)
                    word_bits_left = 32
                except StopIteration:
                    if len(vals) + 1 < n or bits_needed - take > 0:
                        raise ValueError("Truncated bitstream")
                    word = None
                    word_bits_left = 0
            bits_needed -= take
        vals.append(v)
    return vals


class PForDelta:
    def __init__(self, block_size: int = 128, max_outliers_ratio: float = 0.1):
        self.block_size = block_size
        self.max_outliers_ratio = max_outliers_ratio

    # --------------- Public API ---------------

    def encode(self, arr: List[int]) -> List[Dict[str, Any]]:
        """Encode an increasing integer sequence using delta + PForDelta."""
        if not arr:
            return []
        # delta on the absolute sequence
        deltas = _delta_encode(arr)
        out_blocks = []
        for i in range(0, len(deltas), self.block_size):
            block = deltas[i:i+self.block_size]
            first = block[0]
            rest = block[1:]
            b, outliers_idx = _choose_b(rest, self.max_outliers_ratio)
            # Build the inliers stream: values > limit are stored as 0 in packed stream,
            # and real values are stored in patches.
            limit = (1 << b) - 1 if b > 0 else 0
            packed_values = []
            patches = []
            for j, v in enumerate(rest):
                if v > limit:
                    packed_values.append(0 if b > 0 else 0)
                    patches.append((j+1, v))  # +1 to account for "first" at index 0
                else:
                    packed_values.append(v if b > 0 else 0)
            packed = _pack_bits(packed_values, b)
            out_blocks.append({
                "first": first,
                "b": b,
                "packed": packed,
                "n": len(block),
                "patches": patches,
            })
        return out_blocks

    def decode(self, blocks: List[Dict[str, Any]]) -> List[int]:
        """Decode back to the increasing sequence."""
        if not blocks:
            return []
        deltas: List[int] = []
        for blk in blocks:
            first = blk["first"]
            b = blk["b"]
            n = blk["n"]
            packed = blk["packed"]
            patches = dict(blk["patches"])  # index_in_block -> value
            if n == 0:
                continue
            deltas.append(first)
            if n > 1:
                rest = _unpack_bits(packed, n-1, b) if b >= 0 else [0]*(n-1)
                # apply patches
                for idx, val in patches.items():
                    if 0 <= idx-1 < len(rest):
                        rest[idx-1] = val
                deltas.extend(rest)
        # invert delta
        return _delta_decode(deltas)


# -----------------------------
# Helpers to compress an inverted index structure
# -----------------------------

def build_docid_mapping(doc_ids: List[str]) -> Dict[str, int]:
    """Map document IDs (strings) to dense integer IDs [1..N] in a stable order."""
    return {doc_id: i+1 for i, doc_id in enumerate(sorted(doc_ids))}


def extract_postings(index_obj) -> Dict[Tuple[str, str], Dict[str, List[int]]]:
    """
    Turns the given index.index structure (term -> field -> doc_id -> positions)
    into a simpler dict: (term, field) -> { doc_id: positions_sorted }
    """
    out = {}
    for term, field_dict in index_obj.index.items():
        for field, postings in field_dict.items():
            norm_postings = {}
            for doc_id, positions in postings.items():
                norm_postings[doc_id] = sorted(positions)
            out[(term, field)] = norm_postings
    return out


def postings_to_sorted_docids(postings: Dict[str, List[int]], doc_map: Dict[str, int]) -> List[int]:
    """Return sorted list of integer docIDs for a term/field postings dict."""
    doc_ints = [doc_map[d] for d in postings.keys() if d in doc_map]
    doc_ints.sort()
    return doc_ints


def compress_index_pfordelta(index_obj) -> Dict[str, Any]:
    """
    Build a compressed view of the index using PForDelta both for docIDs and positions.
    Returns a dictionary with metadata and compressed data.
    """
    p4d = PForDelta()
    postings = extract_postings(index_obj)
    doc_map = build_docid_mapping(list(index_obj.documents.keys()))

    compressed = {}
    total_raw_positions = 0
    total_raw_docids = 0
    for (term, field), post in postings.items():
        key = f"{term}||{field}"
        # DocIDs stream
        doc_stream = postings_to_sorted_docids(post, doc_map)
        total_raw_docids += len(doc_stream)
        comp_doc_stream = p4d.encode(doc_stream)

        # Positions per doc (each list delta+PForDelta independently)
        comp_positions = {}
        for doc_id, pos_list in post.items():
            total_raw_positions += len(pos_list)
            if pos_list:
                comp_positions[doc_id] = p4d.encode(sorted(pos_list))
            else:
                comp_positions[doc_id] = []

        compressed[key] = {
            "docids": comp_doc_stream,
            "positions": comp_positions,
        }

    meta = {
        "doc_map": doc_map,  # needed to interpret docIDs
        "block_size": p4d.block_size,
        "max_outliers_ratio": p4d.max_outliers_ratio,
        "total_raw_docids": total_raw_docids,
        "total_raw_positions": total_raw_positions,
    }
    return {"meta": meta, "data": compressed}


def decompress_docids(comp_stream: List[Dict[str, Any]]) -> List[int]:
    return PForDelta().decode(comp_stream)


def decompress_positions(comp_blocks: List[Dict[str, Any]]) -> List[int]:
    return PForDelta().decode(comp_blocks)
