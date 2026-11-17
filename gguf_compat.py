"""Compatibility shim for the legacy Q2_0 GGUF type in gguf-py 0.19."""

from __future__ import annotations

import gguf


Q2_0_TYPE_ID = 42
Q2_0_BLOCK_SIZE = 32
Q2_0_TYPE_SIZE = 10


def ensure_gguf_q2_0_type():
    """Register the ggml Q2_0 type when the installed gguf-py lacks its enum."""
    enum = gguf.GGMLQuantizationType
    member = enum.__members__.get("Q2_0")
    if member is None:
        existing = enum._value2member_map_.get(Q2_0_TYPE_ID)
        if existing is not None:
            raise RuntimeError(
                f"GGML type id {Q2_0_TYPE_ID} is assigned to {existing.name}; cannot register Q2_0"
            )
        member = int.__new__(enum, Q2_0_TYPE_ID)
        member._name_ = "Q2_0"
        member._value_ = Q2_0_TYPE_ID
        type.__setattr__(enum, "Q2_0", member)
        enum._member_names_.append("Q2_0")
        enum._member_map_["Q2_0"] = member
        enum._value2member_map_[Q2_0_TYPE_ID] = member

    expected_size = (Q2_0_BLOCK_SIZE, Q2_0_TYPE_SIZE)
    actual_size = gguf.GGML_QUANT_SIZES.get(member)
    if actual_size is None:
        gguf.GGML_QUANT_SIZES[member] = expected_size
    elif tuple(actual_size) != expected_size:
        raise RuntimeError(f"Unexpected gguf-py Q2_0 block size: {actual_size}")
    return member


ensure_gguf_q2_0_type()
