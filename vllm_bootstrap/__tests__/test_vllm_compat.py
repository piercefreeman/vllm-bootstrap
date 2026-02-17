"""Verify that vllm APIs used by vllm-bootstrap are importable.

This test file is run against multiple vllm versions in CI to catch
API breakage early — even though we cannot run GPU inference in CI.
"""

from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm")


def test_llm_class_importable():
    from vllm import LLM

    assert callable(LLM)


def test_sampling_params_importable():
    from vllm import SamplingParams

    assert callable(SamplingParams)


def test_structured_outputs_params_importable():
    from vllm.sampling_params import StructuredOutputsParams

    assert callable(StructuredOutputsParams)


def test_llm_has_generate_method():
    from vllm import LLM

    assert hasattr(LLM, "generate")


def test_llm_has_embed_method():
    from vllm import LLM

    assert hasattr(LLM, "embed")
