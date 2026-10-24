"""Tests for the package-level exports in ``bayeseor_jax`` and ``bayeseor_jax.inference``."""

from __future__ import annotations

import importlib

import pytest

import bayeseor_jax
import bayeseor_jax.analyze
import bayeseor_jax.inference


@pytest.mark.parametrize(
    "module", [bayeseor_jax, bayeseor_jax.inference, bayeseor_jax.analyze],
    ids=lambda m: m.__name__,
)
class TestExports:
    """Every exported name resolves; unknown names are rejected."""

    def test_all_names_resolve(self, module):
        """Every name in ``__all__`` is importable from the module."""
        for name in module.__all__:
            assert getattr(module, name) is not None, name

    def test_all_has_no_duplicates(self, module):
        """``__all__`` lists each name once."""
        assert len(module.__all__) == len(set(module.__all__))


@pytest.mark.parametrize("module", [bayeseor_jax, bayeseor_jax.inference], ids=lambda m: m.__name__)
class TestLazyExports:
    """Names loaded on first access through ``__getattr__``."""

    def test_lazy_names_are_cached(self, module):
        """A lazily loaded name is stored in the module namespace after first access."""
        for name in module._LAZY_EXPORTS:
            value = getattr(module, name)
            assert module.__dict__[name] is value

    def test_lazy_names_match_source(self, module):
        """Each lazy export is the object defined in its source module."""
        for name, (source, attribute) in module._LAZY_EXPORTS.items():
            source_module = importlib.import_module(source, module.__name__)
            assert getattr(module, name) is getattr(source_module, attribute)

    def test_unknown_name_raises(self, module):
        """Accessing an unknown name raises ``AttributeError`` naming the module."""
        with pytest.raises(AttributeError, match=module.__name__):
            module.not_a_real_export  # noqa: B018


def test_run_nestedsampling_is_run_multinest():
    """``run_nestedsampling`` is an alias of ``run_multinest``."""
    assert bayeseor_jax.run_nestedsampling is bayeseor_jax.run_multinest


def test_top_level_reexports_inference():
    """Top-level inference exports are the same objects as in ``bayeseor_jax.inference``."""
    for name in bayeseor_jax.inference.__all__:
        if name in bayeseor_jax.__all__:
            assert getattr(bayeseor_jax, name) is getattr(bayeseor_jax.inference, name)
