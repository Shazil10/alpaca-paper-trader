"""Strategy lookup rules.

The important behaviour here is a *refusal*: a strategy that only implements
the live `generate_signals` interface must not be resolved automatically. The
bridge that adapts it cannot enforce the point-in-time firewall, so resolving
it silently would produce a look-ahead-contaminated equity curve that is
indistinguishable from a clean one.
"""
import logging
import sys
import types

import pytest

from backtest import strategy as strat


@pytest.fixture(autouse=True)
def _clean_registry():
    strat.clear_registry()
    yield
    strat.clear_registry()


def _install_module(name: str, **attrs):
    """Register a throwaway module on sys.modules for the duration of a test."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


@pytest.fixture
def temp_modules():
    created = []

    def make(name, **attrs):
        created.append(name)
        return _install_module(name, **attrs)

    yield make
    for name in created:
        sys.modules.pop(name, None)


def test_migrated_module_resolves_directly(temp_modules):
    temp_modules("_tw_only", target_weights=lambda ctx: {"SPY": 1.0})

    fn = strat.get_strategy("_tw_only")
    assert fn(None) == {"SPY": 1.0}
    # Resolution is cached under the requested name.
    assert "_tw_only" in strat.list_strategies()


def test_legacy_only_module_is_refused_with_an_actionable_message(temp_modules):
    """The refusal is the safety property: no silent contaminated backtest."""
    temp_modules("_gs_only", generate_signals=lambda **kw: [])

    with pytest.raises(KeyError) as exc:
        strat.get_strategy("_gs_only")

    msg = str(exc.value)
    assert "legacy:_gs_only" in msg       # tells you how to proceed
    assert "point-in-time" in msg          # tells you what you give up
    assert "_gs_only" not in strat.list_strategies()


def test_legacy_prefix_opts_in_and_warns(temp_modules, caplog, monkeypatch):
    sentinel = {"AAPL": 0.5}
    monkeypatch.setattr(
        "backtest.adapter.wrap_generate_signals",
        lambda module_path, **kw: (lambda ctx: sentinel),
    )
    temp_modules("_gs_opt_in", generate_signals=lambda **kw: [])

    with caplog.at_level(logging.WARNING, logger="backtest.strategy"):
        fn = strat.get_strategy("legacy:_gs_opt_in")

    assert fn(None) is sentinel
    warning = " ".join(r.getMessage() for r in caplog.records)
    assert "NOT enforced" in warning and "look ahead" in warning


def test_module_with_neither_interface_is_distinguished(temp_modules):
    temp_modules("_empty_strat", something_else=1)

    with pytest.raises(KeyError) as exc:
        strat.get_strategy("_empty_strat")
    assert "neither" in str(exc.value)


def test_unimportable_module_is_not_reported_as_a_missing_interface():
    """An import failure and a missing function are different faults; a typo'd
    module path used to surface as 'has no target_weights'."""
    with pytest.raises(KeyError) as exc:
        strat.get_strategy("nonexistent.module.path")
    assert "could not be imported" in str(exc.value)


def test_registered_strategy_shadows_module_import(temp_modules):
    temp_modules("_shadowed", target_weights=lambda ctx: {"FROM_MODULE": 1.0})
    strat.register("_shadowed", lambda ctx: {"FROM_REGISTRY": 1.0})

    assert strat.get_strategy("_shadowed")(None) == {"FROM_REGISTRY": 1.0}
