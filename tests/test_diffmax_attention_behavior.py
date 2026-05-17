import importlib

import pytest

torch = pytest.importorskip("torch")


def _reload_attention_module():
    import layers.SelfAttention_Family as attention_family

    return importlib.reload(attention_family)


def test_full_attention_diffmax_path_calls_diffmax_bisect(monkeypatch):
    attention_family = _reload_attention_module()
    monkeypatch.setattr(attention_family, "DIFFMAX_AVAILABLE", True)

    calls = []

    def fake_diffmax_bisect(scores, alpha, dim, n_iter):
        calls.append(
            {
                "shape": tuple(scores.shape),
                "alpha": alpha,
                "dim": dim,
                "n_iter": n_iter,
                "has_inf_mask": torch.isinf(scores).any().item(),
            }
        )
        return torch.softmax(scores, dim=dim)

    monkeypatch.setattr(attention_family, "diffmax_bisect", fake_diffmax_bisect)

    attn = attention_family.FullAttention(
        mask_flag=False,
        attention_dropout=0.0,
        normalizer="diffmax",
        diffmax_alpha=0.3,
        diffmax_n_iter=17,
    )
    queries = torch.randn(2, 5, 3, 4)
    keys = torch.randn(2, 7, 3, 4)
    values = torch.randn(2, 7, 3, 6)

    output, weights = attn(queries, keys, values, attn_mask=None)

    assert output.shape == (2, 5, 3, 6)
    assert weights is None
    assert len(calls) == 1
    assert calls[0]["shape"] == (2, 3, 5, 7)
    assert calls[0]["alpha"] == pytest.approx(0.3)
    assert calls[0]["dim"] == -1
    assert calls[0]["n_iter"] == 17
    assert calls[0]["has_inf_mask"] is False


def test_full_attention_softmax_path_does_not_call_diffmax(monkeypatch):
    attention_family = _reload_attention_module()
    monkeypatch.setattr(attention_family, "DIFFMAX_AVAILABLE", True)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("softmax path unexpectedly called diffmax_bisect")

    monkeypatch.setattr(attention_family, "diffmax_bisect", fail_if_called)

    attn = attention_family.FullAttention(
        mask_flag=False,
        attention_dropout=0.0,
        normalizer="softmax",
    )
    queries = torch.randn(2, 5, 3, 4)
    keys = torch.randn(2, 7, 3, 4)
    values = torch.randn(2, 7, 3, 6)

    output, _ = attn(queries, keys, values, attn_mask=None)

    assert output.shape == (2, 5, 3, 6)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("attention_cls", ["FullAttention", "ProbAttention"])
def test_diffmax_without_dependency_fails_fast(monkeypatch, attention_cls):
    attention_family = _reload_attention_module()
    monkeypatch.setattr(attention_family, "DIFFMAX_AVAILABLE", False)

    with pytest.raises(ImportError, match="diffmax not installed"):
        getattr(attention_family, attention_cls)(normalizer="diffmax")


def test_full_attention_causal_mask_reaches_diffmax_as_negative_infinity(monkeypatch):
    attention_family = _reload_attention_module()
    monkeypatch.setattr(attention_family, "DIFFMAX_AVAILABLE", True)

    observed = {}

    def fake_diffmax_bisect(scores, alpha, dim, n_iter):
        observed["scores"] = scores.detach().clone()
        return torch.softmax(scores, dim=dim)

    monkeypatch.setattr(attention_family, "diffmax_bisect", fake_diffmax_bisect)

    attn = attention_family.FullAttention(
        mask_flag=True,
        attention_dropout=0.0,
        normalizer="diffmax",
        diffmax_alpha=0.5,
        diffmax_n_iter=9,
    )
    queries = torch.randn(1, 4, 2, 3)
    keys = torch.randn(1, 4, 2, 3)
    values = torch.randn(1, 4, 2, 5)

    output, _ = attn(queries, keys, values, attn_mask=None)

    assert torch.isfinite(output).all()
    assert "scores" in observed
    scores = observed["scores"]
    assert scores.shape == (1, 2, 4, 4)
    assert torch.isneginf(scores[..., 0, 1:]).all()
    assert torch.isfinite(scores[..., 0, 0]).all()


def test_real_diffmax_produces_finite_normalized_attention_if_installed():
    pytest.importorskip("diffmax")
    attention_family = _reload_attention_module()

    attn = attention_family.FullAttention(
        mask_flag=False,
        attention_dropout=0.0,
        output_attention=True,
        normalizer="diffmax",
        diffmax_alpha=0.3,
        diffmax_n_iter=50,
    )
    queries = torch.randn(2, 5, 3, 4)
    keys = torch.randn(2, 5, 3, 4)
    values = torch.randn(2, 5, 3, 6)

    output, weights = attn(queries, keys, values, attn_mask=None)

    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= -1e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-4)
