from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _base_config(**overrides):
    config = {
        "task_name": "long_term_forecast",
        "seq_len": 24,
        "label_len": 12,
        "pred_len": 12,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "d_model": 16,
        "n_heads": 2,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 32,
        "factor": 3,
        "dropout": 0.0,
        "activation": "gelu",
        "embed": "timeF",
        "freq": "h",
        "distil": False,
        "normalizer": "diffmax",
        "diffmax_alpha": 0.37,
        "diffmax_n_iter": 23,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def _attention_modules(model):
    from layers.SelfAttention_Family import FullAttention, ProbAttention

    return [module for module in model.modules() if isinstance(module, (FullAttention, ProbAttention))]


@pytest.mark.parametrize(
    ("module_name", "config_overrides", "expected_attention_count"),
    [
        ("models.Transformer", {}, 4),
        ("models.Informer", {}, 4),
        ("models.iTransformer", {}, 2),
        ("models.PatchTST", {}, 2),
        ("models.Crossformer", {}, 18),
    ],
)
def test_cross_backbone_models_wire_diffmax_to_all_attention_modules(
    module_name,
    config_overrides,
    expected_attention_count,
):
    pytest.importorskip("diffmax")
    module = __import__(module_name, fromlist=["Model"])
    model = module.Model(_base_config(**config_overrides))

    attention_modules = _attention_modules(model)

    assert len(attention_modules) == expected_attention_count
    for attention in attention_modules:
        assert attention.normalizer == "diffmax"
        assert attention.diffmax_alpha == pytest.approx(0.37)
        assert attention.diffmax_n_iter == 23


@pytest.mark.parametrize(
    "module_name",
    [
        "models.Transformer",
        "models.Informer",
        "models.iTransformer",
        "models.PatchTST",
        "models.Crossformer",
    ],
)
def test_cross_backbone_models_wire_softmax_when_requested(module_name):
    module = __import__(module_name, fromlist=["Model"])
    config = _base_config(normalizer="softmax", diffmax_alpha=0.91, diffmax_n_iter=31)
    model = module.Model(config)

    attention_modules = _attention_modules(model)

    assert attention_modules
    for attention in attention_modules:
        assert attention.normalizer == "softmax"
        assert attention.diffmax_alpha == pytest.approx(0.91)
        assert attention.diffmax_n_iter == 31


def test_no_cross_backbone_model_hardcodes_plain_attention_without_normalizer():
    import ast
    from pathlib import Path

    model_files = [
        Path("models/Transformer.py"),
        Path("models/Informer.py"),
        Path("models/iTransformer.py"),
        Path("models/PatchTST.py"),
        Path("models/Crossformer.py"),
        Path("layers/SelfAttention_Family.py"),
    ]

    violations = []
    for path in model_files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Name) or func.id not in {"FullAttention", "ProbAttention"}:
                continue
            keyword_names = {keyword.arg for keyword in node.keywords}
            if "normalizer" not in keyword_names:
                violations.append(f"{path}:{node.lineno} calls {func.id} without normalizer=")

    assert not violations, "\n".join(violations)
