import ast
import re
from pathlib import Path


def _bash_files_under(path):
    return sorted(Path(path).glob("**/*.sh"))


def test_run_py_exposes_diffmax_cli_arguments():
    source = Path("run.py").read_text()

    assert "--normalizer" in source
    assert "choices=['softmax', 'diffmax']" in source or 'choices=["softmax", "diffmax"]' in source
    assert "--diffmax_alpha" in source
    assert "--diffmax_n_iter" in source


def test_print_args_reports_diffmax_configuration_for_log_auditability():
    source = Path("utils/print_args.py").read_text()

    missing = [
        name
        for name in ("normalizer", "diffmax_alpha", "diffmax_n_iter")
        if f"args.{name}" not in source
    ]
    assert not missing, (
        "print_args should include diffmax configuration so logs prove which "
        f"normalizer ran. Missing: {missing}"
    )


def test_training_setting_contains_normalizer_not_only_freeform_description():
    source = Path("run.py").read_text()

    assert "args.normalizer" in source, (
        "run.py setting/checkpoint names should include args.normalizer explicitly; "
        "depending only on model_id/des makes result parsing fragile."
    )


def test_long_term_result_file_records_normalizer_and_diffmax_hyperparameters():
    source = Path("exp/exp_long_term_forecasting.py").read_text()

    missing = [
        name
        for name in ("normalizer", "diffmax_alpha", "diffmax_n_iter")
        if f"self.args.{name}" not in source
    ]
    assert not missing, (
        "result text should record normalizer and diffmax hyperparameters next to "
        f"metrics. Missing: {missing}"
    )


def test_all_diffmax_scripts_schedule_both_softmax_and_diffmax_runs():
    scripts = _bash_files_under("diffmax_scripts")
    assert scripts, "No diffmax scripts found"

    violations = []
    for script in scripts:
        text = script.read_text()
        if "--normalizer softmax" not in text:
            violations.append(f"{script}: missing --normalizer softmax")
        if "--normalizer diffmax" not in text:
            violations.append(f"{script}: missing --normalizer diffmax")
        if "--diffmax_alpha" not in text:
            violations.append(f"{script}: missing --diffmax_alpha for diffmax branch")
        if "run_one \"softmax\"" not in text:
            violations.append(f"{script}: softmax branch is not scheduled")
        if "run_one \"diffmax\"" not in text:
            violations.append(f"{script}: diffmax branch is not scheduled")

    assert not violations, "\n".join(violations)


def test_all_diffmax_scripts_use_distinct_ids_for_softmax_and_diffmax():
    scripts = _bash_files_under("diffmax_scripts")
    violations = []

    for script in scripts:
        text = script.read_text()
        softmax_model_ids = re.findall(r'model_id="[^"]*softmax[^"]*"', text)
        diffmax_model_ids = re.findall(r'model_id="[^"]*diffmax[^"]*"', text)
        softmax_des = re.findall(r'des="[^"]*softmax[^"]*"', text)
        diffmax_des = re.findall(r'des="[^"]*diffmax[^"]*"', text)
        if not softmax_model_ids:
            violations.append(f"{script}: softmax model_id does not include softmax")
        if not diffmax_model_ids:
            violations.append(f"{script}: diffmax model_id does not include diffmax")
        if not softmax_des:
            violations.append(f"{script}: softmax des does not include softmax")
        if not diffmax_des:
            violations.append(f"{script}: diffmax des does not include diffmax")

    assert not violations, "\n".join(violations)


def test_experiment_scripts_do_not_default_to_one_epoch_smoke_runs():
    scripts = [path for path in _bash_files_under("diffmax_scripts") if path.name != "test.sh"]
    violations = []

    for script in scripts:
        text = script.read_text()
        if re.search(r"--train_epochs\s+1(?!\d)(\s|\\)", text):
            violations.append(f"{script}: uses --train_epochs 1 outside smoke test")

    assert not violations, "\n".join(violations)


def test_diffmax_attention_implementation_has_no_softmax_fallback_in_diffmax_branch():
    tree = ast.parse(Path("layers/SelfAttention_Family.py").read_text())

    normalize_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_normalize"
    ]
    assert normalize_functions

    violations = []
    for function in normalize_functions:
        for node in ast.walk(function):
            if not isinstance(node, ast.If):
                continue
            condition = ast.unparse(node.test)
            if 'self.normalizer == "diffmax"' not in condition and "self.normalizer == 'diffmax'" not in condition:
                continue
            branch_source = "\n".join(ast.unparse(item) for item in node.body)
            if "diffmax_bisect" not in branch_source:
                violations.append(f"_normalize at line {function.lineno}: diffmax branch lacks diffmax_bisect")
            if "softmax" in branch_source:
                violations.append(f"_normalize at line {function.lineno}: diffmax branch contains softmax fallback")

    assert not violations, "\n".join(violations)
