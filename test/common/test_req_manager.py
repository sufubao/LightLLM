import ast
from pathlib import Path


def test_linear_att_state_buffer_log_reports_shape_and_memory():
    source = Path("lightllm/common/req_manager.py").read_text()
    module = ast.parse(source)

    class_node = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "ReqManagerForMamba"
    )
    init_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    init_source = ast.unparse(init_node)

    assert "logger.info" in init_source
    assert "conv_state shape=" in init_source
    assert "ssm_state shape=" in init_source
    assert "total memory=" in init_source
    assert "_format_nbytes(conv_nbytes)" in init_source
    assert "_format_nbytes(ssm_nbytes)" in init_source
