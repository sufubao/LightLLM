import ast
from pathlib import Path


def test_model_kvargs_uses_running_max_req_size_without_extra_padding():
    source = Path("lightllm/server/router/manager.py").read_text()
    module = ast.parse(source)

    for node in ast.walk(module):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "max_req_num":
                assert ast.unparse(value) == "self.args.running_max_req_size"
                return

    raise AssertionError("max_req_num kvarg was not found")
