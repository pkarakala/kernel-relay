"""Deliberately narrow source language, not a general Python security sandbox."""

import ast


IMPORTS = {
    "torch": None, "torch.nn.functional": "functional",
    "triton": None, "triton.language": "tl",
}
BUILTINS = {"abs", "min", "max", "range", "len", "int", "float", "tuple", "RuntimeError", "ValueError"}
ATTRIBUTES = {
    "shape", "dtype", "device", "ndim", "numel", "size", "stride", "element_ty",
    "to", "float", "double", "contiguous", "reshape", "view", "clone",
    "empty_like", "zeros_like", "ones_like", "empty", "zeros", "ones", "full_like",
    "add", "sub", "mul", "div", "sum", "mean", "var", "rsqrt", "sqrt", "exp",
    "abs", "where", "sigmoid", "silu", "layer_norm", "float16", "float32", "float64",
    "int32", "int64", "jit", "constexpr", "program_id", "arange", "load", "store",
    "maximum", "minimum", "cdiv", "next_power_of_2",
}
TENSOR_ATTRIBUTES = {
    "shape", "dtype", "device", "ndim", "numel", "size", "stride", "element_ty",
    "to", "float", "double", "contiguous", "reshape", "view", "clone",
}
MODULE_ATTRIBUTES = {
    "torch": {"empty_like", "zeros_like", "ones_like", "empty", "zeros", "ones", "full_like",
              "add", "sub", "mul", "div", "sum", "mean", "var", "rsqrt", "sqrt", "exp",
              "abs", "where", "sigmoid", "float16", "float32", "float64", "int32", "int64"},
    "functional": {"layer_norm", "silu"},
    "triton": {"jit", "cdiv", "next_power_of_2"},
    "tl": {"constexpr", "program_id", "arange", "load", "store", "sum", "rsqrt", "sqrt",
           "exp", "abs", "where", "sigmoid", "maximum", "minimum", "float16", "float32",
           "float64", "int32", "int64"},
}
NODES = {
    ast.Module, ast.Import, ast.alias, ast.FunctionDef, ast.arguments, ast.arg,
    ast.Expr, ast.Constant, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return,
    ast.If, ast.IfExp, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass, ast.Raise,
    ast.Name, ast.Attribute, ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Dict,
    ast.Call, ast.keyword, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
    ast.Load, ast.Store, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.BitAnd, ast.BitOr,
}


def validate_source(source: str, candidate_type: str) -> ast.Module:
    tree = ast.parse(source, filename="candidate.py")
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "run" not in functions:
        raise ValueError("candidate must define run")
    reserved = BUILTINS | {"torch", "triton", "functional", "tl"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in IMPORTS or alias.asname != IMPORTS[alias.name]:
                    raise ValueError("only canonical torch/functional/triton/tl imports are allowed")
                if candidate_type != "triton_source" and alias.name.startswith("triton"):
                    raise ValueError("Triton imports require triton_source")
        elif not isinstance(node, ast.FunctionDef) and not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            raise ValueError("module scope allows only imports, functions, and docstrings")
    for node in ast.walk(tree):
        if type(node) not in NODES:
            raise ValueError(f"unsupported source construct: {type(node).__name__}")
        if isinstance(node, ast.Import) and node not in tree.body:
            raise ValueError("imports must be at module scope")
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or (isinstance(node.ctx, ast.Store) and node.id in reserved):
                raise ValueError("private or reserved names are forbidden")
            if node.id in MODULE_ATTRIBUTES and not (
                isinstance(parents.get(node), ast.Attribute) and parents[node].value is node
            ):
                raise ValueError("module aliases and module values are forbidden")
        if isinstance(node, ast.Attribute):
            allowed = (MODULE_ATTRIBUTES[node.value.id] if isinstance(node.value, ast.Name)
                       and node.value.id in MODULE_ATTRIBUTES else TENSOR_ATTRIBUTES)
            if node.attr not in allowed or isinstance(node.ctx, ast.Store):
                raise ValueError(f"unsupported attribute: {node.attr}")
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            raise ValueError("subscript mutation is forbidden")
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_") or node.name in reserved or node not in tree.body:
                raise ValueError("only public top-level functions are allowed")
            for decorator in node.decorator_list:
                if ast.unparse(decorator) != "triton.jit" or candidate_type != "triton_source":
                    raise ValueError("only @triton.jit is allowed")
            for default in node.args.defaults + [item for item in node.args.kw_defaults if item is not None]:
                if not isinstance(default, ast.Constant):
                    raise ValueError("function defaults must be constants")
        if isinstance(node, ast.arg) and (node.arg.startswith("_") or node.arg in reserved):
            raise ValueError("private or reserved argument names are forbidden")
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id not in functions | BUILTINS:
                raise ValueError(f"unsupported function: {function.id}")
            if not isinstance(function, (ast.Name, ast.Attribute, ast.Subscript)):
                raise ValueError("dynamic calls are forbidden")
            if isinstance(function, ast.Subscript) and not (
                isinstance(function.value, ast.Name) and function.value.id in functions - {"run"}
                and candidate_type == "triton_source"
            ):
                raise ValueError("subscript calls are restricted to local Triton kernels")
    entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    if ([argument.arg for argument in entry.args.args] != ["x", "input_bias", "gamma", "beta"]
            or [argument.arg for argument in entry.args.kwonlyargs] != ["eps", "launch_parameters"]
            or entry.args.posonlyargs or entry.args.vararg or entry.args.kwarg or entry.decorator_list):
        raise ValueError("run must accept (x, input_bias, gamma, beta, *, eps, launch_parameters)")
    return tree
