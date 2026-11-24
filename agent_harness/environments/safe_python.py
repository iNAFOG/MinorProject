"""A deliberately small AST interpreter. Never eval/exec untrusted source."""
import ast
import operator


class UnsafeCode(ValueError):
    pass


class SafePython:
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
              ast.Mod: operator.mod}
    compare = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
               ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
               ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
    builtins = {'sum': sum, 'len': len, 'min': min, 'max': max, 'abs': abs,
                'sorted': sorted, 'range': range, 'list': list, 'int': int,
                'bool': bool, 'enumerate': enumerate}
    permitted = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
                 ast.Assign, ast.AugAssign, ast.If, ast.For, ast.Pass, ast.Expr,
                 ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp,
                 ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.List, ast.Tuple,
                 ast.Dict, ast.Subscript, ast.Slice, ast.And, ast.Or, ast.Not,
                 ast.USub, ast.UAdd, *binary, *compare)

    def __init__(self, source, max_steps=10000):
        if len(source) > 50000:
            raise UnsafeCode('source too large')
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, self.permitted):
                raise UnsafeCode(f'unsupported syntax: {type(node).__name__}')
            if isinstance(node, ast.Name) and node.id.startswith('_'):
                raise UnsafeCode('private names forbidden')
            if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.name.startswith('_')):
                raise UnsafeCode('decorators/private functions forbidden')
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.keywords):
                raise UnsafeCode('only positional calls to named functions')
        if not all(isinstance(n, ast.FunctionDef) for n in tree.body):
            raise UnsafeCode('only function definitions at module level')
        self.functions = {n.name: n for n in tree.body}
        self.max_steps = max_steps
        self.remaining = max_steps
        self.depth = 0

    def invoke(self, name, args):
        """Top-level entry point: each independent check case gets a fresh step budget."""
        if name not in self.functions:
            raise UnsafeCode(f'function not defined: {name}')
        self.remaining, self.depth = self.max_steps, 0
        return self.call(name, list(args))

    def tick(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise UnsafeCode('interpreter step limit')

    def bounded(self, value):
        if isinstance(value, (str, bytes, list, tuple, dict, range)) and len(value) > 10000:
            raise UnsafeCode('value size limit')
        if isinstance(value, int) and value.bit_length() > 256:
            raise UnsafeCode('integer size limit')
        return value

    def call(self, name, args):
        self.tick()
        if self.depth >= 16:
            raise UnsafeCode('function depth limit')
        if name in self.builtins:
            if name == 'range':
                return self.bounded(range(*args))
            for arg in args:
                self.bounded(arg)
            return self.bounded(self.builtins[name](*args))
        fn = self.functions[name]
        if len(fn.args.args) != len(args) or fn.args.vararg or fn.args.kwarg or fn.args.defaults:
            raise UnsafeCode('unsupported function arguments')
        env = dict(zip([a.arg for a in fn.args.args], args))
        self.depth += 1
        try:
            returned, result = self.block(fn.body, env)
            return result if returned else None
        finally:
            self.depth -= 1

    def expr(self, node, env):
        self.tick()
        if isinstance(node, ast.Constant):
            return self.bounded(node.value)
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            value = [self.expr(n, env) for n in node.elts]
            return self.bounded(tuple(value) if isinstance(node, ast.Tuple) else value)
        if isinstance(node, ast.Dict):
            return self.bounded({self.expr(k, env): self.expr(v, env) for k, v in zip(node.keys, node.values)})
        if isinstance(node, ast.BinOp):
            left, right = self.expr(node.left, env), self.expr(node.right, env)
            # Check before allocation (sequence repetition/concatenation).
            sequences = (str, bytes, list, tuple)
            if isinstance(node.op, ast.Mult) and isinstance(left, sequences):
                if not isinstance(right, int) or len(left) * max(right, 0) > 10000:
                    raise UnsafeCode('allocation limit')
            if isinstance(node.op, ast.Mult) and isinstance(right, sequences):
                if not isinstance(left, int) or len(right) * max(left, 0) > 10000:
                    raise UnsafeCode('allocation limit')
            # printf-style formatting can request an arbitrary field width before any size check.
            if isinstance(node.op, ast.Mod) and isinstance(left, (str, bytes)):
                raise UnsafeCode('string formatting is not supported')
            return self.bounded(self.binary[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand, env)
            return {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}[type(node.op)](value)
        if isinstance(node, ast.BoolOp):
            result = None
            for value in node.values:
                result = self.expr(value, env)
                if isinstance(node.op, ast.And) and not result or isinstance(node.op, ast.Or) and result:
                    break
            return result
        if isinstance(node, ast.Compare):
            left = self.expr(node.left, env)
            for op, other in zip(node.ops, node.comparators):
                right = self.expr(other, env)
                if not self.compare[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.expr(node.body if self.expr(node.test, env) else node.orelse, env)
        if isinstance(node, ast.Call):
            return self.call(node.func.id, [self.expr(n, env) for n in node.args])
        if isinstance(node, ast.Subscript):
            return self.expr(node.value, env)[self.expr(node.slice, env)]
        if isinstance(node, ast.Slice):
            return slice(*(self.expr(n, env) if n else None for n in (node.lower, node.upper, node.step)))
        raise UnsafeCode(f'unsupported expression {type(node).__name__}')

    def assign(self, target, value, env):
        if isinstance(target, ast.Name):
            env[target.id] = self.bounded(value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if len(target.elts) != len(value):
                raise UnsafeCode('unpacking mismatch')
            for node, item in zip(target.elts, value):
                self.assign(node, item, env)
        else:
            raise UnsafeCode('unsupported assignment')

    def block(self, nodes, env):
        for node in nodes:
            self.tick()
            if isinstance(node, ast.Return):
                return True, self.expr(node.value, env) if node.value else None
            if isinstance(node, ast.Assign):
                value = self.expr(node.value, env)
                for target in node.targets:
                    self.assign(target, value, env)
            elif isinstance(node, ast.AugAssign):
                if not isinstance(node.target, ast.Name):
                    raise UnsafeCode('unsupported augmented assignment')
                current = ast.BinOp(ast.Name(node.target.id, ast.Load()), node.op, node.value)
                self.assign(node.target, self.expr(current, env), env)
            elif isinstance(node, ast.If):
                returned, result = self.block(node.body if self.expr(node.test, env) else node.orelse, env)
                if returned:
                    return returned, result
            elif isinstance(node, ast.For):
                for value in self.expr(node.iter, env):
                    self.tick()
                    self.assign(node.target, value, env)
                    returned, result = self.block(node.body, env)
                    if returned:
                        return returned, result
                returned, result = self.block(node.orelse, env)
                if returned:
                    return returned, result
            elif isinstance(node, ast.Expr):
                self.expr(node.value, env)
            elif not isinstance(node, ast.Pass):
                raise UnsafeCode('unsupported statement')
        return False, None
