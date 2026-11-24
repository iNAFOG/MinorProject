from .coding import CodingEnvironment


class MockEnvironment(CodingEnvironment):
    """File-backed fixture environment with the same independent AST checks."""
    def execute(self, argv):
        if argv == ['check']:
            from dataclasses import asdict
            return asdict(self.verify())
        raise PermissionError('mock environment only supports check')
