from dataclasses import asdict
from pathlib import Path
import subprocess
from .base import Environment, VerificationResult
from .safe_python import SafePython


class CodingEnvironment(Environment):
    def __init__(self, task, root, limits, allow_local_execution=False):
        self.task = task
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        self.allow_local_execution = allow_local_execution
        for path, content in task.files.items():
            dest = self.path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding='utf-8')

    def path(self, relative):
        if not isinstance(relative, str) or ':' in relative or relative.startswith(('/', '\\')):
            raise PermissionError('relative paths required')
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise PermissionError('path escapes task workspace')
        if any(p.startswith('.') for p in Path(relative).parts):
            raise PermissionError('hidden control paths are inaccessible')
        return path

    def read(self, path):
        target = self.path(path)
        with target.open(encoding='utf-8') as f:
            return f.read(self.limits.max_output_chars)

    def write(self, path, content):
        if path not in self.task.writable_paths:
            raise PermissionError('path not in immutable writable manifest')
        if len(content) > 50000:
            raise ValueError('write size limit')
        target = self.path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return {'written': path, 'chars': len(content)}

    def search(self, query):
        if not isinstance(query, str) or len(query) > 500:
            raise ValueError('invalid search query')
        found = []
        for path in sorted(self.root.rglob('*')):
            if len(found) >= 30:
                break
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(self.root):
                relative = path.relative_to(self.root).as_posix()
                try:
                    content = self.read(relative)
                except (UnicodeError, PermissionError):
                    continue
                if query.lower() in relative.lower() or query.lower() in content.lower():
                    found.append({'path': relative, 'preview': content[:300]})
        return found

    def execute(self, argv):
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise ValueError('execute requires an argv list; shell is disabled')
        if not self.allow_local_execution:
            raise PermissionError('local process execution is disabled; enable only for trusted repositories')
        command = argv
        # Drain output continuously while retaining a bounded preview in memory.
        from threading import Thread
        import os
        retained = bytearray()
        process = subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        def drain():
            while chunk := process.stdout.read(4096):
                remaining = self.limits.max_output_chars-len(retained)
                if remaining > 0:
                    retained.extend(chunk[:remaining])
        reader=Thread(target=drain,daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=min(self.limits.max_execution_time,30))
        except subprocess.TimeoutExpired:
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=5)
            else:
                process.kill()
            process.wait(timeout=5)
            raise TimeoutError('process time limit')
        finally:
            reader.join(timeout=2)
        text=bytes(retained).decode('utf-8',errors='replace')
        return {'returncode': code, 'output': text}

    def _read_for_check(self, path):
        # Controller-side checks see the whole file; agent reads stay bounded by max_output_chars.
        return self.path(path).read_text(encoding='utf-8')

    def verify(self):
        passed, errors = 0, []
        for index, check in enumerate(self.task.checks):
            try:
                if check['kind'] == 'function':
                    engine = SafePython(self._read_for_check(check['path']))
                    for case in check['cases']:
                        result = engine.invoke(check['function'], case['args'])
                        if result != case['expected']:
                            raise AssertionError('output mismatch')
                elif check['kind'] == 'contains':
                    if check['text'] not in self._read_for_check(check['path']):
                        raise AssertionError('required content missing')
                elif check['kind'] == 'command':
                    if self.execute(check['argv'])['returncode']:
                        raise AssertionError('repository checks failed')
                else:
                    raise ValueError('unknown verification kind')
                passed += 1
            except Exception as exc:
                errors.append(f'check {index}: {type(exc).__name__}: {exc}')
        return VerificationResult(passed == len(self.task.checks), passed, len(self.task.checks), tuple(errors))
