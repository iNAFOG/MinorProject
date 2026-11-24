from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


class Objects:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, value):
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')
        ref = hashlib.sha256(raw).hexdigest()
        path = self.root / f'{ref}.json'
        if not path.exists():
            path.write_bytes(raw)
        return ref

    def get(self, ref):
        if len(ref) != 64 or any(c not in '0123456789abcdef' for c in ref):
            raise ValueError('invalid object reference')
        raw = (self.root / f'{ref}.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref:
            raise ValueError('object integrity failure')
        return json.loads(raw)


class Trajectory:
    def __init__(self, root, task_id, version, config):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / 'trajectories.jsonl'
        self.objects = Objects(root / 'objects')
        self.task_id, self.episode_id = task_id, uuid4().hex
        self.events = []
        self.log('start', harness_version=version, configuration=config)

    def log(self, event, **data):
        row = dict(task_id=self.task_id, episode_id=self.episode_id,
                   timestamp=now(), event=event, **data)
        self.events.append(row)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        return row
