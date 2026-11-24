"""SQLite-backed H=(strategy, agents, skills, memory), with controller-only writes."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import sqlite3
from uuid import uuid4
from agent_harness.core.trajectory import now


KINDS = ('memory', 'skill', 'agent', 'note')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


class State:
    def __init__(self, path=':memory:', initial=None):
        self.path = str(path)
        if self.path != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS entries(kind TEXT, id TEXT, data TEXT,
                PRIMARY KEY(kind,id));
            CREATE TABLE IF NOT EXISTS versions(version INTEGER PRIMARY KEY,
                parent INTEGER, state TEXT, manifest TEXT, digest TEXT);
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS audit(sequence INTEGER PRIMARY KEY,
                event TEXT, previous TEXT, digest TEXT);
            CREATE TABLE IF NOT EXISTS experience(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE, data TEXT);
        ''')
        if self.db.execute("SELECT 1 FROM metadata WHERE key='version'").fetchone() is None:
            self.db.execute("INSERT INTO metadata VALUES('version','0')")
            if initial:
                for kind, entries in initial.items():
                    for entry in entries:
                        self.db.execute('INSERT INTO entries VALUES(?,?,?)', (kind, entry['id'], canonical(entry)))
            self.db.commit()
            self._save_version(0, None, [], {'initial': True})
        from agent_harness.memory.sqlite_store import SQLiteMemory
        from agent_harness.skills.registry import SkillRegistry
        from agent_harness.agents.registry import AgentRegistry
        self.memory = SQLiteMemory(self)
        self.skills = SkillRegistry(self)
        self.agents = AgentRegistry(self)

    def close(self):
        self.db.close()

    def record_experience(self, record):
        if record['split'] != 'adaptation':
            raise PermissionError('only adaptation experience may enter the learning archive')
        with self.db:
            self.db.execute('INSERT INTO experience(task_id,data) VALUES(?,?) '
                            'ON CONFLICT(task_id) DO UPDATE SET data=excluded.data',
                            (record['task_id'],canonical(record)))

    def experiences(self):
        return [json.loads(row[0]) for row in self.db.execute('SELECT data FROM experience ORDER BY sequence')]

    @property
    def version(self):
        return int(self.db.execute("SELECT value FROM metadata WHERE key='version'").fetchone()[0])

    def entries(self, kind):
        return [json.loads(row[0]) for row in self.db.execute(
            'SELECT data FROM entries WHERE kind=? ORDER BY id', (kind,))]

    def get(self, kind, entry_id):
        row = self.db.execute('SELECT data FROM entries WHERE kind=? AND id=?', (kind, entry_id)).fetchone()
        if row is None:
            raise KeyError(entry_id)
        return json.loads(row[0])

    def put(self, kind, data):
        if kind not in KINDS:
            raise ValueError('only external harness entries are mutable')
        if not data.get('source') or not data.get('provenance'):
            raise ValueError('persistent state requires source and provenance')
        data = dict(data)
        data.setdefault('id', uuid4().hex)
        data.setdefault('timestamp', now())
        data.setdefault('version', 1)
        self.db.execute('INSERT OR REPLACE INTO entries VALUES(?,?,?)', (kind, data['id'], canonical(data)))
        self.db.commit()
        return data

    def delete(self, kind, entry_id):
        self.get(kind, entry_id)
        self.db.execute('DELETE FROM entries WHERE kind=? AND id=?', (kind, entry_id))
        self.db.commit()

    def notes(self):
        return [e['content'] for e in self.entries('note') if e.get('status') == 'active']

    def export(self):
        return {kind: self.entries(kind) for kind in KINDS}

    def clone(self):
        return State(initial=self.export())

    def fingerprint(self):
        return hashlib.sha256(canonical(self.export()).encode()).hexdigest()

    def _save_version(self, version, parent, changes, verification):
        snapshot = canonical(self.export())
        manifest = dict(version_id=version, parent_version=parent, change_set=changes,
                        verification_result=verification, creation_time=now())
        digest = hashlib.sha256(snapshot.encode()).hexdigest()
        self.db.execute('INSERT INTO versions VALUES(?,?,?,?,?)',
                        (version, parent, snapshot, canonical(manifest), digest))
        self.db.commit()
        if self.path != ':memory:':
            folder = Path(self.path).parent/'harness'/f'v{version:04d}'
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'state.json').write_text(snapshot, encoding='utf-8')
            (folder/'manifest.json').write_text(canonical(manifest | {'digest': digest}), encoding='utf-8')

    def promote(self, candidate, changes, verification):
        snapshot = candidate.export()
        parent = self.version
        version = self.db.execute('SELECT MAX(version) FROM versions').fetchone()[0] + 1
        # State, version metadata and its immutable DB snapshot share one transaction.
        manifest = dict(version_id=version, parent_version=parent, change_set=changes,
                        verification_result=verification, creation_time=now())
        raw = canonical(snapshot)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        with self.db:
            self.db.execute('DELETE FROM entries')
            for kind, entries in snapshot.items():
                self.db.executemany('INSERT INTO entries VALUES(?,?,?)',
                                    [(kind, e['id'], canonical(e)) for e in entries])
            self.db.execute("UPDATE metadata SET value=? WHERE key='version'", (str(version),))
            self.db.execute('INSERT INTO versions VALUES(?,?,?,?,?)',
                            (version, parent, raw, canonical(manifest), digest))
        if self.path != ':memory:':
            folder = Path(self.path).parent/'harness'/f'v{version:04d}'
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'state.json').write_text(raw, encoding='utf-8')
            (folder/'manifest.json').write_text(canonical(manifest | {'digest': digest}), encoding='utf-8')
        self.audit('promote', {'version': version, 'parent': parent, 'verification': verification})
        return version

    def history(self):
        return [json.loads(row[0]) for row in self.db.execute('SELECT manifest FROM versions ORDER BY version')]

    def rollback(self, version):
        row = self.db.execute('SELECT state,digest FROM versions WHERE version=?', (version,)).fetchone()
        if not row:
            raise KeyError(f'unknown version {version}')
        if hashlib.sha256(row['state'].encode()).hexdigest() != row['digest']:
            raise ValueError('snapshot integrity failure')
        snapshot = json.loads(row['state'])
        with self.db:
            self.db.execute('DELETE FROM entries')
            for kind, entries in snapshot.items():
                self.db.executemany('INSERT INTO entries VALUES(?,?,?)',
                                    [(kind, e['id'], canonical(e)) for e in entries])
            self.db.execute("UPDATE metadata SET value=? WHERE key='version'", (str(version),))
        self.audit('rollback', {'version': version})

    def audit(self, event, details):
        previous = self.db.execute('SELECT digest FROM audit ORDER BY sequence DESC LIMIT 1').fetchone()
        previous = previous[0] if previous else '0'*64
        raw = canonical({'event': event, 'details': details, 'timestamp': now()})
        digest = hashlib.sha256((previous+raw).encode()).hexdigest()
        with self.db:
            self.db.execute('INSERT INTO audit(event,previous,digest) VALUES(?,?,?)', (raw, previous, digest))

    def audit_rows(self):
        rows, previous = [], '0'*64
        for row in self.db.execute('SELECT * FROM audit ORDER BY sequence'):
            if row['previous'] != previous or hashlib.sha256((previous+row['event']).encode()).hexdigest() != row['digest']:
                raise ValueError('audit integrity failure')
            previous = row['digest']
            rows.append(json.loads(row['event']))
        return rows
