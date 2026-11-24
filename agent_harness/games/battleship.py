"""Hidden-fleet search: a compact, single-player Battleship salvo variant.

The private fleet is sampled once. Conditional on that fleet, shots are
deterministic; uncertainty comes from concealed random placement, not dice per shot.
"""
import random


class Battleship:
    name = 'battleship'
    rules = (
        'Play a single-player Battleship search variant on a 5x5 hidden board. '
        'Cells are integers 0..24 in row-major order. Two straight horizontal or '
        'vertical ships occupy 3 and 2 cells. Ships do not overlap but may touch. '
        'The hidden fleet is randomly placed once and stays fixed. '
        'Use fire(cells) with EXACTLY THREE distinct, previously untested cells per salvo. '
        'All three choices are committed before feedback: no within-salvo adaptation. '
        'You have FIVE salvos (at most 15 shots). A malformed/repeated shot batch '
        'wastes one salvo without firing. The board shows ? unknown, o miss, X hit. '
        'Only a fully sunk ship has its cells revealed; other ship locations are secret. '
        'Win by hitting all five occupied cells within the limit. Partial score is '
        'occupied cells hit divided by five; it is not a win. Use hit/miss feedback '
        'and sunk-ship information to choose subsequent salvos. Do not finish early.'
    )

    def __init__(self, seed, **unused):
        rng = random.Random(seed)
        self.size, self.max_steps, self.salvo_size = 5, 5, 3
        self._ships = []
        occupied = set()
        for length in (3, 2):
            placements = []
            for row in range(5):
                for col in range(5):
                    for dx, dy in ((1, 0), (0, 1)):
                        if col + dx*(length-1) < 5 and row + dy*(length-1) < 5:
                            cells = tuple((row+dy*i)*5+col+dx*i for i in range(length))
                            if not occupied.intersection(cells):
                                placements.append(cells)
            ship = rng.choice(placements)
            self._ships.append(ship)
            occupied.update(ship)
        self._occupied = occupied
        self._shots = set()
        self.steps, self.illegal = 0, 0
        self.terminal, self.outcome = False, 'in_progress'
        self.history = []

    def observe(self):
        hits = self._shots & self._occupied
        sunk = [list(ship) for ship in self._ships if set(ship) <= self._shots]
        board = ['X' if c in hits else 'o' if c in self._shots else '?' for c in range(25)]
        return {'game': self.name, 'size': 5, 'board': board,
                'rows': [''.join(board[i:i+5]) for i in range(0, 25, 5)],
                'fleet_lengths': [3, 2], 'sunk_ships': sunk,
                'legal_cells': [c for c in range(25) if c not in self._shots],
                'salvo_size': 3, 'salvos_used': self.steps, 'max_salvos': 5,
                'hits': len(hits), 'shots_fired': len(self._shots),
                'illegal_moves': self.illegal, 'terminal': self.terminal, 'outcome': self.outcome}

    def step(self, cells):
        if self.terminal:
            return self.observe() | {'error': 'game already ended'}
        self.steps += 1
        valid = (isinstance(cells, list) and len(cells) == 3
                 and all(type(c) is int and 0 <= c < 25 and c not in self._shots for c in cells)
                 and len(set(cells)) == 3)
        results = []
        error = None
        if valid:
            for cell in cells:
                self._shots.add(cell)
                results.append({'cell': cell, 'result': 'hit' if cell in self._occupied else 'miss'})
        else:
            self.illegal += 1
            error = 'fire requires exactly three distinct, previously untested integer cells 0..24'
        if self._occupied <= self._shots:
            self.terminal, self.outcome = True, 'win'
        elif self.steps == self.max_steps:
            self.terminal, self.outcome = True, 'loss'
        obs = self.observe()
        self.history.append({'cells': cells, 'results': results, 'error': error, 'observation': obs})
        return obs | ({'error': error} if error else {})

    def metrics(self):
        obs = self.observe()
        return {'game': self.name, 'outcome': self.outcome, 'game_score': obs['hits']/5,
                'steps': self.steps, 'illegal_moves': self.illegal, 'hits': obs['hits'],
                'shots_fired': obs['shots_fired'], 'ships_sunk': len(obs['sunk_ships']),
                'hit_rate': obs['hits']/max(1, obs['shots_fired']),
                'path_efficiency': None}


def reference_salvo(observation, mode='hunt_target', seed=0):
    """Non-learning controls consuming public observations ONLY; never an oracle.

    Hunt/target prioritizes extensions of unresolved hit runs, then a checkerboard
    search. This is a reference strategy, not the learned harness or a smoke gain.
    """
    legal = observation['legal_cells']
    if mode == 'random':
        return random.Random(seed).sample(legal, 3)
    if mode != 'hunt_target':
        raise ValueError('unknown reference controller')
    sunk = {c for ship in observation['sunk_ships'] for c in ship}
    hits = {c for c, value in enumerate(observation['board']) if value == 'X'} - sunk
    def priority(cell):
        row, col = divmod(cell, 5)
        adjacent = sum(abs(row-h//5)+abs(col-h%5) == 1 for h in hits)
        extension = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            near, far = (row+dy, col+dx), (row+2*dy, col+2*dx)
            if all(0 <= x < 5 and 0 <= y < 5 for y, x in (near, far)):
                extension += near[0]*5+near[1] in hits and far[0]*5+far[1] in hits
        parity = (row+col) % 2 == 0
        center = -abs(row-2)-abs(col-2)
        return extension, adjacent, parity, center, -cell
    return sorted(legal, key=priority, reverse=True)[:3]
