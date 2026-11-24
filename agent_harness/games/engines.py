"""Authoritative game rules. No model output can alter rewards or legality."""
from collections import deque
from functools import lru_cache
import random
from .battleship import Battleship


LINES = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))


def winner(board):
    for a,b,c in LINES:
        if board[a] != '.' and board[a] == board[b] == board[c]:
            return board[a]
    return None


@lru_cache(maxsize=20000)
def minimax_value(board, turn):
    won = winner(board)
    if won:
        return 1 if won == turn else -1
    free = [i for i,c in enumerate(board) if c == '.']
    if not free:
        return 0
    other = 'O' if turn == 'X' else 'X'
    return max(-minimax_value(board[:i]+turn+board[i+1:], other) for i in free)


class TicTacToe:
    name = 'tic_tac_toe'
    rules = ('Play standard 3x3 tic-tac-toe. Cells are 0..8 in row-major order. '
             'Place your mark in an empty cell with move(cell). Three marks in a row, '
             'column or diagonal win. A full board without a winner is a draw. '
             'The opponent moves automatically after each legal agent move. '
             'Win if possible and otherwise avoid losing. Only use legal cells. '
             'Three illegal requests forfeit the episode. Do not claim completion early.')

    def __init__(self, seed, opponent='tactical', agent_mark='X', **unused):
        if opponent not in {'random','tactical','perfect'} or agent_mark not in {'X','O'}:
            raise ValueError('invalid opponent or player mark')
        self.rng, self.opponent, self.agent_mark = random.Random(seed), opponent, agent_mark
        self.other = 'O' if agent_mark == 'X' else 'X'
        self.board = ['.']*9
        self.terminal, self.outcome, self.steps, self.illegal = False, 'in_progress', 0, 0
        self.history = []
        if self.other == 'X':
            self._opponent_move()

    def _finish(self):
        won = winner(self.board)
        if won or '.' not in self.board:
            self.terminal = True
            self.outcome = 'win' if won == self.agent_mark else 'loss' if won else 'draw'

    def _place(self, cell, mark):
        self.board[cell] = mark
        self.history.append({'mark':mark, 'cell':cell})
        self._finish()

    def _opponent_move(self):
        legal = [i for i,c in enumerate(self.board) if c == '.']
        if self.opponent == 'perfect':
            scored = [(i, -minimax_value(''.join(self.board[:i])+self.other+''.join(self.board[i+1:]),self.agent_mark)) for i in legal]
            best = max(v for _,v in scored)
            cell = self.rng.choice([i for i,v in scored if v == best])
        elif self.opponent == 'tactical':
            cell = None
            for mark in (self.other, self.agent_mark):
                winning = [i for i in legal if winner(self.board[:i]+[mark]+self.board[i+1:]) == mark]
                if winning:
                    cell = self.rng.choice(winning)
                    break
            if cell is None:
                preferred = [i for i in (4,0,2,6,8) if i in legal]
                cell = self.rng.choice(preferred or legal)
        else:
            cell = self.rng.choice(legal)
        self._place(cell,self.other)

    def step(self, cell):
        if self.terminal:
            raise ValueError('game already ended')
        self.steps += 1
        if type(cell) is not int or not 0 <= cell < 9 or self.board[cell] != '.':
            self.illegal += 1
            if self.illegal >= 3:
                self.terminal,self.outcome = True,'illegal_move_forfeit'
            return self.observe() | {'error':'Choose an empty cell from legal_cells.'}
        self._place(cell,self.agent_mark)
        if not self.terminal:
            self._opponent_move()
        return self.observe()

    def observe(self):
        return {'game':self.name,'board':list(self.board),'rows':[''.join(self.board[i:i+3]) for i in (0,3,6)],
                'agent_mark':self.agent_mark,'opponent_mark':self.other,
                'legal_cells':[] if self.terminal else [i for i,c in enumerate(self.board) if c == '.'],
                'terminal':self.terminal,'outcome':self.outcome,'moves_requested':self.steps,
                'illegal_moves':self.illegal,'history':list(self.history)}

    def metrics(self):
        reward = {'win':1.0,'draw':0.5}.get(self.outcome,0.0)
        return {'game':self.name,'outcome':self.outcome,'game_score':reward,'win':self.outcome=='win',
                'draw':self.outcome=='draw','loss':self.outcome=='loss','steps':self.steps,'illegal_moves':self.illegal,
                'opponent':self.opponent,'agent_mark':self.agent_mark}


DIRECTIONS = {'up':(0,-1),'down':(0,1),'left':(-1,0),'right':(1,0)}


def shortest_path(grid, start, has_key=False):
    """Evaluation oracle and smoke-controller utility, never included in agent prompts."""
    queue = deque([(tuple(start),has_key,[])])
    seen = {(tuple(start),has_key)}
    while queue:
        pos,key,path = queue.popleft()
        if grid[pos[1]][pos[0]] == 'G':
            return path
        for direction,(dx,dy) in DIRECTIONS.items():
            x,y = pos[0]+dx,pos[1]+dy
            if not (0 <= y < len(grid) and 0 <= x < len(grid[0])):
                continue
            cell = grid[y][x]
            if cell == '#' or cell == 'D' and not key:
                continue
            new_key = key or cell == 'K'
            state = ((x,y),new_key)
            if state not in seen:
                seen.add(state)
                queue.append(((x,y),new_key,path+[direction]))
    return None


class KeyDoor:
    name = 'key_door'
    rules = ('Reach G on the grid using move(direction), where direction is up, down, left or right. '
             'Coordinates are [x,y], x increases right and y increases down. '
             '# is an impassable wall, . is floor, K is the key, D is a locked door, G is the goal. '
             'Entering K collects the key. Entering D requires the key and opens the door. '
             'The key is reusable. Each move request costs one step, including illegal moves. '
             'The full current map and position are visible. Plan a legal route within the step limit. '
             'The board and rule engine alone decide success.')

    def __init__(self, seed, size=7, max_steps=48, **unused):
        if size not in (7,9) or max_steps < 1:
            raise ValueError('supported grid sizes are 7 and 9; positive step limit required')
        rng = random.Random(seed)
        middle = size//2
        for attempt in range(1000):
            grid = [['#' if x in (0,size-1) or y in (0,size-1) or x==middle else '.'
                     for x in range(size)] for y in range(size)]
            door = (middle,rng.randrange(1,size-1))
            grid[door[1]][door[0]]='D'
            left = [(x,y) for y in range(1,size-1) for x in range(1,middle)]
            right = [(x,y) for y in range(1,size-1) for x in range(middle+1,size-1)]
            start,key = rng.sample(left,2)
            goal = rng.choice(right)
            protected = {start,key,goal,door}
            for x,y in left+right:
                if (x,y) not in protected and rng.random() < 0.12:
                    grid[y][x]='#'
            grid[key[1]][key[0]],grid[goal[1]][goal[0]]='K','G'
            route = shortest_path(grid,start)
            if route and len(route) <= max_steps:
                break
        else:
            raise ValueError('could not generate a solvable layout')
        self.grid,self.position,self.max_steps = grid,start,max_steps
        self.optimal_steps = len(route)
        self.has_key,self.terminal,self.outcome = False,False,'in_progress'
        self.steps,self.illegal,self.history = 0,0,[]

    def step(self,direction):
        if self.terminal:
            raise ValueError('game already ended')
        self.steps += 1
        error = None
        if direction not in DIRECTIONS:
            error = 'Use up, down, left or right.'
        else:
            dx,dy = DIRECTIONS[direction]
            x,y = self.position[0]+dx,self.position[1]+dy
            cell = self.grid[y][x]
            if cell == '#' or cell == 'D' and not self.has_key:
                error = 'A wall or locked door blocks this move.'
            else:
                self.position=(x,y)
                if cell == 'K':
                    self.has_key=True
                    self.grid[y][x]='.'
                elif cell == 'D':
                    self.grid[y][x]='.'
                elif cell == 'G':
                    self.terminal,self.outcome=True,'goal'
        if error:
            self.illegal += 1
        if not self.terminal and self.steps >= self.max_steps:
            self.terminal,self.outcome=True,'step_limit'
        self.history.append({'direction':direction,'position':list(self.position),'error':error})
        return self.observe() | ({'error':error} if error else {})

    def observe(self):
        legal=[]
        for name,(dx,dy) in DIRECTIONS.items():
            x,y=self.position[0]+dx,self.position[1]+dy
            cell=self.grid[y][x]
            if cell != '#' and (cell != 'D' or self.has_key):
                legal.append(name)
        return {'game':self.name,'grid':[''.join(row) for row in self.grid],
                'position':list(self.position),'has_key':self.has_key,
                'legal_directions':[] if self.terminal else legal,'steps':self.steps,'max_steps':self.max_steps,
                'terminal':self.terminal,'outcome':self.outcome,'illegal_moves':self.illegal}

    def metrics(self):
        solved=self.outcome=='goal'
        return {'game':self.name,'outcome':self.outcome,'game_score':float(solved),'win':solved,'draw':False,
                'loss':not solved,'steps':self.steps,'illegal_moves':self.illegal,
                'optimal_steps':self.optimal_steps,'path_efficiency':self.optimal_steps/self.steps if solved else 0.0,
                'size':len(self.grid),'opponent':None,'agent_mark':None}


def make_game(metadata):
    factories={'tic_tac_toe':TicTacToe,'key_door':KeyDoor,'battleship':Battleship}
    if metadata.get('game') not in factories:
        raise ValueError('unknown game')
    return factories[metadata['game']](**{k:v for k,v in metadata.items() if k!='game'})
