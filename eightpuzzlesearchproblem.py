import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import heapq

class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.counter = 0  

    def put(self, priority, item):
        """Insert item into the priority queue with the given priority."""
        heapq.heappush(self.elements, (priority, self.counter, item)) 
        self.counter += 1  

    def get(self):
        """Remove and return the item with the highest priority (smallest priority value)."""
        return heapq.heappop(self.elements)[2] 

    def empty(self):
        """Return True if the priority queue is empty."""
        return len(self.elements) == 0




# 8-Puzzle Class
class EightPuzzle:
    def __init__(self, initial_state):
        self.state = np.array(initial_state)
        self.blank_pos = np.argwhere(self.state == 0)[0]

    def move(self, direction):
        """Move the blank tile in a given direction if possible."""
        row, col = self.blank_pos
        if direction == 'up' and row > 0:
            self._swap((row, col), (row - 1, col))
        elif direction == 'down' and row < 2:
            self._swap((row, col), (row + 1, col))
        elif direction == 'left' and col > 0:
            self._swap((row, col), (row, col - 1))
        elif direction == 'right' and col < 2:
            self._swap((row, col), (row, col + 1))

    def _swap(self, pos1, pos2):
        """Swap two positions in the puzzle."""
        self.state[tuple(pos1)], self.state[tuple(pos2)] = self.state[tuple(pos2)], self.state[tuple(pos1)]
        self.blank_pos = pos2  # Update the blank position

    def is_solved(self):
        """Check if the puzzle is solved."""
        return np.array_equal(self.state, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]]))

    def copy(self):
        """Return a copy of the current puzzle."""
        return EightPuzzle(self.state.copy())

    def legal_moves(self):
        """Returns a list of legal moves from the current state."""
        row, col = self.blank_pos
        moves = []
        if row > 0:
            moves.append('up')
        if row < 2:
            moves.append('down')
        if col > 0:
            moves.append('left')
        if col < 2:
            moves.append('right')
        return moves

    def result(self, move):
        """Returns the resulting state from applying a move."""
        new_puzzle = self.copy()
        new_puzzle.move(move)
        return new_puzzle

    def __eq__(self, other):
        return np.array_equal(self.state, other.state)

    def __hash__(self):
        return hash(self.state.tobytes())

    def __str__(self):
        """String representation of the puzzle state."""
        return '\n'.join([' '.join(map(str, row)) for row in self.state]) + '\n'


# BFS Solver
def bfs_solve(puzzle):
    """Solve the 8-puzzle using BFS and return the sequence of moves."""
    frontier = deque([(puzzle.copy(), [])])  
    explored = set()
    explored_nodes = 0

    while frontier:
        current_puzzle, moves = frontier.popleft()

        if current_puzzle.is_solved():
            return moves, explored_nodes  

        if current_puzzle not in explored:
            explored.add(current_puzzle)
            explored_nodes += 1

            for move in current_puzzle.legal_moves():
                new_puzzle = current_puzzle.result(move)
                frontier.append((new_puzzle, moves + [move]))

    return [], explored_nodes 



def tilesnotinplace(puzzle):
    """Return the number of misplaced tiles."""
    goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    return np.sum(puzzle.state != goal_state) - 1  

def manhattandistance(puzzle):
    """Return the sum of the Manhattan distances of the tiles from their goal positions."""
    goal_positions = {1: (0, 0), 2: (0, 1), 3: (0, 2),
                      4: (1, 0), 5: (1, 1), 6: (1, 2),
                      7: (2, 0), 8: (2, 1), 0: (2, 2)}
    distance = 0
    for i in range(3):
        for j in range(3):
            tile = puzzle.state[i][j]
            if tile != 0:
                goal_i, goal_j = goal_positions[tile]
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance

def euclideandistance(puzzle):
    """Return the sum of the Euclidean distances of the tiles from their goal positions, ignoring the blank tile."""
    goal_positions = {1: (0, 0), 2: (0, 1), 3: (0, 2),
                      4: (1, 0), 5: (1, 1), 6: (1, 2),
                      7: (2, 0), 8: (2, 1), 0: (2, 2)}  

    distance = 0
    for i in range(3):
        for j in range(3):
            tile = puzzle.state[i][j]
            if tile != 0:  
                goal_i, goal_j = goal_positions[tile]
                distance += math.sqrt((i - goal_i) ** 2 + (j - goal_j) ** 2)
    return distance

import math
def infinitenorm(puzzle):
    """Return the sum of the Chebyshev (infinite norm) distances of the tiles from their goal positions."""
    goal_positions = {1: (0, 0), 2: (0, 1), 3: (0, 2),
                      4: (1, 0), 5: (1, 1), 6: (1, 2),
                      7: (2, 0), 8: (2, 1), 0: (2, 2)} 

    distance = 0
    for i in range(3):
        for j in range(3):
            tile = puzzle.state[i][j]
            if tile != 0:  
                goal_i, goal_j = goal_positions[tile]
                # Chebyshev distance
                distance += max(abs(i - goal_i), abs(j - goal_j))
    return distance
    
def heuristic_boules_frontieres(puzzle):
    """
    Heuristique basÃ©e sur des boules fermÃ©es pour le 8-puzzle.
    - 1 et 2 forment une premiÃ¨re boule
    - 3 et 4 forment une deuxiÃ¨me boule
    - 5 et 6 forment une troisiÃ¨me boule
    - 7 et 8 forment une quatriÃ¨me boule
    Si une tuile est Ã  la bonne place dans sa boule, on rend 0. Si elle est Ã  la frontiÃ¨re de sa boule, on rend 1. Si elle est Ã  l'extÃ©rieur, on rend la distance de Manhattan.
    """
    # Positions finales des tuiles
    goal_positions = {1: (0, 0), 2: (0, 1), 3: (0, 2),
                      4: (1, 0), 5: (1, 1), 6: (1, 2),
                      7: (2, 0), 8: (2, 1), 0: (2, 2)}  # La position de la case vide n'a pas d'importance

    # DÃ©finir les boules
    boules = {
        1: [(0, 0), (0, 1)],  # Boule 1 pour tuiles 1 et 2
        2: [(0, 0), (0, 1)],  # Boule 2 pour tuiles 1 et 2 (mÃªme boule)
        3: [(0, 2), (1, 0)],  # Boule 3 pour tuiles 3 et 4
        4: [(0, 2), (1, 0)],  # Boule 4 pour tuiles 3 et 4 (mÃªme boule)
        5: [(1, 1), (1, 2)],  # Boule 5 pour tuiles 5 et 6
        6: [(1, 1), (1, 2)],  # Boule 6 pour tuiles 5 et 6 (mÃªme boule)
        7: [(2, 0), (2, 1)],  # Boule 7 pour tuiles 7 et 8
        8: [(2, 0), (2, 1)]   # Boule 8 pour tuiles 7 et 8 (mÃªme boule)
    }

    total_cost = 0

    # Calculer le coÃ»t pour chaque tuile
    for i in range(3):
        for j in range(3):
            tile = puzzle.state[i][j]
            if tile != 0:  # Ignorer la case vide (0)
                goal_i, goal_j = goal_positions[tile]

                # Si la tuile est dans sa boule (intÃ©rieur), mais Ã  la mauvaise place
                if (i, j) in boules[tile]:
                    if (i, j) == (goal_i, goal_j):
                        # La tuile est bien placÃ©e
                        cost = 0
                    else:
                        # La tuile est dans sa boule mais Ã  la mauvaise place
                        cost = 0.5
                elif any((i, j) in boules[b] for b in boules if b != tile):
                    # Si la tuile est dans la mauvaise boule (frontiÃ¨re), attribuer un coÃ»t de 1
                    cost = 1
                else:
                    # Sinon, la tuile est Ã  l'extÃ©rieur (extÃ©rieur) et on calcule la distance de Manhattan
                    cost = abs(i - goal_i) + abs(j - goal_j)

                total_cost += cost

    return total_cost


def uniformcost(puzzle):
    return 0;


# A* Search Algorithm
def a_star_search(puzzle):
    """A* search to solve the 8-puzzle."""
    frontier = PriorityQueue()  
    frontier.put(0, (puzzle, []))  
    explored = set() 
    nodes_explored = 0  
    
    while not frontier.empty():
        current_puzzle, path = frontier.get() 
        
        if current_puzzle.is_solved():
            return path, nodes_explored
        
        if current_puzzle in explored:
            continue  
        
        explored.add(current_puzzle)  
        nodes_explored += 1 
        
        for move in current_puzzle.legal_moves():
            new_puzzle = current_puzzle.result(move)
            new_path = path + [move]
            g_cost = len(new_path)  
            h_cost = infinitenorm(new_puzzle)  
            f_cost = g_cost + h_cost  
            frontier.put(f_cost, (new_puzzle, new_path)) 
    
    return [], nodes_explored  




def print_solution_path(puzzle_initial_state, solution_moves):
    """Print the puzzle states along the solution path."""
    current_puzzle = EightPuzzle(puzzle_initial_state.copy())
    print("Initial state:")
    print(current_puzzle)

    for move in solution_moves:
        current_puzzle.move(move)
        print(f"After move {move}:")
        print(current_puzzle)


def update_display(puzzle, ax):
    """Update the puzzle display."""
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    for i in range(3):
        for j in range(3):
            value = puzzle.state[2 - i][j]  
            label = '' if value == 0 else str(value)
            ax.text(j + 0.5, i + 0.5, label, ha='center', va='center', fontsize=45, fontweight='bold',
                    bbox=dict(facecolor='lightgray' if value == 0 else 'white', 
                              edgecolor='black', boxstyle='round,pad=0.6', linewidth=2))

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1) 
    plt.draw()

def on_click(event, puzzle, ax, solution_moves, step_counter):
    if step_counter[0] < len(solution_moves):
        move = solution_moves[step_counter[0]]
        puzzle.move(move)
        update_display(puzzle, ax)
        step_counter[0] += 1

def manual_animation_with_button(puzzle_initial_state, solution_moves):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax_next = plt.axes([0.8, 0.02, 0.15, 0.07])
    next_btn = Button(ax_next, 'Next')

    puzzle = EightPuzzle(puzzle_initial_state.copy())

    step_counter = [0]

    next_btn.on_clicked(lambda event: on_click(event, puzzle, ax, solution_moves, step_counter))

    update_display(puzzle, ax)
    plt.show()

if __name__ == "__main__":
    initial_state = [[8, 1, 6],
                     [5, 4, 7],
                     [2, 3, 0]]  

    puzzle1 = EightPuzzle(initial_state)

    puzzle = EightPuzzle(initial_state)

    print("Solving with A* Search...")
    solution_moves, nodes_explored = a_star_search(puzzle)

    if solution_moves:
        print(f"Solution found! Moves: {solution_moves}")
        print(f"Number of nodes explored: {nodes_explored}")

        manual_animation_with_button(initial_state, solution_moves)
        print("the len is: " ,len(solution_moves))
    else:
        print("No solution found or the puzzle is unsolvable.")