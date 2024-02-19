from utils import *
from itertools import combinations
import time
from utils import Instance, List, Tuple
from collections import deque
import sys

class CustomWall(Wall):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """
    def __init__(self, w, h):
        super().__init__(w, h)
        self.capacity_matrix: List[List[Tuple[int,int]]] = [
            [(w - x, h - y) for y in range(h)]
            for x in range(w)
        ]
        self.available_tile_count = w * h

class CustomSolution(Solution):
    def __init__(self,items: List[Tuple[int,int,int,int]], instance: Instance):
        super().__init__(items)
        self.placed_art_ids = set([iter[0] for iter in items])
        '''
        The dictionary key is the wall id and the values are a 2D matrix
        where each cell contains two values wrapped in a tuple. The first
        value is the number of vacant cells to the right and the second
        is the number of vacant cells above.
        '''
        wall_w, wall_h = instance.wall.width(), instance.wall.height()
        self.walls: List[CustomWall] = [
            CustomWall(wall_w, wall_h) 
            for _ in range(len(self.items_by_wall.keys()))
        ]
        for item in items:
            art_id, wall_id, art_x, art_y = item
            art_w, art_h = instance.artpieces[art_id - 1].width(), instance.artpieces[art_id - 1].height()
            self.walls[wall_id].available_tile_count -= art_w * art_h

            # Sets the block as being taken
            for x in range(art_x, art_x + art_w):
                for y in range(art_y, art_y + art_h):
                    self.walls[wall_id].capacity_matrix[x][y] = (0,0)
            
            # TODO optimize this (Currently never used so ignore it)
            # Adjusts left of the block
            for y in range(art_y, art_y + art_h):
                for x in range(art_x - 1, -1, -1): # Goes backwards so we can skip some tiles
                    space_x, space_y = self.walls[wall_id].capacity_matrix[x][y]
                    # we can skip the rest of this row if we find another painting
                    if space_x == 0:
                        break
                    self.walls[wall_id].capacity_matrix[x][y] = (art_x - x, space_y)
            
            # Adjusts bottom of the block
            for x in range(art_x, art_x + art_w):
                for y in range(art_y - 1, -1, -1): # Goes backwards so we can skip some tiles
                    space_x, space_y = self.walls[wall_id].capacity_matrix[x][y]
                    # we can skip the rest of this row if we find another painting
                    if space_x == 0:
                        break
                    self.walls[wall_id].capacity_matrix[x][y] = (space_x, art_y - y)

    def try_place(self, instance: Instance, art_id: int, wall_id: int, art_x: int, art_y: int) -> bool:
        space_x, space_y = self.walls[wall_id].capacity_matrix[art_x][art_y]
        if space_x < instance.artpieces[art_id - 1].width() or \
            space_y < instance.artpieces[art_id - 1].height():
            return False
        self.place(instance, art_id, wall_id, art_x, art_y)
        return True

    def place(self, instance: Instance, art_id: int, wall_id: int, art_x: int, art_y: int):
        added_item = (art_id, wall_id, art_x, art_y)
        self.items.append(added_item)
        self.items_by_wall[wall_id].append(added_item)
        self.placed_art_ids.add(art_id)
        
        art_w = instance.artpieces[art_id - 1].width()
        art_h = instance.artpieces[art_id - 1].height()
        self.walls[wall_id].available_tile_count -= art_w * art_h
            
        # Sets the block as being taken
        for x in range(art_x, art_x + art_w):
            for y in range(art_y, art_y + art_h):
                self.walls[wall_id].capacity_matrix[x][y] = (0,0)
        
        # Adjusts left of the block
        for y in range(art_y, art_y + art_h):
            for x in range(art_x - 1, -1, -1): # Goes backwards so we can skip some tiles
                space_x, space_y = self.walls[wall_id].capacity_matrix[x][y]
                # we can skip the rest of this row if we find another painting
                if space_x == 0:
                    break
                self.walls[wall_id].capacity_matrix[x][y] = (art_x - x, space_y)
        
        # Adjusts bottom of the block
        for x in range(art_x, art_x + art_w):
            for y in range(art_y - 1, -1, -1): # Goes backwards so we can skip some tiles
                space_x, space_y = self.walls[wall_id].capacity_matrix[x][y]
                # we can skip the rest of this row if we find another painting
                if space_x == 0:
                    break
                self.walls[wall_id].capacity_matrix[x][y] = (space_x, art_y - y)
    
    def evaluate(self):
        # Bad since the ratio only decreases if # walls goes down
        # Sum of empty tiles
        # self.score = sum([wall.available_tile_count for wall in self.walls])
        
        # Bad since the ratio only decreases if # walls goes down
        # Ratio of empty tiles
        # wall_area = self.walls[0].width() * self.walls[0].height()
        # self.score =  sum([wall.available_tile_count for wall in self.walls]) / (wall_area * len(self.walls))
        
        # Not representative enough and can lead to filling the last wall, not emptying it
        # Last wall empty tiles
        # wall_area = self.walls[0].width() * self.walls[0].height()
        # self.score =  self.walls[-1].available_tile_count
        
        # Sum of empty tiles before last wall
        self.score = sum([wall.available_tile_count for wall in self.walls[:-1]])
        
        # Sum of empty tiles weighted by wall number
        # n = len(self.walls)
        # self.score = sum([wall.available_tile_count*(n - iter) for iter, wall in enumerate(self.walls[:-1])])

class TabuSearch():
    def __init__(self):
        self.MEMORY = 5
        
        self.best_neighbor: CustomSolution
        self.best_score: int = sys.maxsize
        self.ignored_art_ids: deque[int] = deque([])

def solve(instance: Instance) -> Solution:
    """
    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of tuples of the form (<artipiece_id>, <wall_id>, <x_pos>, <y_pos>)
    """
    TIME_IN_SECONDS = 300 # Change this to control the execution time
    END_TIME = time.time() + TIME_IN_SECONDS

    # Initial solution following intuitive heuristic
    ordered_art_pieces = {
        id:
        max((instance.artpieces[id - 1].width()/instance.wall.width()),
        (instance.artpieces[id - 1].height()/instance.wall.height()))
        for id in range(1, instance.n + 1)
    }
    ordered_art_ids = [iter[0] for iter in sorted(ordered_art_pieces.items(), key= lambda item: -item[1])]
    
    # Randomized initial solution
    # ordered_art_ids = list(range(1, instance.n + 1))
    # random.shuffle(ordered_art_ids)

    current_custom_solution: CustomSolution = place_pieces_in_order(instance, ordered_art_ids)
    best_solution: Solution = Solution(current_custom_solution.items)
    best_score = current_custom_solution.score
    print(f"{best_solution.nwalls} walls with a score of {best_score} with {best_solution.nwalls} walls")
    
    while time.time() < END_TIME:
        current_custom_solution = swap_last_local_search(current_custom_solution, instance, END_TIME)
        # This is added to provide an anytime solution
        # print(f"Time left {END_TIME - time.time()}")
        if time.time() >= END_TIME:
            break
        if current_custom_solution.score < best_score:
            best_solution = Solution(current_custom_solution.items)
            best_score = current_custom_solution.score
            print(f"{best_solution.nwalls} walls with a score of {best_score} with the solution {[iter[0] for iter in best_solution.items]}")
        else:
            # It cannot be greater so they are equal
            # Do a restart
            random.shuffle(ordered_art_ids)
            # print(f"Restarting with {ordered_art_ids}")
            current_custom_solution = place_pieces_in_order(instance, ordered_art_ids)

    print(f"Time taken is {time.time() - END_TIME + TIME_IN_SECONDS}")
    return best_solution

def place_pieces_in_order(instance: Instance, ordered_art_ids: List[int]) -> CustomSolution:
    solution = CustomSolution([], instance)
    wall_id = 0
    while not len(solution.items) == instance.n:
        # print("wall: ", wall_id)
        solution.walls.append(CustomWall(instance.wall.width(), instance.wall.height()))
        solution.items_by_wall[wall_id] = []
        solution.nwalls += 1
        for art_id in ordered_art_ids:
            solution = fit_art_on_wall(instance, solution, instance.artpieces[art_id - 1], wall_id)
        # print("Placed ", len(solution.items), " of ", instance.n)
        wall_id += 1
    solution.evaluate()
    return solution

def two_swap_local_search(original_solution: CustomSolution, instance: Instance, END_TIME: int) -> CustomSolution:
    permutations = list(combinations(range(instance.n), 2))
    random.shuffle(permutations)
    ordered_art_ids = [iter[0] for iter in original_solution.items]
    
    for left, right in permutations:
        # Exchanges two values before solving again
        left_value = ordered_art_ids[left]
        ordered_art_ids[left] = ordered_art_ids[right]
        ordered_art_ids[right] = left_value

        # Solve for a solution
        solution: CustomSolution = place_pieces_in_order(instance, ordered_art_ids)
        
        # We check if this solution passed the allocated time and if so we return the original one
        if time.time() >= END_TIME:
            return original_solution
        
        # Return the first solution which improves the current best solution
        if solution.score < original_solution.score:
            return solution
        
        # In the case we did not improve, revert the order before the next permutation
        ordered_art_ids[right] = ordered_art_ids[left]
        ordered_art_ids[left] = left_value
    
    # return the original solution if no improvements were made
    return original_solution

def swap_last_local_search(original_solution: CustomSolution, instance: Instance, END_TIME: int) -> CustomSolution:
    permutations = list(range(instance.n))
    random.shuffle(permutations)
    ordered_art_ids = [iter[0] for iter in original_solution.items]
    
    for index in permutations:
        # Reconstructs a list with the last element at a new position
        last = ordered_art_ids[-1]
        ordered_art_ids[-1] = ordered_art_ids[index]
        ordered_art_ids[index] = last

        # Solve for a solution
        solution: CustomSolution = place_pieces_in_order(instance, ordered_art_ids)
        
        # We check if this solution passed the allocated time and if so we return the original one
        if time.time() >= END_TIME:
            return original_solution
        
        # Return the first solution which improves the current best solution
        if solution.score < original_solution.score:
            return solution
        
        # In the case we did not improve, revert the order before the next permutation
        ordered_art_ids[index] = ordered_art_ids[-1]
        ordered_art_ids[-1] = last
    
    # return the original solution if no improvements were made
    return original_solution

def swap_last_tabu_search(best_solution: CustomSolution, instance: Instance, END_TIME: int) -> CustomSolution:
    INITIAL_PATIENCE = 3
    patience = INITIAL_PATIENCE
    permutations = list(range(instance.n))
    search: TabuSearch = TabuSearch()
    
    while patience > 0:
        # Find the best neighbor to switch with the last element
        ordered_art_ids = [iter[0] for iter in best_solution.items]
        last = ordered_art_ids[-1]
        
        # We reset this to always find the best neighbor regardless of the current solution
        search.best_score = sys.maxsize
        for index in permutations:
            if ordered_art_ids[index] in search.ignored_art_ids:
                continue
            
            # Swap the last element with the index position
            ordered_art_ids[-1] = ordered_art_ids[index]
            ordered_art_ids[index] = last
            
            # Solve for a solution
            solution: CustomSolution = place_pieces_in_order(instance, ordered_art_ids)
            
            # We check if this solution passed the allocated time and if so we return the original one
            if time.time() >= END_TIME:
                return best_solution
        
            if solution.score < search.best_score:
                search.best_score = solution.score
                search.best_neighbor = solution
            
            # Revert the order before the next permutation
            ordered_art_ids[index] = ordered_art_ids[-1]
            ordered_art_ids[-1] = last

        # Add the last index to avoid putting it at the end again
        search.ignored_art_ids.appendleft(last)
        # Remove the oldest neighbor
        if len(search.ignored_art_ids) > search.MEMORY:
            search.ignored_art_ids.pop()
        # Adjust the patience
        patience -= 1
        if search.best_score < best_solution.score:
            best_solution = search.best_neighbor
            patience = INITIAL_PATIENCE
            
    return best_solution

def fit_art_on_wall(instance: Instance, solution: CustomSolution, art_piece: ArtPiece, wall_id) -> CustomSolution:
    # Skip this painting if it has already been placed
    if art_piece.get_idx() in solution.placed_art_ids:
        return solution
    
    # Skip this painting if its area is greater than the remaining area
    wall: CustomWall = solution.walls[wall_id]
    if art_piece.width() * art_piece.height() > wall.available_tile_count:
        return solution

    wall_w = instance.wall.width()
    wall_h = instance.wall.height()
    for coord_sum in range(wall_w + wall_h + 1):
        # Constraints: above 0, where the sum is possible, painting doesn't exceed wall height
        start_x = max(0, coord_sum - wall_h + 1, coord_sum + art_piece.height() - wall_h)
        # Constraints: where the sum is possible, possible, painting doesn't exceed wall width
        end_x = min(coord_sum + 1, wall_w - art_piece.width() + 1)
        for x in range(start_x, end_x):
            y = coord_sum - x
            if solution.try_place(instance, art_piece.get_idx(), wall_id, x, y):
                return solution
    return solution
