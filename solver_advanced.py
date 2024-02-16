from utils import *

class CustomWall(Wall):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, w, h):
        super().__init__(w, h)

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
        self.space_matrix: dict[int : List[List[Tuple(int,int)]]] = {}
        self.available_tile_count: List[int] = []
        for item in items:
            art_id = item[0]
            wall_id = item[1]
            art_x = item[2]
            art_y = item[3]
            
            art_w = instance.artpieces[art_id - 1].width()
            art_h = instance.artpieces[art_id - 1].height()
            
            if not wall_id in self.space_matrix.keys():
                self.add_wall(instance.wall)
            self.available_tile_count[wall_id] -= art_w * art_h
                
            # Sets the block as being taken
            for x in range(art_x, art_x + art_w):
                for y in range(art_y, art_y + art_h):
                    self.space_matrix[wall_id][x][y] = (0,0)
            
            # TODO optimize this (Currently never used so ignore it)
            # Adjusts left of the block
            for y in range(art_y, art_y + art_h):
                for x in range(art_x - 1, -1, -1): # Goes backwards so we can skip some tiles
                    space_x, space_y = self.space_matrix[wall_id][x][y]
                    # we can skip the rest of this row if we find another painting
                    if space_x == 0:
                        break
                    self.space_matrix[wall_id][x][y] = (art_x - x, space_y)
            
            # Adjusts bottom of the block
            for x in range(art_x, art_x + art_w):
                for y in range(art_y - 1, -1, -1): # Goes backwards so we can skip some tiles
                    space_x, space_y = self.space_matrix[wall_id][x][y]
                    # we can skip the rest of this row if we find another painting
                    if space_x == 0:
                        break
                    self.space_matrix[wall_id][x][y] = (space_x, art_y - y)

    def add_wall(self, wall: Wall):
        new_wall_id = len(self.available_tile_count)
        self.available_tile_count.append(wall.width() * wall.height())
        self.space_matrix[new_wall_id] = [
            [(wall.width() - x, wall.height() - y) for y in range(wall.height())]
            for x in range(wall.width())
        ]

    def try_place(self, instance: Instance, art_id: int, wall_id: int, art_x: int, art_y: int) -> bool:
        space_x, space_y = self.space_matrix[wall_id][art_x][art_y]
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
        self.available_tile_count[wall_id] -= art_w * art_h
            
        # Sets the block as being taken
        for x in range(art_x, art_x + art_w):
            for y in range(art_y, art_y + art_h):
                self.space_matrix[wall_id][x][y] = (0,0)
        
        # Adjusts left of the block
        for y in range(art_y, art_y + art_h):
            for x in range(art_x - 1, -1, -1): # Goes backwards so we can skip some tiles
                space_x, space_y = self.space_matrix[wall_id][x][y]
                # we can skip the rest of this row if we find another painting
                if space_x == 0:
                    break
                self.space_matrix[wall_id][x][y] = (art_x - x, space_y)
        
        # Adjusts bottom of the block
        for x in range(art_x, art_x + art_w):
            for y in range(art_y - 1, -1, -1): # Goes backwards so we can skip some tiles
                space_x, space_y = self.space_matrix[wall_id][x][y]
                # we can skip the rest of this row if we find another painting
                if space_x == 0:
                    break
                self.space_matrix[wall_id][x][y] = (space_x, art_y - y)

def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of tuples of the form (<artipiece_id>, <wall_id>, <x_pos>, <y_pos>)
    """
    solution = CustomSolution([], instance)
    
    
    # optional, maybe add randomization here for restarts
    ordered_art_pieces = {
        id:
        max((instance.artpieces[id - 1].width()/instance.wall.width()),
        (instance.artpieces[id - 1].height()/instance.wall.height()))
        for id in range(1, instance.n + 1)
    }
    ordered_art_pieces = [iter[0] for iter in sorted(ordered_art_pieces.items(), key= lambda item: -item[1])]
    
    wall_id = 0
    while not len(solution.items) == instance.n:
        print("wall: ", wall_id)
        solution.add_wall(instance.wall)
        solution.items_by_wall[wall_id] = []
        solution.nwalls += 1
        for art_id in ordered_art_pieces:
            solution = fit_art_on_wall(instance, solution, instance.artpieces[art_id - 1], wall_id)
        print("Placed ", len(solution.items), " of ", instance.n)
        wall_id += 1
    
    return solution

def fit_art_on_wall(instance: Instance, solution: CustomSolution, art_piece: ArtPiece, wall_id) -> CustomSolution:
    # Skip this painting if it has already been placed
    if art_piece.get_idx() in solution.placed_art_ids:
        return solution
    
    # Skip this painting if its area is greater than the remaining area
    if art_piece.width() * art_piece.height() > solution.available_tile_count[wall_id]:
        return solution
        
    wall_w = instance.wall.width()
    wall_h = instance.wall.height()
    for coord_sum in range(wall_w + wall_h + 1):
        for x in range(max(0, coord_sum - wall_h + 1), min(coord_sum + 1, wall_w)):
            y = coord_sum - x
            
            # Skip if the shape isn't within the bounds
            if x + art_piece.width() > wall_w or y + art_piece.height() > wall_h:
                continue
            
            if solution.try_place(instance, art_piece.get_idx(), wall_id, x, y):
                return solution
    return solution
