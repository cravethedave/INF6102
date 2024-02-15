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
        self.available_tiles: dict[int : List[List[int]]] = {}
        self.available_tile_count: List[int] = []
        for item in items:
            art_id = item[0]
            wall_id = item[1]
            art_x = item[2]
            art_y = item[3]
            art_w = instance.artpieces[art_id - 1].width()
            art_h = instance.artpieces[art_id - 1].height()
            
            if not wall_id in self.available_tiles.keys():
                self.add_wall(instance.wall)
            self.available_tile_count[wall_id] -= art_w * art_h
            
            # print("Size of this wall:", len(self.available_tiles[wall_id]),len(self.available_tiles[wall_id][0]))
            # print("Pos of this painting:", art_x,art_y)
            # print("Size of this painting:", instance.artpieces[art_id - 1].width(),instance.artpieces[art_id - 1].height())
            for x in range(art_x, art_x + art_w):
                for y in range(art_y, art_y + art_h):
                    # print("current insert pos: ", x,y)
                    self.available_tiles[wall_id][x][y] = 1

    def add_wall(self, wall: Wall):
        new_wall_id = len(self.available_tile_count)
        self.available_tile_count.append(wall.width() * wall.height())
        self.available_tiles[new_wall_id] = [[0]*wall.height() for _ in range(wall.width())]

    # def add(self, art_id, wall_id, x, y):
    #     item = (art_id, wall_id, x, y)
    #     self.items.append(item)
    #     if wall_id not in self.items_by_wall:
    #         self.items_by_wall[wall_id] = []
    #     self.items_by_wall[wall_id].append(item)
    #     self.placed_art_ids.append(art_id)

def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of tuples of the form (<artipiece_id>, <wall_id>, <x_pos>, <y_pos>)
    """
    solution = CustomSolution([], instance)
    # print(solution)
    
    
    # optional, maybe add randomization here for restarts
    ordered_art_pieces = {
        id:
        max((instance.artpieces[id - 1].width()/instance.wall.width()),
        (instance.artpieces[id - 1].height()/instance.wall.height()))
        for id in range(1, instance.n + 1)
    }
    ordered_art_pieces = [iter[0] for iter in sorted(ordered_art_pieces.items(), key= lambda item: -item[1])]
    
    wall_id = -1
    while not len(solution.items) == instance.n:
        wall_id += 1
        print("wall: ", wall_id)
        solution.add_wall(instance.wall)
        for art_id in ordered_art_pieces:
            solution = fit_art_on_wall(instance, solution, instance.artpieces[art_id - 1], wall_id)
        print("total placed: ", len(solution.items))
    
    # print(solution)
    
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
        for x in range(min(coord_sum + 1, wall_w)):
            y = coord_sum - x
            
            # Skip if the shape isn't within the bounds
            if x + art_piece.width() > wall_w or y + art_piece.height() > wall_h:
                continue
            
            # Skip if taken in matrix
            if solution.available_tiles[wall_id][x][y] == 1:
                continue
            
            # TODO Change construction
            new_solution = CustomSolution(solution.items + [(art_piece.get_idx(), wall_id, x, y)], instance)
            # TODO Change matrix to use distances and not occupation
            if instance.is_valid_solution(new_solution):
                return new_solution
            # TODO Physically remove taken matrix tiles to fully skip them? Might be more work than gain
    return solution