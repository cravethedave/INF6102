from utils import *

class CustomWall(Wall):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, w, h):
        super().__init__(w, h)

    
def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of tuples of the form (<artipiece_id>, <wall_id>, <x_pos>, <y_pos>)
    """
    raise Exception('This should be implemented')
