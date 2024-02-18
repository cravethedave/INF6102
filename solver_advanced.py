import time
from itertools import combinations
from typing import Iterator

from utils import *


class CustomSolution(Solution):
    def __init__(self, items: List[Tuple[int, int, int, int]], instance: Instance):
        super().__init__(items)
        self.available_tile_count = {0: instance.wall.height() * instance.wall.width()}
        self.space_matrix = {
            0: [[(instance.wall.width() - x, instance.wall.height() - y) for y in range(instance.wall.height())] for
                x in range(instance.wall.width())]}
        for item in items:
            art_id = item[0]
            wall_id = item[1]
            art_x = item[2]
            art_y = item[3]
            art_w = instance.artpieces_dict[art_id].width()
            art_h = instance.artpieces_dict[art_id].height()
            self.available_tile_count[wall_id] -= art_w * art_h

            if wall_id not in self.available_tile_count:
                self.available_tile_count[wall_id] = instance.wall.height() * instance.wall.width()
                self.space_matrix[wall_id] = [
                    [(instance.wall.width() - x, instance.wall.height() - y) for y in range(instance.wall.height())] for
                    x in range(instance.wall.width())]

            # Sets the block as being taken
            for x in range(art_x, art_x + art_w):
                for y in range(art_y, art_y + art_h):
                    self.space_matrix[wall_id][x][y] = (0, 0)

    def __len__(self):
        return len(self.items)

    def evaluate(self) -> float:
        """
        The lower the better the solution is
        """
        n_walls = len(self.items_by_wall)
        sol_quality = sum([self.available_tile_count[w_id] * (n_walls - w_id) for w_id in range(n_walls - 1)])

        return sol_quality

    def add(self, item: tuple[int, int, int, int], instance: Instance):
        """
        Complexité O(1) - Complexité de len(dict) est constante selon internet
        """
        piece_id, wall_id, art_x, art_y = item
        self.items.append(item)
        if wall_id not in self.items_by_wall:
            self.items_by_wall[wall_id] = []
            self.available_tile_count[wall_id] = instance.wall.height() * instance.wall.width()

        self.items_by_wall[wall_id].append(item)
        art_w = instance.artpieces_dict[piece_id].width()
        art_h = instance.artpieces_dict[piece_id].height()
        self.available_tile_count[wall_id] -= art_w * art_h

        # Sets the block as being taken
        for x in range(art_x, art_x + art_w):
            for y in range(art_y, art_y + art_h):
                self.space_matrix[wall_id][x][y] = (0, 0)

        # Adjusts left of the block
        for y in range(art_y, art_y + art_h):
            for x in range(art_x - 1, -1, -1):  # Goes backwards so we can skip some tiles
                space_x, space_y = self.space_matrix[wall_id][x][y]
                # we can skip the rest of this row if we find another painting
                if space_x == 0:
                    break
                self.space_matrix[wall_id][x][y] = (art_x - x, space_y)

        # Adjusts bottom of the block
        for x in range(art_x, art_x + art_w):
            for y in range(art_y - 1, -1, -1):  # Goes backwards so we can skip some tiles
                space_x, space_y = self.space_matrix[wall_id][x][y]
                # we can skip the rest of this row if we find another painting
                if space_x == 0:
                    break
                self.space_matrix[wall_id][x][y] = (space_x, art_y - y)

    def get_empty_tiles(self):
        return sum([self.available_tile_count[wall_id] for wall_id in self.available_tile_count])


# Voisinage inversion d'une sous liste
def sub_invert_gen(liste: list[int]) -> Iterator[list[int]]:
    permutations = list(combinations(range(len(liste)), 2))
    random.shuffle(permutations)

    for i, j in permutations:
        yield liste[:i] + liste[i:j + 1][::-1] + liste[j + 1:]


# Voisinage two-swap
def two_swaps_gen(liste: list[int]) -> Iterator[list[int]]:
    list_swaps = list(combinations(range(len(liste)), 2))
    random.shuffle(list_swaps)  # On mélange les permutations possibles
    for i, j in list_swaps:
        yield liste[:i] + [liste[j]] + liste[i + 1:j] + [liste[i]] + liste[j + 1:]


# Todo : Prouver que ce grid order n'ignore aucune solution optimale, ie que le voisinage est connecté
def grid_order_iterator(width: int, height: int) -> Iterator[tuple[int, int]]:
    for k in range(2 * max(width, height)):
        for y in range(min(k, 2 * max(width, height)) + 1):
            if k - y >= width or y >= height:  # Ne pas dépasser des dimensions des murs
                continue
            yield k - y, y


def place_order(instance: Instance, ordered_piece_id: list[int]) -> CustomSolution:
    solution = CustomSolution([], instance)
    len_solution = 0
    nbr_wall = 0
    piece_to_place = {k: 0 for k in
                      ordered_piece_id}  # Dictionary with piece_id: bool (true if the piece is in solution)
    while len_solution < instance.n:
        coord_iter = grid_order_iterator(width=instance.wall.width(), height=instance.wall.height())
        for coord_x, coord_y in coord_iter:
            for piece_id in [p for p in piece_to_place.keys() if piece_to_place[p] == 0]:
                piece_item = (piece_id, nbr_wall, coord_x, coord_y)

                art_w = instance.artpieces_dict[piece_item[0]].width()
                art_h = instance.artpieces_dict[piece_item[0]].height()

                # Si la surface de la pièce est plus grande que l'espace restant sur le mur, on ne place pas la pièce
                if art_w * art_h > solution.available_tile_count[nbr_wall]:
                    continue

                # Si la pièce ne rentre dans le mur, on ne l'ajoute pas à la solution
                space_x, space_y = solution.space_matrix[nbr_wall][coord_x][coord_y]
                if space_x < art_w or space_y < art_h:
                    continue

                # Sinon la pièce "rentre" dans l'espace restant du mur, elle est ajoutée à la solution
                else:
                    solution.add(piece_item, instance)

                    len_solution += 1
                    piece_to_place[piece_id] = 1

                    break

        nbr_wall += 1
        solution.available_tile_count[nbr_wall] = instance.wall.height() * instance.wall.width()
        solution.space_matrix[nbr_wall] = [
            [(instance.wall.width() - x, instance.wall.height() - y) for y in range(instance.wall.height())] for x in
            range(instance.wall.width())]

    return solution


def solve(instance: Instance, exec_time: float = 5 * 60.) -> CustomSolution:
    """
    Args:
        instance (Instance): An Instance object containing all you need to solve the problem
        exec_time (float): approximate duration of the algo in seconds

    Returns:
        CustomSolution: A solution object initialized with
                  a list of tuples of the form (<artipiece_id>, <wall_id>, <x_pos>, <y_pos>)
    """

    time_out = time.time() + exec_time
    new_sol_found = False

    # Solution initiale
    current_order = [piece.get_idx() for piece in
                     sorted(instance.artpieces, key=lambda p: abs(p.width() - p.height()))]
    best_sol = place_order(instance=instance, ordered_piece_id=current_order)
    print(
        f"{len(best_sol.items_by_wall)} walls with a score of {best_sol.evaluate()} with the solution {[iter_[0] for iter_ in best_sol.items]}")

    # Exploration des voisins
    while time.time() < time_out:
        for ord_ in two_swaps_gen(current_order):
            new_sol_found = False
            if time.time() >= time_out:  # Si on a dépassé le temps, on sort de la boucle
                break

            neighbour = place_order(instance=instance, ordered_piece_id=ord_)  # Construction du voisin

            if neighbour.evaluate() < best_sol.evaluate():
                best_sol = neighbour
                print(
                    f"{len(best_sol.items_by_wall)} walls with a score of {best_sol.evaluate()} with the solution {[iter_[0] for iter_ in best_sol.items]}")
                new_sol_found = True
                break

        if not new_sol_found:  # Relance aléatoire
            random.shuffle(current_order)
            print(f'Relance with {current_order}')

    return best_sol
