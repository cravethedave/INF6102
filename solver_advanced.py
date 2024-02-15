import time
from itertools import combinations
from typing import Iterator

from utils import *


class CustomSolution(Solution):
    def __init__(self, items: List[Tuple[int, int, int, int]], instance: Instance):
        super().__init__(items)
        self.available_tile_count = {}
        for item in items:
            art_id = item[0]
            wall_id = item[1]
            art_w = instance.artpieces_dict[art_id].width()
            art_h = instance.artpieces_dict[art_id].height()
            if wall_id not in self.available_tile_count:
                self.available_tile_count[wall_id] = instance.wall.height() * instance.wall.width()
            self.available_tile_count[wall_id] -= art_w * art_h

    def __len__(self):
        return len(self.items)

    def get_piece_in_solution(self, instance):
        pieces_id = [item[0] for item in self.items]
        piece_in_solution = [piece for piece in instance.artpieces if piece.get_idx() in pieces_id]
        return piece_in_solution

    def evaluate(self) -> float:
        """
        The lower the better the solution is
        """
        sol_quality = sum([self.available_tile_count[w_id] for w_id in range(self.nwalls - 1)])

        return sol_quality

    def add(self, item: tuple[int, int, int, int], instance: Instance):
        """
        Complexité O(1) - Complexité de len(dict) est constante selon internet
        """
        self.items.append(item)
        if item[1] not in self.items_by_wall:
            self.items_by_wall[item[1]] = []
            self.available_tile_count[item[1]] = instance.wall.height() * instance.wall.width()
        self.items_by_wall[item[1]].append(item)
        art_w = instance.artpieces_dict[item[0]].width()
        art_h = instance.artpieces_dict[item[0]].height()
        self.available_tile_count[item[1]] -= art_w * art_h
        self.nwalls = len(self.items_by_wall)

    def remove(self, item: tuple[int, int, int, int], instance: Instance):
        """
        Complexité O(1) - Complexité de len(dict) est constante selon internet
        """
        self.items.remove(item)
        self.items_by_wall[item[1]].remove(item)
        art_w = instance.artpieces_dict[item[0]].width()
        art_h = instance.artpieces_dict[item[0]].height()
        self.available_tile_count[item[1]] += art_w * art_h
        self.nwalls = len(self.items_by_wall)

    def get_empty_tiles(self):
        return sum([self.available_tile_count[wall_id] for wall_id in self.available_tile_count])


def piece_overlap(instance: Instance, solution: CustomSolution, item: Tuple[int, int, int, int]):
    """True if piece overlap, false otherwise
    Complexité O(k)
    """
    overlap = False
    piece_id = item[0]
    wall_id = item[1]
    x_pos_1, y_pos_1 = item[2], item[3]
    if wall_id not in solution.items_by_wall:
        return overlap

    width_1 = instance.artpieces_dict[piece_id].width()
    height_1 = instance.artpieces_dict[piece_id].height()

    for p_id, _, x_pos_2, y_pos_2 in solution.items_by_wall[wall_id]:
        if p_id == piece_id:
            continue
        width_2 = instance.artpieces_dict[p_id].width()
        height_2 = instance.artpieces_dict[p_id].height()

        overlap = not ((x_pos_1 + width_1 <= x_pos_2) or (x_pos_2 + width_2 <= x_pos_1) or (
                y_pos_1 + height_1 <= y_pos_2) or (y_pos_2 + height_2 <= y_pos_1))

        if overlap:
            return overlap

    return overlap


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
def grid_order_iterator(width: int,
                        height: int,
                        x_min: int = 0,
                        y_min: int = 0,
                        skipped_coord=None) -> Iterator[tuple[int, int]]:
    if skipped_coord is None:
        skipped_coord = []
    for k in range(2 * max(width, height)):
        for y in range(min(k, 2 * max(width, height)) + 1):
            if k - y > width or y > height:  # Ne pas dépasser des dimensions des murs
                continue
            if k - y < x_min or y < y_min:  # Ne pas être inférieur aux dimensions minimum
                continue
            for (x_skip_1, y_skip_1, x_skip_2,
                 y_skip_2) in skipped_coord:  # Ignorer les coordonnées qui doivent être ignorées
                if x_skip_1 <= k - y <= x_skip_2 or y_skip_1 <= y <= y_skip_2:
                    continue
            yield k - y, y


def place_order(instance: Instance, ordered_piece_id: list[int]) -> CustomSolution:
    solution = CustomSolution([], instance)
    len_solution = 0
    nbr_wall = 0
    wall_w, wall_h = instance.wall.width(), instance.wall.height()
    piece_to_place = {k: 0 for k in
                      ordered_piece_id}  # Dictionary with piece_id: bool (true if the piece is in solution)
    skipped_coord = []
    while len_solution < instance.n:
        coord_iter = grid_order_iterator(width=instance.wall.width(), height=instance.wall.height(),
                                         skipped_coord=skipped_coord)
        for coord in coord_iter:
            for piece_id in [p for p in piece_to_place.keys() if piece_to_place[p] == 0]:
                piece_item = (piece_id, nbr_wall, coord[0], coord[1])

                solution.add(piece_item, instance)

                art_w = instance.artpieces_dict[piece_item[0]].width()
                art_h = instance.artpieces_dict[piece_item[0]].height()
                if len(solution) > 0 and art_w * art_h > solution.available_tile_count[piece_item[1]]:
                    solution.remove(piece_item, instance)
                    continue

                # Si la pièce "ne rentre pas dans l'espace restant", on essaie la pièce suivante
                piece_exceeds_wall = (coord[0] + instance.artpieces_dict[piece_id].width() > wall_w) or (
                        coord[1] + instance.artpieces_dict[piece_id].height() > wall_h)
                if piece_exceeds_wall:
                    solution.remove(piece_item, instance)
                    continue

                is_piece_overlap = piece_overlap(instance, solution, piece_item)
                if is_piece_overlap:
                    solution.remove(piece_item, instance)
                    continue

                # Sinon la pièce "rentre" dans l'espace restant du mur, elle est ajoutée à la solution
                else:
                    len_solution += 1
                    piece_to_place[piece_id] = 1
                    skipped_coord += [(coord[0],
                                       coord[1],
                                       coord[0] + instance.artpieces_dict[piece_id].width(),
                                       coord[1] + instance.artpieces_dict[piece_id].height())]

                    break

        nbr_wall += 1
        skipped_coord = []

    return solution


def solve(instance: Instance, exec_time: float = 5 * 60.) -> CustomSolution:
    """Write your code here

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
    current_order = list(range(1, len(instance.artpieces) + 1))
    best_sol = place_order(instance=instance, ordered_piece_id=current_order)
    print(
        f"{best_sol.nwalls} walls with a score of {best_sol.evaluate()} with the solution {[iter_[0] for iter_ in best_sol.items]}")

    # Exploration des voisins
    while time.time() < time_out:
        for ord_ in sub_invert_gen(current_order):
            new_sol_found = False
            if time.time() >= time_out:  # Si on a dépassé le temps, on sort de la boucle
                break

            neighbour = place_order(instance=instance, ordered_piece_id=ord_)  # Construction du voisin

            if neighbour.evaluate() < best_sol.evaluate():
                # print('Nouvelle solution adoptée')
                best_sol = neighbour
                print(
                    f"{best_sol.nwalls} walls with a score of {best_sol.evaluate()} with the solution {[iter_[0] for iter_ in best_sol.items]}")
                new_sol_found = True
                break

        if not new_sol_found:  # Relance aléatoire
            random.shuffle(current_order)
            print('Relance')

    return best_sol
