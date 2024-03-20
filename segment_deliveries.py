import copy

from utils import Mine
from utils2 import CustomPizzeria, euclidean_distance, deliveries_exceeding_cap


def segment_deliveries(deliveries: dict[CustomPizzeria:float], truck_cap) -> list[dict[CustomPizzeria:float]]:
    """Returns the list of deliveries. Each element of the list corresponds to a truck"""
    segment = [{}]
    truck_id = 0
    for p, q in deliveries.items():
        # "truck_cap-q" because the new cargo must not exceed truck cap
        if deliveries_exceeding_cap(segment[truck_id], capacity=truck_cap - q):
            segment.append({p: q})
            truck_id += 1
        else:
            segment[truck_id][p] = q
    return segment


def delivery_cost(delivery: list[dict[CustomPizzeria:float]], mine: Mine) -> float:
    cost = 0
    for truck in delivery:
        pizzeria_path = list(truck.keys())
        path = [mine] + pizzeria_path + [mine]
        for i in range(len(path) - 1):
            cost += euclidean_distance(path[i], path[i + 1])

    return cost


def get_neighbours(delivery: list[dict[CustomPizzeria:float]],
                   truck_cap: float) -> list[list[dict[CustomPizzeria:float]]]:
    neighbours = []
    for i, truck in enumerate(delivery):
        for j in range(len(delivery)):
            for p, q in truck.items():
                if j != i and (not deliveries_exceeding_cap(delivery[j], truck_cap - q)):
                    n = copy.deepcopy(delivery)
                    to_move = [m for m in n[i].keys() if m.id == p.id][0]
                    del n[i][to_move]
                    n[j][to_move] = q
                    neighbours.append(n)

    return neighbours


def segment_deliveries_ls(deliveries: dict[CustomPizzeria:float],
                          truck_cap: float,
                          mine: Mine) -> list[dict[CustomPizzeria:float]]:
    """Returns the list of deliveries. Each element of the list corresponds to a truck"""
    best = segment_deliveries(deliveries, truck_cap)
    continue_search = True
    while continue_search:
        continue_search = False
        neighbours = get_neighbours(best, truck_cap)
        for n in neighbours:
            if delivery_cost(n, mine) < delivery_cost(best, mine):
                best = n
                continue_search = True
                break
    return best
