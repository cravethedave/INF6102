import copy

from utils import *


class CustomPizzeria(Pizzeria):
    def __init__(self, id_, x_, y, startingLevel, maxInventory, minInventory, dailyConsumption, inventoryCost):
        super().__init__(id_, x_, y, startingLevel, maxInventory, minInventory, dailyConsumption, inventoryCost)

    def consume(self):
        self.inventoryLevel -= self.dailyConsumption

    def make_delivery(self, qty: float):
        self.inventoryLevel += qty


class CustomRoute(Route):
    def __init__(self, mine: Mine, route: List[Tuple[int, float]], pizzeria_dict: Dict):
        super().__init__(mine=mine, route=route, pizzeria_dict=pizzeria_dict)
        self.pizzeria_dict = pizzeria_dict
        self.route_list = route

    def get_route_n(self) -> list[Route]:
        """
        Returns: Neighbours of given route, by inserting a pizzeria first in the route
        """
        neighbours = []
        for i, pizzeria in enumerate(self.pizzeria_path):
            if i == 0:
                continue  # the original order is not kept
            new_order = [pizzeria] + self.pizzeria_path[:i] + self.pizzeria_path[(i + 1):]
            qty = [elem[1] for p in new_order for elem in self.route_list if elem[0] == p.id]
            new_route = [(p.id, qty[i]) for i, p in enumerate(new_order)]
            n = CustomRoute(mine=self.mine, route=new_route, pizzeria_dict=self.pizzeria_dict)
            n.total_goods = self.total_goods
            neighbours.append(n)

        return neighbours


def euclidean_distance(a: CustomPizzeria | Mine, b: CustomPizzeria | Mine) -> int:
    return int(sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2) + .5)


def consume_oven(pizzerias: list[CustomPizzeria]):
    for p in pizzerias:
        p.consume()


def deliver_route(to_deliver: list[tuple[int, float]], pizzerias):
    for p in to_deliver:
        i, q = p
        pizzeria = pizzerias[i - 1]
        pizzeria.make_delivery(q)


def depleted_pizzeria(p: CustomPizzeria) -> bool:
    """True if the pizzeria will not be able to continue working without delivery"""
    return (p.inventoryLevel - p.dailyConsumption) < p.minInventory


def time_when_depleted(p: CustomPizzeria) -> int:
    time = 0
    while (p.inventoryLevel - time * p.dailyConsumption) >= p.minInventory:
        time += 1
    return time


def is_oven_profitable(oven: CustomPizzeria, mine: Mine) -> bool:
    """True if it is more profitable to stock at the pizzeria rather than the mine"""
    return oven.inventoryCost < mine.inventoryCost


def oven_profit(oven: CustomPizzeria, mine: Mine) -> float:
    """Higher value means more profitability for stocking in pizzeria"""
    return mine.inventoryCost - oven.inventoryCost


def sort_profitable_ovens(ovens: list[CustomPizzeria], mine: Mine) -> list[CustomPizzeria]:
    """Sort ovens, most interesting stock oven first"""
    return sorted(ovens, key=lambda p: oven_profit(p, mine))


def compute_best_quantities(ovens: list[CustomPizzeria],
                            mine: Mine,
                            truck_capacity: float) -> dict[CustomPizzeria, float]:
    """Returns best quantities to deliver to pizzeria (not necessarily feasible)"""
    quantities = {}
    for p in ovens:
        if is_oven_profitable(oven=p, mine=mine):
            quantities[p] = min(p.maxInventory - p.inventoryLevel, truck_capacity)
        else:
            quantities[p] = min(max(0., p.minInventory - (p.inventoryLevel - p.dailyConsumption)), truck_capacity)
    return quantities


def oven_priority(p: CustomPizzeria, qty: float, mine: Mine) -> float:
    return oven_profit(p, mine) * qty / 10 * time_when_depleted(p)


# We want to minimize the number of deliveries (cost of distance),
# that's why we consider only delivering the max possible each time
def sort_profitable_delivery(ovens: list[CustomPizzeria],
                             mine: Mine,
                             truck_capacity: float) -> dict[CustomPizzeria:float]:
    """Sort ovens, most interesting delivery to make first"""
    qty = compute_best_quantities(ovens, mine, truck_capacity)
    return {p: qty[p] for p in sorted(ovens, key=lambda p: oven_priority(p, qty[p], mine)) if qty[p] != 0}


def get_mandatory_delivery(ovens: list[CustomPizzeria],
                           mine: Mine,
                           remain_time: int,
                           truck_capacity: float) -> dict[CustomPizzeria:float]:
    """Returns the list of all mandatory delivery"""
    if remain_time > 1:
        qty = compute_best_quantities(ovens, mine, truck_capacity)
        return {p: qty[p] for p in ovens if depleted_pizzeria(p)}
    else:
        return {}


def deliveries_exceeding_cap(deliveries: dict[CustomPizzeria:float], capacity: float) -> bool:
    return sum(deliveries.values()) > capacity


def get_prof_delivery_in_capacity(ovens: list[CustomPizzeria],
                                  mine: Mine,
                                  capacity: float,
                                  truck_capacity: float) -> dict[CustomPizzeria:float]:
    """Returns delivery, most interesting delivery to make first, with the limit of a given capacity"""
    prof_delivery = sort_profitable_delivery(ovens, mine, truck_capacity)
    qty_delivered = 0
    prof_delivery_in_capacity = {}
    for p, q in prof_delivery.items():
        if qty_delivered + q < capacity:
            prof_delivery_in_capacity[p] = q
            qty_delivered += q
    return prof_delivery_in_capacity


def get_delivery_to_make(ovens: list[CustomPizzeria],
                         mine: Mine,
                         capacity: float,
                         remain_time: int,
                         truck_capacity: float) -> dict[CustomPizzeria:float]:
    """Returns the delivery to make at this timestep"""
    delivery_to_make = get_mandatory_delivery(ovens, mine, remain_time, truck_capacity)
    if deliveries_exceeding_cap(delivery_to_make, capacity) or deliveries_exceeding_cap(delivery_to_make,
                                                                                        mine.inventoryLevel):
        raise ValueError(
            f"Cannot deliver all pizzerias that need coal ! Need to deliver {sum(delivery_to_make.values())} but trucks capacity is {capacity} and mine capacity is {mine.inventoryLevel}")
    for p, q in get_prof_delivery_in_capacity(ovens, mine, capacity, truck_capacity).items():
        delivery_to_make[p] = q
    return delivery_to_make


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
                    if len(n[i]) == 0:
                        del n[i]
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


def shortest_route(initial_route: CustomRoute) -> CustomRoute:
    """
    Hill climbing to find the best route for a truck knowing what pizzeria it visits
    Args:
        initial_route: Initial Route

    Returns: The result of hill climbing for a given path

    """
    shortest = initial_route
    continue_search = True
    while continue_search:
        continue_search = False
        neighbours = shortest.get_route_n()
        for n in neighbours:
            if n.distance_cost < shortest.distance_cost:
                shortest = n
                continue_search = True
                break
    return shortest


def solve(instance: Instance) -> Solution:
    """
    This function generates a solution where at each timestep
    the first truck goes through every pizzeria and delivers pizzeria.dailyConsumption

    Args:
        instance (Instance): a problem instance

    Returns:
        Solution: the generated solution
    """
    solution: list[list[list[tuple[int, float]]]] = []
    mine = instance.mine
    pizzerias = instance.pizzerias
    pizzerias = [CustomPizzeria(id_=p.id, x_=p.x, y=p.y, startingLevel=p.inventoryLevel, maxInventory=p.maxInventory,
                                minInventory=p.minInventory, dailyConsumption=p.dailyConsumption,
                                inventoryCost=p.inventoryCost) for p in pizzerias]
    truck_capacity = instance.Q
    nbr_truck = instance.M
    nbr_pizzerias = instance.npizzerias

    for t in range(instance.T):
        print(f"INFO : Timestep {t}")
        timestep = []

        # Pour chaque time step, on sait quelle pizzeria livrer et en quelle quantité, trié par importance
        deliveries_to_make = get_delivery_to_make(ovens=pizzerias, mine=mine,
                                                  capacity=truck_capacity * nbr_truck, remain_time=instance.T - t,
                                                  truck_capacity=truck_capacity)

        print(f"INFO : Partitioning deliveries {t}")
        # On partitionne les pizzerias à livrer entre différents camions
        initial_deliveries = segment_deliveries_ls(deliveries_to_make, truck_capacity, mine=mine)
        # On fait une recherche locale pour trouver le meilleur chemin de chaque camion
        print(f"INFO : Optimizing routes {t}")
        for delivery in initial_deliveries:
            initial_route = [(p.id, q) for p, q in delivery.items()]
            pizzeria_dict = {p.id: p for p in delivery.keys()}
            initial_route = CustomRoute(mine=mine, route=initial_route, pizzeria_dict=pizzeria_dict)
            shortest = shortest_route(initial_route)
            timestep.append(shortest.route_list)
            deliver_route(to_deliver=shortest.route_list, pizzerias=pizzerias)

        # If all the camions are not on the road, append the empty routes
        while len(timestep) < nbr_truck:
            timestep.append([])
        solution.append(timestep)
        consume_oven(pizzerias)

    return Solution(npizzerias=nbr_pizzerias, raw_solution=solution)
