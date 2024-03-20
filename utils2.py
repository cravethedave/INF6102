from math import sqrt

from utils import Pizzeria, Route, Mine


class CustomPizzeria(Pizzeria):
    def __init__(self, id_, x_, y, startingLevel, maxInventory, minInventory, dailyConsumption, inventoryCost):
        super().__init__(id_, x_, y, startingLevel, maxInventory, minInventory, dailyConsumption, inventoryCost)

    def consume(self):
        self.inventoryLevel -= self.dailyConsumption

    def make_delivery(self, qty: float):
        self.inventoryLevel += qty


class CustomRoute(Route):
    def __init__(self, mine: Mine, route: list[tuple[int, float]], pizzeria_dict: dict):
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


def euclidean_distance(a: Pizzeria | CustomPizzeria | Mine, b: Pizzeria | CustomPizzeria | Mine) -> int:
    return int(sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2) + .5)


def consume_oven(pizzerias: list[CustomPizzeria]):
    for p in pizzerias:
        p.consume()


def deliver_route(pizzeria_dict: dict[int, CustomPizzeria], route: CustomRoute):
    for (p, q) in route.route_list:
        pizzeria = pizzeria_dict[p]
        pizzeria.make_delivery(q)


Plan = list[dict[CustomPizzeria, float]]
Delivery = dict[CustomPizzeria:float]
TimeStepDeliveries = list[Delivery]
SolutionDeliveries = list[TimeStepDeliveries]


def deliveries_exceeding_cap(deliveries: dict[Pizzeria:float], capacity: float) -> bool:
    return sum(deliveries.values()) > capacity
