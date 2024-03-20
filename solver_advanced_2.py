import copy

from segment_deliveries import segment_deliveries_ls
from shortest_route import shortest_route
from utils import *
from utils2 import Plan, TimeStepDeliveries, CustomRoute, CustomPizzeria, consume_oven, deliver_route


def cost_plan(plan: Plan,
              trucks_capacity: float,
              mine_capacity: float,
              pizzerias: list[Pizzeria]) -> float:
    """Pour un plan de livraison, compte le nombre de conflits avec les trois contraintes :
    1. Capacité des camions
    2. Capacité de la mine
    3. Bornes des fours

    Un plan de livraison est une liste et chaque élément représente un timestep.
    Un time step est un dict qui présente toutes les pizzerias à livrer avec les quantités associées"""

    cost = 0.
    capacity = min(trucks_capacity, mine_capacity)
    current_level = {p: [p.inventoryLevel] for p in pizzerias}
    for t, timestep in enumerate(plan):
        load = sum([qty for qty in timestep.values()])
        # Si une des capacités est dépassée, on l'ajoute au coût
        cost += 10 * max(0., load - capacity)
        # Si une des bornes des pizzas est dépassées, on l'ajoute aussi au coût
        for p in pizzerias:
            if p in plan[t]:
                current_level[p].append(current_level[p][t] - p.dailyConsumption + timestep[p])
                cost += max(0., current_level[p][t + 1] - p.maxInventory)
                cost += max(0., p.minInventory - current_level[p][t + 1])

    return cost


def cost_plan_instance(plan: Plan, instance: Instance):
    return cost_plan(plan,
                     trucks_capacity=instance.M * instance.Q,
                     mine_capacity=instance.mine.dailyProduction,
                     pizzerias=instance.pizzerias)


# Todo
def optimize_plan(instance: Instance, plan: Plan) -> Plan:
    """À partir d'un plan de livraison, détermine les meilleures quantités à livrer"""
    ...


def get_first_plan(instance: Instance) -> Plan:
    """Retourne un plan qui à chaque timestep associe à chaque pizzeria une part égale de charbon"""
    plan = []
    # On répartit équitablement la quantité de charbon entre les pizzerias
    for t in range(instance.T):
        plan.append({})
        for p in instance.pizzerias:
            custom_p = CustomPizzeria(id_=p.id, x_=p.x, y=p.y, startingLevel=p.inventoryLevel,
                                      maxInventory=p.maxInventory,
                                      minInventory=p.minInventory, dailyConsumption=p.dailyConsumption,
                                      inventoryCost=p.inventoryCost)
            plan[t][custom_p] = p.dailyConsumption

    return plan


def get_close_plan(plan: Plan, instance) -> list[Plan]:
    """Complexité O(T^3*n)"""
    neighbours = []
    for t, timestep in enumerate(plan):
        for u in range(len(plan)):
            if u != t:
                for p, q in timestep.items():
                    n = copy.deepcopy(plan)
                    to_move = [m for m in n[t].keys() if m.id == p.id][0]
                    n[u][to_move] = q
                    del n[t][to_move]
                    neighbours.append(optimize_plan(instance, n))

    return neighbours


def get_delivery_to_make(instance: Instance) -> TimeStepDeliveries:
    """Recherche locale sur les possibles plans de livraison """
    # Initialise solution en respectant les contraintes

    best: Plan = get_first_plan(instance)
    continue_search = True
    while continue_search:
        continue_search = False
        neighbours: list[Plan] = get_close_plan(best, instance)
        for n in neighbours:
            if cost_plan_instance(n, instance) < cost_plan_instance(best, instance):
                print(cost_plan_instance(n, instance), cost_plan_instance(best, instance))
                best = n
                continue_search = True
                break
    return best


def generate_solution_from_plan(plan: Plan, instance: Instance) -> Solution:
    solution: list[list[list[tuple[int, float]]]] = []
    mine = instance.mine
    pizzerias = instance.pizzerias
    pizzerias = [CustomPizzeria(id_=p.id, x_=p.x, y=p.y, startingLevel=p.inventoryLevel, maxInventory=p.maxInventory,
                                minInventory=p.minInventory, dailyConsumption=p.dailyConsumption,
                                inventoryCost=p.inventoryCost) for p in pizzerias]
    truck_capacity = instance.Q
    nbr_truck = instance.M
    nbr_pizzerias = instance.npizzerias

    for t, deliveries_to_make in enumerate(plan):
        timestep = []
        # On partitionne les pizzerias à livrer entre différents camions
        initial_deliveries = segment_deliveries_ls(deliveries_to_make, truck_capacity, mine=mine)
        # On fait une recherche locale pour trouver le meilleur chemin de chaque camion
        for delivery in initial_deliveries:
            initial_route = [(p.id, q) for p, q in delivery.items()]
            pizzeria_dict = {p.id: p for p in delivery.keys()}
            initial_route = CustomRoute(mine=mine, route=initial_route, pizzeria_dict=pizzeria_dict)
            shortest = shortest_route(initial_route)
            timestep.append(shortest.route_list)
            deliver_route(pizzeria_dict=pizzeria_dict, route=shortest)

        # If all the camions are not on the road, append the empty routes
        while len(timestep) < nbr_truck:
            timestep.append([])
        solution.append(timestep)
        consume_oven(pizzerias)

    return Solution(npizzerias=nbr_pizzerias, raw_solution=solution)


def solve(instance: Instance) -> Solution:
    # Planning schedule
    plan = get_delivery_to_make(instance)
    # Generate the best solution from the plan
    solution = generate_solution_from_plan(plan, instance)
    return solution
