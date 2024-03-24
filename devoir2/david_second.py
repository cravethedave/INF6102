from utils import Pizzeria, Route, Mine, Dict, Tuple, List, sqrt, Instance, Solution
import random
import sys
import time

MIN_DELIVERY = 10

#region Custom classes
class CustomPizzeria(Pizzeria):
    def __init__(self, pizzeria: Pizzeria):
        super().__init__(
            pizzeria.id,
            pizzeria.x,
            pizzeria.y,
            pizzeria.i,
            pizzeria.maxInventory,
            pizzeria.minInventory,
            pizzeria.dailyConsumption,
            pizzeria.inventoryCost
        )
        
    def min_delivery(self) -> int:
        return  round(self.L - (self.i - self.r), 2)

    def consume(self):
        self.i = round(self.i - self.dailyConsumption, 2)
        self.inventoryLevel = self.i

    def make_delivery(self, qty: float):
        self.i = round(self.i + qty, 2)
        self.inventoryLevel = self.i

class Delivery():
    def __init__(self, id: int, value: float) -> None:
        self.id = id
        self.v = value

class Truck():
    def __init__(self, deliveries: dict[int, float]) -> None:
        self.deliveries = deliveries
        self.occupied = round(sum(d for d in deliveries.values()), 2)
    
    def add_delivery(self, delivery: Delivery):
        self.deliveries[delivery.id] = delivery.v
        self.occupied = round(self.occupied + delivery.v, 2)
    
    def adjust_load(self, id: int, value: float):
        self.occupied = round(self.occupied - self.deliveries[id] + value, 2)
        self.deliveries[id] = value

class TimeStep():
    def __init__(self, trucks: list[Truck]) -> None:
        self.trucks = trucks
#endregion


#region Solution builder
def build_solution(
    instance: Instance,
    forced_deliveries: list[set[int]]
) -> list[list[list[tuple[int, float]]]]:
    solution: list[list[list[tuple[int, float]]]] = []
    pizzerias = {p.id:CustomPizzeria(p) for p in instance.pizzerias}
    
    for ts in range(instance.T):
        ordered_deliveries = get_deliveries_priority(pizzerias)

        # order according to least added cost and select best one that fits
        chosen_deliveries = select_deliveries(ordered_deliveries, forced_deliveries[ts], pizzerias, instance)

        # do smart loading and give as many cycles as you can to the furthest from mine
        loaded_trucks = load_trucks(chosen_deliveries, instance, pizzerias)

        for truck in loaded_trucks:
            for pid, value in truck.deliveries.items():
                pizzerias[pid].make_delivery(value)
                if pizzerias[pid].U < pizzerias[pid].i:
                    print(f"[WARN] Too much delivered to {pid} on timestep {ts}")

        for p in pizzerias.values():
            p.consume()
            if pizzerias[pid].L > pizzerias[pid].i:
                print(f"[WARN] Not enough delivered to {pid} on timestep {ts}")

        solution.append([[(id, v) for id, v in truck.deliveries.items()] for truck in loaded_trucks])
    
    # print_solution(instance, solution)
    
    return solution

def get_deliveries_priority(pizzerias: dict[int, CustomPizzeria]) -> list[Delivery]:
    return [Delivery(p.id, p.min_delivery()) for p in sorted(pizzerias.values(), key=lambda x: x.i/x.r)]

def select_deliveries(
    ordered_deliveries: list[Delivery],
    forced_deliveries: set[int],
    pizzerias: dict[int, CustomPizzeria],
    instance: Instance
) -> list[Truck]:
    trucks: list[Truck] = [Truck({}) for _ in range(instance.M)]
    # necessary deliveries
    selected: set[int] = set()
    for delivery in ordered_deliveries:
        if delivery.v < 0:
            continue
        for truck in trucks:
            if truck.occupied + delivery.v > instance.Q:
                continue
            selected.add(delivery.id)
            truck.add_delivery(delivery)
            break
    # Forced deliveries
    for id in forced_deliveries:
        # Already selected
        if id in selected:
            continue
        for truck in trucks:
            if truck.occupied + MIN_DELIVERY >= instance.Q:
                continue
            truck.add_delivery(Delivery(id, MIN_DELIVERY))
            break
    return trucks

def load_trucks(
    trucks: list[Truck],
    instance: Instance,
    pizzerias: dict[int, CustomPizzeria]
) -> list[Truck]:
    for truck in trucks:
        if len(truck.deliveries) == 0:
            continue
        # remove to respect upper pizzeria bound
        for p in (pizzerias[id] for id in truck.deliveries.keys()):
            if truck.deliveries[p.id] + p.i > p.U:
                truck.adjust_load(p.id, round(p.U - p.i, 2))
        
        empty_space = round(instance.Q - truck.occupied, 2)
        # It should not be possible to have a negative value at this point
        if empty_space == 0:
            continue
        elif empty_space < 0:
            print("[ERROR] How did this happen")
            continue
        
        non_full_pizzerias = sum(1 for id, v in truck.deliveries.items() if v != pizzerias[id].U)
        for id, current_delivery in truck.deliveries.items():
            space_in_pizzeria = round(pizzerias[id].U - pizzerias[id].i, 2)
            if current_delivery == space_in_pizzeria:
                continue
            balanced_delivery = round(current_delivery + empty_space / non_full_pizzerias, 2)
            if space_in_pizzeria < balanced_delivery:
                empty_space = round(empty_space + current_delivery - space_in_pizzeria, 2)
                truck.adjust_load(id, space_in_pizzeria)
            else:
                empty_space = round(empty_space + current_delivery - balanced_delivery, 2)
                truck.adjust_load(id, balanced_delivery)
            non_full_pizzerias -= 1
    
    return trucks 
#endregion


#region Local Search
class SolutionWrapper():
    def __init__(
        self,
        instance: Instance,
        forced_deliveries: list[set[int]],
    ) -> None:
        self.forced_deliveries = forced_deliveries
        self.raw = build_solution(instance, forced_deliveries)
        self.cost, self.valid = check_validity(instance, self.raw)

def local_search(instance: Instance, limit: int) -> list[list[list[tuple[int, float]]]]:
    end = time.time() + limit
    current_forced_deliveries: list[set[int]] = [set() for _ in range(instance.T)]
    best_cost = sys.maxsize

    while time.time() < end:        
        solution: SolutionWrapper = SolutionWrapper(instance, current_forced_deliveries)
        
        if solution.valid and solution.cost < best_cost:
            print(f"[SUCCESS] Good new start {current_forced_deliveries} at a cost of {solution.cost}")
            best_cost = solution.cost
            best_solution = solution

        found, solution = search_iteration(instance, current_forced_deliveries, best_cost)
        
        if time.time() > end:
            return best_solution.raw
        
        if found:
            best_cost = solution.cost
            best_solution = solution
            current_forced_deliveries = solution.forced_deliveries
        else:
            current_forced_deliveries = generate_rnd_solution(instance)

    # return the best
    return best_solution.raw

def search_iteration(
    instance: Instance,
    current_forced_deliveries: list[set[int]],
    best_cost: float
) -> SolutionWrapper:
    # generate neighborhood
    neighbours: list[list[set[int]]] = generate_neighbours(instance, current_forced_deliveries)
    
    # iterate and find next solution
    for neighbour in neighbours:
        solution: SolutionWrapper = SolutionWrapper(instance, neighbour)
        if not solution.valid:
            continue
        if solution.cost < best_cost:
            print(f"[SUCCESS] It worked with {neighbour} at a cost of {solution.cost}")
            # best_cost = cost
            # best_solution = solution
            return (True, solution)
    
    return (False, solution)

def generate_neighbours(instance: Instance, current: list[set[int]]) -> list[list[set[int]]]:
    neighbours: list[list[set[int]]] = []
    
    # remove something currently forced
    for day, day_ids in enumerate(current):
        for removed_id in day_ids:
            neighbours.append([d.copy() for d in current])
            neighbours[-1][day].remove(removed_id)
    
    # add a new value
    for day, day_ids in enumerate(current):
        # We only want to add elements that are not already there to avoid looking at the current solution
        potential_additions = [i for i in range(1,instance.npizzerias+1) if i not in day_ids]
        for id in potential_additions:
            neighbours.append([d.copy() for d in current])
            neighbours[-1][day].add(id)
    
    random.shuffle(neighbours)
    return neighbours

def generate_rnd_solution(instance: Instance) -> list[set[int]]:
    solution: list[set[int]] = [set() for _ in range(instance.T)]
    for day in range(instance.T):
        for id in range(1,instance.npizzerias+1):
            if random.randint(0,1) == 0:
                continue
            solution[day].add(id)
    return solution
            
def check_validity(instance: Instance, solution: list[list[list[tuple[int, float]]]]) -> (float, bool):
    return instance.solution_cost_and_validity(Solution(instance.npizzerias, solution))

def copy_solution(s: list[list[list[tuple[int, float]]]]) -> list[list[list[tuple[int, float]]]]:
    return [[[(id, v) for id, v in t] for t in ts] for ts in s]
#endregion


#region Misc
def should_visit():
    pass
    # if dist_to(truck_schedule.last_stop(), delivery.pizzeria) -
    #    dist_to(truck_schedule.last_stop(), instance.mine) <=
    #    dist_to(delivery.pizzeria, instance.mine):

def dist_to(a: CustomPizzeria, b: CustomPizzeria | Mine):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def print_solution(instance: Instance, solution: list[list[list[tuple[int, float]]]]):
    pizzerias = {p.id:CustomPizzeria(p) for p in instance.pizzerias}
    for ts in range(instance.T):
        print("[INFO] NEW CYCLE")        
        print("Pizzerias")
        for id, p in pizzerias.items():
            print(f"\tPizzeria {id} needs {p.min_delivery()}. Current state: {p.L} {p.i} {p.U}")
        
        deliveries_by_truck = [{id:v for id,v in truck} for truck in solution[ts]]
        
        print("Deliveries")
        for i, truck in enumerate(deliveries_by_truck):
            occupied = sum(truck.values())
            print(f"\tTruck #{i} is loaded at {occupied} of {instance.Q}")
            for k,v in truck.items():
                print(f"\t\tDelivering {v} to {k}")
        
        for truck in deliveries_by_truck:
            for pid, value in truck.items():
                pizzerias[pid].make_delivery(value)
        
        for p in pizzerias.values():
            p.consume()
#endregion


def solve(instance: Instance) -> Solution:
    """
    This function generates a solution where at each timestep
    the first truck goes through every pizzeria and delivers pizzeria.dailyConsumption

    Args:
        instance (Instance): a problem instance

    Returns:
        Solution: the generated solution
    """
    # Change the limit to time in seconds
    limit = 600
    return Solution(instance.npizzerias, local_search(instance, limit))
