from utils import Pizzeria, Route, Mine, Dict, Tuple, List, sqrt, Instance, Solution
import random

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

class TruckDestinations():
    def __init__(self) -> None:
        self.destinations: list[int] = []
        self.occupied: int = 0
    
    def add_destination(self, id: int, value: int):
        self.destinations.append(id)
        self.occupied = round(self.occupied + value, 2)

class TruckDeliveries():
    def __init__(self, destinations: list[int], pizzerias: dict[int,CustomPizzeria]) -> None:
        self.deliveries: dict[int,int] = {}
        self.occupied: int = 0
        for pid in destinations:
            self.deliveries[pid] = pizzerias[pid].min_delivery()
            self.occupied += self.deliveries[pid]
        self.occupied = round(self.occupied, 2)
    
    def adjust_load(self, id: int, value: int):
        self.occupied = round(self.occupied - self.deliveries[id] + value, 2)
        self.deliveries[id] = value

class Delivery():
    def __init__(self, id: int, value: float) -> None:
        self.id = id
        self.v = value

class Truck():
    def __init__(self, deliveries: list[Delivery]) -> None:
        self.deliveries = deliveries

class TimeStep():
    def __init__(self, trucks: list[Truck]) -> None:
        self.trucks = trucks

class SolutionWrapper():
    # Make funtion to switch to a new neighbour
    def __init__(self,  npizzerias, raw_solution:List[List[List[Tuple[int,float]]]]) -> None:
        self.n = npizzerias
        self.raw = raw_solution
        self.steps = [
            TimeStep([
                Truck(
                    [Delivery(id, v) for id, v in t]
                ) for t in ts
            ]) for ts in raw_solution
        ]
    
    def get_utils_solution(self) -> Solution:
        return Solution(self.n, self.raw)

def solve(instance: Instance) -> Solution:
    """
    This function generates a solution where at each timestep
    the first truck goes through every pizzeria and delivers pizzeria.dailyConsumption

    Args:
        instance (Instance): a problem instance

    Returns:
        Solution: the generated solution
    """
    raw_solution: list[list[list[tuple[int, float]]]] = []
    pizzerias = {p.id:CustomPizzeria(p) for p in instance.pizzerias}
    
    for _ in range(instance.T):
        print("[INFO] NEW CYCLE")
        ordered_deliveries = get_deliveries_priority(pizzerias)
        
        print("Pizzerias")
        for pid, v in ordered_deliveries:
            print(f"\tPizzeria {pid} needs {v}. Current state: {pizzerias[pid].L} {pizzerias[pid].i} {pizzerias[pid].U}")
        
        chosen_deliveries = select_deliveries(ordered_deliveries, instance)
    
        loaded_trucks = load_trucks(chosen_deliveries, instance, pizzerias)
        
        print("Deliveries")
        for i, truck in enumerate(loaded_trucks):
            print(f"\tTruck #{i} is loaded at {truck.occupied} of {instance.Q}")
            for k,v in truck.deliveries.items():
                print(f"\t\tDelivering {v} to {k}")
        
        for truck in loaded_trucks:
            for pid, value in truck.deliveries.items():
                pizzerias[pid].make_delivery(value)
        
        for p in pizzerias.values():
            p.consume()
            
        raw_solution.append([[(id, v) for id, v in truck.deliveries.items()] for truck in loaded_trucks])
    
    # solution: SolutionWrapper = SolutionWrapper(instance.npizzerias, raw_solution)
    
    return Solution(instance.npizzerias, local_search(raw_solution, instance))

def local_search(original: list[list[list[tuple[int, float]]]], instance: Instance) -> list[list[list[tuple[int, float]]]]:
    best_solution = original
    best_cost, _ = instance.solution_cost_and_validity(Solution(instance.npizzerias, best_solution))
    changed = copy_solution(original)
    # generate neighborhood
    
    # iterate and find next solution
    for ts, time_step in enumerate(best_solution):
        for t, truck in enumerate(time_step):
            to_remove = set()
            for d, (id, value) in enumerate(truck):
                changed[ts][t] = [delivery for delivery in truck if delivery[0] != id]
                cost, valid = check_validity(instance, changed)
                if valid and cost <= best_cost:
                    to_remove.add(id)

    # return the best
    return changed

def check_validity(instance: Instance, solution: list[list[list[tuple[int, float]]]]) -> (float, bool):
    return instance.solution_cost_and_validity(Solution(instance.npizzerias, solution))

def copy_solution(s: list[list[list[tuple[int, float]]]]) -> list[list[list[tuple[int, float]]]]:
    return [[[(id, v) for id, v in t] for t in ts] for ts in s]

def get_deliveries_priority(pizzerias: dict[int, CustomPizzeria]) -> list[tuple[int, int]]:
    return [(p.id, p.min_delivery()) for p in sorted(pizzerias.values(), key=lambda x: x.i/x.r)]

def select_deliveries(ordered_deliveries: list[tuple[int, int]], instance: Instance):
    trucks = [TruckDestinations() for _ in range(instance.M)]
    for pid, min_delivery in ordered_deliveries:
        if min_delivery < 0:
            continue
        for truck in trucks:
            if truck.occupied + min_delivery > instance.Q:
                continue
            truck.add_destination(pid, min_delivery)
            break
    return trucks

def load_trucks(
    chosen_deliveries: list[TruckDestinations],
    instance: Instance,
    pizzerias: dict[int, CustomPizzeria]
) -> list[TruckDeliveries]:
    loaded_trucks: list[TruckDeliveries] = [
        TruckDeliveries(chosen_deliveries[m].destinations, pizzerias)
        for m in range(instance.M)
    ]
    for truck in loaded_trucks:
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
    
    return loaded_trucks

def should_visit():
    pass
    # if dist_to(truck_schedule.last_stop(), delivery.pizzeria) -
    #    dist_to(truck_schedule.last_stop(), instance.mine) <=
    #    dist_to(delivery.pizzeria, instance.mine):

def dist_to(a: CustomPizzeria, b: CustomPizzeria | Mine):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5