from utils import Pizzeria, Mine, Instance, Solution
from math import exp
import random
import sys
import time

MIN_DELIVERY = 10
HEAT_DECREASE = 0.95
INITIAL_PATIENCE = 10

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
#endregion


#region Solution builder
def build_solution(
    instance: Instance,
    forced_deliveries: list[set[int]]
) -> list[list[list[tuple[int, float]]]]:
    """Builds a solution based on forced deliveries"""
    solution: list[list[list[tuple[int, float]]]] = []
    pizzerias = {p.id:CustomPizzeria(p) for p in instance.pizzerias}
    
    for ts in range(instance.T):
        # Orders deliveries by urgency
        ordered_deliveries = get_deliveries_priority(pizzerias)

        # Allocates deliveries to their respective trucks while avoiding overloading
        chosen_deliveries = select_deliveries(ordered_deliveries, forced_deliveries[ts], pizzerias, instance)

        # Load balances the trucks to optimize the delivery
        loaded_trucks = load_trucks(chosen_deliveries, instance, pizzerias)
        
        # Remove deliveries that are at 0 after load balancing
        for truck in loaded_trucks:
            empty_deliveries = []
            for pid, value in truck.deliveries.items():
                if value == 0:
                    empty_deliveries.append(pid)
            for pid in empty_deliveries:
                truck.deliveries.pop(pid)
        
        # Make the deliveries to update the state of the problem
        for truck in loaded_trucks:
            for pid, value in truck.deliveries.items():
                pizzerias[pid].make_delivery(value)
                if pizzerias[pid].U < pizzerias[pid].i:
                    print(f"[WARN] Too much delivered to {pid} on timestep {ts}")

        # Have each pizzeria consume and adjust its internal state
        for p in pizzerias.values():
            p.consume()
            if p.L > p.i:
                print(f"[WARN] Not enough delivered to {pid} on timestep {ts}")

        solution.append([[(id, v) for id, v in truck.deliveries.items()] for truck in loaded_trucks])
        
    return solution

def get_deliveries_priority(pizzerias: dict[int, CustomPizzeria]) -> list[Delivery]:
    return [Delivery(p.id, p.min_delivery()) for p in sorted(pizzerias.values(), key=lambda x: (x.i - x.L)/x.r)]

def select_deliveries(
    ordered_deliveries: list[Delivery],
    forced_deliveries: set[int],
    pizzerias: dict[int, CustomPizzeria],
    instance: Instance
) -> list[Truck]:
    """Associates deliveries to their respective trucks"""
    trucks: list[Truck] = [Truck({}) for _ in range(instance.M)]
    
    # Adds the necessary deliveries
    selected: set[int] = set()
    for delivery in ordered_deliveries:
        if delivery.v <= 0:
            continue
        # Sort by cost added if assigned to each truck
        trucks_by_added_cost: list[Truck] = sorted(trucks,
            key = lambda t: added_cost_to_path(
                get_last_stop(pizzerias, t, instance),
                pizzerias[delivery.id],
                instance.mine
            )
        )
        # Assign the delivery to the first truck that fits it
        for truck in trucks_by_added_cost:
            if truck.occupied + delivery.v > instance.Q:
                continue
            selected.add(delivery.id)
            truck.add_delivery(delivery)
            break
        
    # Adds the forced deliveries
    for id in forced_deliveries:
        # Avoid adding already selected deliveries
        if id in selected:
            continue
        # Sort by cost added if assigned to each truck
        trucks_by_added_cost: list[Truck] = sorted(trucks,
            key = lambda t: added_cost_to_path(
                get_last_stop(pizzerias, t, instance),
                pizzerias[delivery.id],
                instance.mine
            )
        )
        # Assign the delivery to the first truck that fits it
        for truck in trucks_by_added_cost:
            # We force a minimum delivery to avoid deliveries of small amounts
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
    """
    Load balances trucks to avoid breaking constraints and to maximize the delivery size.
    Maximizing delivery size hopefully allows us to avoid doing a future trip too soon.
    """
    for truck in trucks:
        if len(truck.deliveries) == 0:
            continue
        
        # Remove voliume to respect the pizzeria's upper bound
        for p in (pizzerias[id] for id in truck.deliveries.keys()):
            if truck.deliveries[p.id] + p.i > p.U:
                truck.adjust_load(p.id, round(p.U - p.i, 2))
        
        empty_space = round(instance.Q - truck.occupied, 2)
        
        # It should not be possible to have a negative value at this point but we have a check
        if empty_space == 0:
            continue
        elif empty_space < 0:
            print("[ERROR] Trucks should not be past their capacity at this point")
            continue
        
        # Sort the pizzerias by distance to the mine
        ordered_destinations = sorted(
            (
                id for id in truck.deliveries.keys()
                if pizzerias[id].i + truck.deliveries[id] < pizzerias[id].U
            ),
            key=lambda id: dist_to(pizzerias[id], instance.mine),
            reverse=True
        )
        
        # Keep filling the pizzerias until we can no longer do so, we fill the furthest ones first
        # We only add one cycle per iteration to try and keep all pizerias equally full
        while len(ordered_destinations) != 0 and empty_space > 0:
            filled_up = []
            for id in ordered_destinations:
                if empty_space == 0:
                    break
                current_delivery = truck.deliveries[id]
                space_in_pizzeria = round(pizzerias[id].U - pizzerias[id].i, 2)
                
                # Should not happen since we remove full ones, but we have a check
                if current_delivery == space_in_pizzeria:
                    continue
                
                volume_to_fill = space_in_pizzeria - current_delivery
                # We add the smallest value between a cycle's consumption, 
                # the space left in the pizzeria and the space left in the truck
                adjusted_delivery = current_delivery + min(pizzerias[id].r, empty_space, volume_to_fill)
                
                if adjusted_delivery == space_in_pizzeria:
                    filled_up.append(id)
                
                # Adjust delivery and empty space in the truck
                empty_space = round(empty_space + current_delivery - adjusted_delivery, 2)
                truck.adjust_load(id, adjusted_delivery)
            # Remove any full pizzerias from the next iteration
            for i in filled_up:
                ordered_destinations.remove(i)
    
    return trucks 
#endregion


#region Local Search
class SolutionWrapper():
    """A class to hold information on our local search"""
    def __init__(
        self,
        instance: Instance,
        forced_deliveries: list[set[int]],
    ) -> None:
        self.forced_deliveries = forced_deliveries
        self.raw = build_solution(instance, forced_deliveries)
        cost, valid = instance.solution_cost_and_validity(Solution(instance.npizzerias, self.raw))
        self.cost: float = cost
        self.valid: bool = valid

def local_search(instance: Instance, limit: int) -> list[list[list[tuple[int, float]]]]:
    """Starts a local search on an instance and keeps searching until it runs out of time."""
    end = time.time() + limit
    current_forced_deliveries: list[set[int]] = [set() for _ in range(instance.T)]
    best_cost = sys.maxsize
    heat = 10
    patience = INITIAL_PATIENCE

    while time.time() < end:
        # Construct an initial solution from forced deliveries
        solution: SolutionWrapper = SolutionWrapper(instance, current_forced_deliveries)
        
        # We test the initial solution
        if solution.valid and solution.cost < best_cost:
            best_cost = solution.cost
            best_solution = solution

        # A solution is accepted if it is valid and better or valid and worse with a probability
        accepted, solution = search_iteration(instance, current_forced_deliveries, best_cost, heat)
        
        if time.time() > end:
            return best_solution.raw
        
        # We accepted a new solution
        if accepted:
            if solution.cost < best_cost:
                best_cost = solution.cost
                best_solution = solution
                patience = INITIAL_PATIENCE
            else:
                patience -= 1
            current_forced_deliveries = solution.forced_deliveries
        
        if not accepted or patience == 0:
            patience = INITIAL_PATIENCE
            current_forced_deliveries = generate_rnd_solution(instance)
        heat *= HEAT_DECREASE

    # return the best
    return best_solution.raw

def accept_solution(solution: SolutionWrapper, best_cost: float, heat: float) -> bool:
    """Calculates whether we should accept or not based on probability and cost"""
    return solution.cost != best_cost and (
        solution.cost < best_cost or
        random.random() < exp(-(solution.cost - best_cost) / heat)
    )

def search_iteration(
    instance: Instance,
    current_forced_deliveries: list[set[int]],
    best_cost: float,
    heat: float
) -> tuple[bool, SolutionWrapper]:
    """One iteration of a neighborhood search, we return the first accepted solution"""
    # Generate neighborhood
    neighbours: list[list[set[int]]] = generate_neighbours(instance, current_forced_deliveries)
    
    # Iterate and find next solution
    for neighbour in neighbours:
        solution: SolutionWrapper = SolutionWrapper(instance, neighbour)
        if not solution.valid:
            continue
        if solution.cost < best_cost or accept_solution(solution, best_cost, heat):
            return (True, solution)
    
    # No solution was accepted
    return (False, solution)

def generate_neighbours(instance: Instance, current: list[set[int]]) -> list[list[set[int]]]:
    """Generates all neighbours and shuffles them to avoid bias introduced by the order"""
    neighbours: list[list[set[int]]] = []
    
    # Remove something currently forced
    for day, day_ids in enumerate(current):
        for removed_id in day_ids:
            neighbours.append([d.copy() for d in current])
            neighbours[-1][day].remove(removed_id)
    
    # Add a new value
    for day, day_ids in enumerate(current):
        # We only want to add elements that are not already there to avoid inserting the current solution
        potential_additions = [i for i in range(1,instance.npizzerias+1) if i not in day_ids]
        for id in potential_additions:
            neighbours.append([d.copy() for d in current])
            neighbours[-1][day].add(id)
    
    random.shuffle(neighbours)
    return neighbours

def generate_rnd_solution(instance: Instance) -> list[set[int]]:
    """Generates a new random solution that we can use as a starting point"""
    solution: list[set[int]] = [set() for _ in range(instance.T)]
    for day in range(instance.T):
        for id in range(1,instance.npizzerias+1):
            # Each element has a 50% chance of being added
            if random.randint(0,1) == 0:
                continue
            solution[day].add(id)
    return solution
#endregion


#region Misc
def added_cost_to_path(last_stop: CustomPizzeria, destination: CustomPizzeria, mine: Mine) -> float:
    """Calculates the added cost of adding the destination to the current truck's path"""
    return (
        dist_to(last_stop, destination) + \
        dist_to(destination, mine) - \
        dist_to(last_stop, mine)
    )

def get_last_stop(pizzerias: dict[int, CustomPizzeria], truck: Truck, instance: Instance) -> CustomPizzeria | Mine:
    """Returns a truck's last stop or the mine if he has not left it"""
    if len(truck.deliveries) == 0:
        return instance.mine
    id = list(truck.deliveries.keys())[-1]
    return pizzerias[id]

def dist_to(a: CustomPizzeria, b: CustomPizzeria | Mine) -> float:
    """Returns the distance between two different points in the problem"""
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def print_solution(instance: Instance, solution: list[list[list[tuple[int, float]]]]):
    """Added for debugging, not necessary to the code"""
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


def solve(instance: Instance, limit: int) -> Solution:
    # Change this line to change the allocated time to the problem
    # While we were running it, we added a --time flag in main and passed it instead
    limit = 300
    return Solution(instance.npizzerias, local_search(instance, limit))
