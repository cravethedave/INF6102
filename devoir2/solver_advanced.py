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
        return  self.L - (self.i - self.r)

    def consume(self):
        self.inventoryLevel -= self.dailyConsumption
        self.i = self.inventoryLevel

    def make_delivery(self, qty: float):
        self.inventoryLevel += qty
        self.i = self.inventoryLevel

class Delivery():
    def __init__(self, pizzeria: CustomPizzeria, value: float):
        self.pizzeria: CustomPizzeria = pizzeria
        self.value: float = value

class TruckSchedule():
    def __init__(self, deliveries: 'list[Delivery]'):
        self.deliveries = deliveries
        self.volume = sum(iter.pizzeria.U - iter.pizzeria.i for iter in self.deliveries)
    
    def add_delivery(self, d: Delivery):
        self.deliveries.append(d)
        self.volume += d.pizzeria.U - d.pizzeria.i
    
    def adjust_delivery(self, index: int, value: float):
        old_value = self.deliveries[index].value
        self.deliveries[index].value = value
        self.volume = self.volume - old_value + value
    
    def last_stop(self) -> CustomPizzeria:
        return self.deliveries[-1].pizzeria

def solve(instance: Instance) -> Solution:
    """
    This function generates a solution where at each timestep
    the first truck goes through every pizzeria and delivers pizzeria.dailyConsumption

    Args:
        instance (Instance): a problem instance

    Returns:
        Solution: the generated solution
    """
    # [TimeStep], TimeStep = [Route], Route = [(id,amount)]
    solution: list[list[list[tuple[int, float]]]] = []
    pizzerias = {p.id:CustomPizzeria(p) for p in instance.pizzerias}
    
    # for p in pizzerias:
    #     print(p.i)
    
    for t in range(instance.T):
        print("[INFO] NEW CYCLE")
        deliveries_to_make = get_deliveries_to_make(pizzerias.values())
        
        print("Pizzerias")
        for p in pizzerias.values():
            print(f"\t{p.id}: {p.L} {p.i} {p.U}, consumes {p.r}")
        
        truck_schedules = partition_deliveries(deliveries_to_make, instance)
        for i, t in enumerate(truck_schedules):
            if t.volume == 0:
                continue
            print(f"Truck # {i} carrying {t.volume} of {instance.Q}:")
            for d in t.deliveries:
                print(f"\t{d.value} to {d.pizzeria.id}")
        
        solution.append([[(d.pizzeria.id, d.value) for d in truck.deliveries] for truck in truck_schedules])
    
        # Keep at the bottom of the timetable loop
        for p in pizzerias.values():
            p.consume()
        
        make_deliveries(truck_schedules)
    
    return Solution(instance.npizzerias, solution)

class DayPlan():
    def __init__(self) -> None:
        self.min_volume = 0
        self.pizzerias: 'set[int]' = set()
        
    def add_pizzerias(self, p: CustomPizzeria):
        self.pizzerias.add(p.id)
        self.min_volume += p.min_delivery()
        
    def rm_pizzerias(self, p: CustomPizzeria):
        self.pizzerias.remove(p.id)
        self.min_volume -= p.min_delivery()

def get_future_deliveries(pizzerias: 'dict[int,CustomPizzeria]', instance: Instance, days_left: int):
    future_planning = [DayPlan() for _ in range(days_left)]
    for p in pizzerias.values():
        index = int(p.i / p.r)
        if index < days_left:
            future_planning[index].add_pizzerias(p)
    
    trucks_capacity = instance.M * instance.Q
    disallowed_pairs: list[set[int]] = [set() for _ in range(days_left)]
    while any(d.min_volume > trucks_capacity for d in future_planning):
        random
        

def get_deliveries_to_make(pizzerias: 'list[CustomPizzeria]') -> 'list[Delivery]':
    deliveries = []
    for p in pizzerias:
        if p.i - p.r <= p.L:
            deliveries.append(Delivery(p, p.U - p.i))
    return deliveries

def partition_deliveries(deliveries_to_make: 'list[Delivery]', instance: Instance) -> 'list[TruckSchedule]':
    """Returns a list containing the list of pizzerias that a truck must visit.
    
    Returns:
        list[list[int]]: Trucks containing the pizzeria ids to visit
    """
    delivery_schedule: 'list[TruckSchedule]' = [TruckSchedule([]) for _ in range(instance.M)]
    if len(deliveries_to_make) == 0:
        return delivery_schedule
    deliveries_by_dist = sorted(
        deliveries_to_make,
        key=lambda x: dist_to(x.pizzeria, instance.mine),
        reverse=True
    )
    
    for truck_schedule in delivery_schedule:
        if len(deliveries_by_dist) == 0:
            break
        truck_schedule.add_delivery(deliveries_by_dist[0])
        undelivered: 'list[Delivery]' = []
        
        for delivery in deliveries_by_dist[1:]:
            if truck_schedule.volume >= instance.Q:
                break
        
            # Add to a truck's path if it is faster than doing it alone
            if dist_to(truck_schedule.last_stop(), delivery.pizzeria) -\
                dist_to(truck_schedule.last_stop(), instance.mine) <=\
                dist_to(delivery.pizzeria, instance.mine):
                truck_schedule.add_delivery(delivery)
            else:
                undelivered.append(delivery)
        
        deliveries_by_dist = undelivered
    
    # Places all deliveries at the nearest stop
    for d in deliveries_by_dist:
        sorted(delivery_schedule, key=lambda s: dist_to(d.pizzeria, s.last_stop()))[0].add_delivery(d)
        
    # Balances truck loads so they are respected and give as many cycles of provisions
    for truck in delivery_schedule:
        if truck.volume == 0:
            continue
        minimum_to_pass_cycle = [
            delivery.pizzeria.L - (delivery.pizzeria.i - delivery.pizzeria.r)
            for delivery in truck.deliveries
        ]
        summed_min_to_pass_cycle = sum(minimum_to_pass_cycle)
        extra_vol = instance.Q - summed_min_to_pass_cycle
        if extra_vol > 0:
            max_provided_cycles = extra_vol / sum(delivery.pizzeria.r for delivery in truck.deliveries)
        else:
            max_provided_cycles = 0
            ratio = instance.Q / summed_min_to_pass_cycle
            minimum_to_pass_cycle = [v * ratio for v in minimum_to_pass_cycle]
            print("[WARN] The minimum cannot be achieved, this cycle is invalid due to previous poor planning")
        
        for i, delivery in enumerate(truck.deliveries):
            truck.adjust_delivery(i, round(minimum_to_pass_cycle[i] + max_provided_cycles * delivery.pizzeria.r, 2))
    
    return delivery_schedule

def make_deliveries(schedule: 'list[TruckSchedule]'):
    for truck in schedule:
        for delivery in truck.deliveries:
            delivery.pizzeria.make_delivery(delivery.value)

def dist_to(a: CustomPizzeria, b: CustomPizzeria | Mine):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5