import copy
import random
import time
from itertools import permutations
from math import sqrt

import numpy as np

from utils import Instance, Mine, Pizzeria, Solution

NBR_NEIGHBOURS = 100


def euclidean_distance(a: Mine | Pizzeria, b: Mine | Pizzeria) -> int:
    return int(sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2) + .5)


class TruckAssignation:
    def __init__(self,
                 instance: Instance,
                 to_deliver: list[tuple[int, int]],
                 assignation: None | list[list[tuple[int, int]]] = None):
        """to_deliver : list des tuples (pizzeria id to deliver, qty to deliver)"""
        self.to_deliver = to_deliver
        self.instance = instance
        self.truck_cap = self.instance.Q
        # Default assignation assigne toutes les livraisons au premier camion
        if not assignation:
            self.assignation = [to_deliver]
            for _ in range(self.instance.M - 1):
                self.assignation.append([])
        else:
            self.assignation = assignation

    def overload(self) -> int:
        """Count the number of coal in excess in each truck"""
        overload = 0.
        for truck in self.assignation:
            overload += min(0, sum([elem[1] for elem in truck]) - self.truck_cap)
        return overload

    def cost_assignation(self) -> int:
        cost = 0
        for truck in self.assignation:
            path = [self.instance.mine] + [self.instance.pizzeria_dict[elem[0] + 1] for elem in truck] + [
                self.instance.mine]
            for i in range(len(path) - 1):
                cost += euclidean_distance(path[i], path[i + 1])
        return cost

    def neighbours(self) -> list:
        # Les voisins sont les swaps entre camion, et les swaps des clients au sein d'un même camion
        neighbours = []
        # Swap entre clients pour un même camion
        for i, truck in enumerate(self.assignation):
            n_assignation = copy.deepcopy(self.assignation)
            for p in list(permutations(truck))[1:]:  # [1:] -> Avoid considering original permutation
                n_assignation[i] = p

            neighbours.append(TruckAssignation(instance=self.instance,
                                               to_deliver=self.to_deliver,
                                               assignation=n_assignation))

        # Insertion client dans un camion
        for i, truck in enumerate(self.assignation):
            for j, client in enumerate(truck):
                for k in range(len(self.assignation)):
                    if k != i:
                        n_assignation = copy.deepcopy(self.assignation)
                        del n_assignation[i][j]
                        n_assignation[k].append(client)

                        neighbours.append(TruckAssignation(instance=self.instance,
                                                           to_deliver=self.to_deliver,
                                                           assignation=n_assignation))

        return neighbours

    def assign_route(self):
        """Assign trucks based on what pizzeria to visit at one timestep"""
        continue_search = True
        while continue_search:
            continue_search = False
            neighbours = self.neighbours()
            for n in neighbours:
                if (self.overload() == 0) and (n.overload() == 0):
                    if n.cost_assignation() <= self.cost_assignation():
                        self.assignation = n.assignation
                        continue_search = True
                        break
                if (self.overload() > 0) and n.overload() <= self.overload():
                    self.assignation = n.assignation
                    continue_search = True
                    break


class Schedule:
    def __init__(self, instance: Instance):
        self.instance = instance
        # Le plan par défaut est de livrer des quantités aléatoires
        self._plan = np.random.randint(low=0, high=self.instance.Q / 10,
                                       size=(self.instance.T, len(self.instance.pizzerias)))
        # Table of profitability of each pizzeria
        self._profitable = np.array([(instance.mine.inventoryCost - p.inventoryCost) for p in instance.pizzerias])

        self.truck_assignations = []
        for t in range(self.instance.T):
            non_zero_delivery = np.nonzero(self._plan[t, :])[0]
            non_zero_qty = self._plan[t, :][non_zero_delivery]

            self.truck_assignations.append(TruckAssignation(instance=self.instance,
                                                            to_deliver=[(p, q) for p, q in
                                                                        zip(non_zero_delivery, non_zero_qty)]))

    def to_solution(self) -> Solution:
        raw_solution: list[list[list[tuple[int, float]]]] = []

        for timestep in self.truck_assignations:
            assignation_solution = timestep.assignation
            for i, truck in enumerate(timestep.assignation):
                assignation_solution[i] = [(p + 1, q) for p, q in timestep.assignation[i]]
            raw_solution.append(assignation_solution)
        return Solution(npizzerias=len(self.instance.pizzerias), raw_solution=raw_solution)

    def conflict_global_cap(self) -> list[tuple[int, int]]:
        """Search if global capacity (total capacity of trucks or mine dailyProduction) is exceeded at any timestep
        Return conflit : [(timestep, excess_quantity),...]"""
        conflicts = []
        global_cap = min(self.instance.mine.dailyProduction, self.instance.Q * self.instance.M)
        for t in range(self.instance.T):
            load_to_deliver = np.sum(self._plan[t, :])
            if load_to_deliver > global_cap:
                conflicts.append((t, load_to_deliver - global_cap))
        return conflicts

    def conflict_individual_cap(self) -> list[tuple[int, int, int]]:
        """Search if individual capacity (instance.Q) is exceeded at any timestep
        Return conflit : [(timestep,truck_id, excess_quantity),...]"""
        conflicts = []
        individual_cap = self.instance.Q
        for timestep, t_assign in enumerate(self.truck_assignations):
            for truck_id in range(self.instance.M):
                truck_path = t_assign.assignation[truck_id]
                truck_load = sum([elem[1] for elem in truck_path])
                if truck_load > individual_cap:
                    conflicts.append((timestep, truck_id, truck_load - individual_cap))
        return conflicts

    def conflict_bounds(self) -> list[tuple[int, int, int]]:
        """Search if pizzeria capacity is out of bounds at any timestep
        Returns : [(timestep, pizzeria_id, excess_quantity), ...]
        excess_quantity > 0 if excess
        excess_quantity < 0 if not enough coal"""
        conflicts = []
        for t in range(self.instance.T):
            for i, p in self.instance.pizzeria_dict.items():
                if t > 0:
                    already_delivered = sum([self._plan[previous_t, i - 1] for previous_t in range(t)])
                    already_consumed = t * p.dailyConsumption
                    level = p.inventoryLevel - already_consumed + already_delivered
                else:
                    level = p.inventoryLevel
                # We assume that pizzerias are sorted by id in instance.pizzerias
                level_after_delivery = level + self._plan[t, i - 1]
                level_after_consumption = level_after_delivery - p.dailyConsumption
                if level_after_delivery > p.maxInventory:
                    conflicts.append((t, i, level_after_delivery - p.maxInventory))
                elif level_after_consumption < p.minInventory:
                    conflicts.append((t, i, level_after_consumption - p.minInventory))
        return conflicts

    def count_conflicts(self) -> tuple[int, int, int]:
        return len(self.conflict_global_cap()), len(self.conflict_individual_cap()), len(self.conflict_bounds())

    def compare_stock(self, other) -> int:
        """Returns negative value if self is more profitable that other"""
        assert self.instance.pizzeria_dict == other.instance.pizzeria_dict
        assert self.instance.T == other.instance.T
        assert np.array_equal(self._profitable, other._profitable)

        comparison = 0.

        levels = np.array([[p.inventoryLevel - t * p.dailyConsumption + self._plan[t, i - 1] for i, p in
                            self.instance.pizzeria_dict.items()] for t in range(self.instance.T)])
        levels_other = np.array([[p.inventoryLevel - t * p.dailyConsumption + other._plan[t, i - 1] for i, p in
                                  self.instance.pizzeria_dict.items()] for t in range(self.instance.T)])

        for t in range(self.instance.T):
            comparison += np.sum(np.multiply(levels[t, :] - levels_other[t, :], self._profitable))

        return comparison

    def compare_route_cost(self, other) -> int:
        """Returns negative value if self is more profitable that other"""
        cost = 0
        for index, assignation in enumerate(self.truck_assignations):
            assignation.cost_assignation() - other.truck_assignations[index].cost_assignation()
        return cost

    def assign_all_route(self):
        """Assign trucks for all time-steps"""
        for assignation in self.truck_assignations:
            assignation.assign_route()

    def generate_neighbours(self):
        l_global = self.conflict_global_cap()
        l_indiv = self.conflict_individual_cap()
        l_bounds = self.conflict_bounds()
        # Si la capacité globale est excédée sur un time step,
        # on réduit toutes les quantités livrées sur ce timestep
        if len(l_global) > 0:
            variations = np.zeros(shape=self._plan.shape)
            random_t = np.random.randint(low=0, high=len(l_global), size=1)[0]
            high = l_global[random_t][1]
            if high == 1:
                random_decrease = -1
            else:
                random_decrease = -np.random.randint(low=1, high=high, size=1)[0]
            variations[random_t, :] = np.full(fill_value=random_decrease, shape=self._plan[random_t, :].shape)

        # Si la capacité individuelle d'un camion est excédée sur un time step, on réduit la quantité livrée
        # du camion, un petit peu partout sur son chemin
        elif len(l_indiv) > 0:
            variations = np.zeros(shape=self._plan.shape)
            random_excess = random.choice(l_indiv)
            timestep, truck_id, qty_excess = random_excess

            path = self.truck_assignations[timestep].assignation[truck_id]
            pizzeria_path = [elem[0] for elem in path]
            variations[timestep, pizzeria_path] = -qty_excess

        # Si les bornes sont excédée pour une pizzeria en un temps donné,
        # On modifie la quantité livrée
        elif len(l_bounds) > 0:
            variations = np.zeros(shape=self._plan.shape)
            for t, p_id, qty in l_bounds:
                previous_t = np.random.randint(low=0, high=t + 1, size=1)[0]
                variations[previous_t, p_id - 1] = -qty

        else:
            variations = np.zeros(shape=self._plan.shape)
            t = np.random.randint(low=0, high=self.instance.T, size=1)[0]
            p = np.random.randint(low=0, high=len(self.instance.pizzerias), size=1)[0]
            variations[t, p] = np.random.randint(low=-20, high=20, size=1)[0]

        new_plan = np.maximum(0, self._plan + variations)
        n = Schedule(instance=self.instance)
        n._plan = new_plan

        # We need to update assignation according to the plan
        n.truck_assignations = []
        for t in range(n.instance.T):
            non_zero_delivery = np.nonzero(n._plan[t, :])[0]
            non_zero_qty = n._plan[t, :][non_zero_delivery]

            n.truck_assignations.append(TruckAssignation(instance=n.instance,
                                                         to_deliver=[(p, q) for p, q in
                                                                     zip(non_zero_delivery,
                                                                         non_zero_qty)]))
        return n

    def neighbour_plan(self) -> list:
        neighbours = []
        for i in range(NBR_NEIGHBOURS):
            n = self.generate_neighbours()
            neighbours.append(n)
        return neighbours

    def search_no_conflict(self) -> None:
        print(self.count_conflicts())
        conflicts = True
        while conflicts:
            neighbours = self.neighbour_plan()
            for n in neighbours:
                if n.count_conflicts() < self.count_conflicts():
                    n.assign_all_route()
                    self._plan = n._plan
                    self.truck_assignations = n.truck_assignations
                    print(self.count_conflicts())
                    conflicts = bool(sum(self.count_conflicts()))
                    break


def solve(instance: Instance, time_limit):
    start = time.time()
    sched = Schedule(instance=instance)
    print("Minimizing conflicts")
    sched.search_no_conflict()
    print("Maximizing profits")
    while time.time() - start < time_limit:
        neighbours = sched.neighbour_plan()
        for n in neighbours:
            if n.count_conflicts() == 0:
                n.assign_all_route()
                if sched.compare_stock(n) + sched.compare_route_cost(n) > 0:
                    sched._plan = n._plan
                    sched.truck_assignations = n.truck_assignations
                    print("Found better solution")
                    break

    return sched.to_solution()


if __name__ == "__main__":
    inst = Instance(
        "/Users/benjamindjian/Documents/DD PolyMTL/Trimestre Hiver 2024 (H24)/INF6102/Devoirs/Devoir2/code_etudiant/instances/instanceE.txt")
    start_time = time.time()
    solution = solve(instance=inst, time_limit=10)
    solving_time = round((time.time() - start_time) / 60, 2)
    inst.visualize(solution)

    cost = inst.solution_cost_and_validity(solution)
    pizzeria_cost, pizzeria_cost_matrix, pizzeria_validity, pizzeria_validity_matrix = inst.solution_pizzeria_cost_and_validity(
        solution)
    mine_cost, mine_cost_matrix, mine_validity, mine_validity_matrix = inst.solution_mine_cost_and_validity(
        solution)
    route_cost, route_cost_matrix, route_validity, route_validity_matrix = inst.solution_route_cost_and_validity(
        solution)
    timestep_validity, _ = inst.solution_timestep_validity(solution)

    print("***********************************************************")
    print("[INFO] Solution obtained")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print(
        f"[INFO] Cost for Marco : {cost[0]}\n\t Loss due to pizzeria coal inventory : {pizzeria_cost}\n\t Luigi charged : {mine_cost}\n\t Trucks did cost : {route_cost}")
    print(
        f"[INFO] Sanity check passed : {cost[-1]}\n\t Truck capacity is respected: {route_validity}\n\t Luigi's coal inventory is never negative : {mine_validity}\n\t Pizzerias bounds are respected : {pizzeria_validity}\n\t Routes do not overlap : {timestep_validity}")
    print("***********************************************************")
