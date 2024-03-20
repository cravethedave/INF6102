from utils import *


def solve(instance: Instance) -> Solution:
    """
    This function generates a solution where at each timestep
    the first truck goes through every pizzeria and delivers pizzeria.dailyConsumption

    Args:
        instance (Instance): a problem instance

    Returns:
        Solution: the generated solution, this might not be a valid solution as truck capacities might be exceeded
    """
    sol_raw = []
    for t in range(instance.T):
        timestep = []
        for truck_id in range(instance.M):
            truck_route = []
            if truck_id == 0:
                for i, pizzeria in instance.pizzeria_dict.items():
                    truck_route.append((i, pizzeria.dailyConsumption))
            timestep.append(truck_route)
        sol_raw.append(timestep)
    return Solution(instance.npizzerias, sol_raw)
