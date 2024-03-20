import copy

from utils import Instance


class Schedule:
    def __init__(self, instance: Instance):
        self.instance = instance
        # needs : dict {id : nombre de tours avant que le four devienne obligatoire}
        self.needs: dict[int, float] = {p.id: p.inventoryLevel // p.dailyConsumption for p in instance.pizzerias}
        self.profitable: dict[int, float] = {p.id: (instance.mine.inventoryCost - p.inventoryCost) for p in
                                             instance.pizzerias}

    def update_need(self, delivery: dict[int, float]) -> None:
        self.needs = {p.id: (p.inventoryLevel + delivery[p.id]) // p.dailyConsumption for p in self.instance.pizzerias}

    def update_needs(self, deliveries: list[dict[int, float]]) -> None:
        self.needs = {p.id: (p.inventoryLevel + delivery[p.id]) // p.dailyConsumption for p in self.instance.pizzerias}

    def get_critical_ovens(self, t: int) -> list[int]:
        return [k for k in self.needs.keys() if self.needs[k] == t]

    def get_not_critical_ovens(self) -> dict[int, float]:
        return {k: v for k, v in self.needs.items() if v != 0}

    def sorted_profit(self) -> dict[int, float]:
        return dict(sorted(self.profitable.items(), key=lambda item: item[1]))

    def opt_sorted_profit(self) -> dict[int, float]:
        not_critical_ovens = self.get_not_critical_ovens()
        not_critical_ovens_profit = {k: self.profitable[k] for k in not_critical_ovens.keys()}
        return dict(sorted(not_critical_ovens_profit.items(), key=lambda item: item[1]))


def only_mandatory(instance: Instance, sched: Schedule) -> list[dict[int, float]]:
    only_mand = []
    for t in range(instance.T):
        only_mand.append({})
        for p in sched.get_critical_ovens(t):
            only_mand[t][p] = instance.pizzeria_dict[p].dailyConsumption

    return only_mand


def count_conflicts(deliveries: list[dict[int, float]], max_capacity) -> float:
    cost = 0.
    for d in deliveries:
        cost += sum([-qty for qty in d.values() if qty < 0])
        cost += max(0., sum(d.values()) - max_capacity)

    return cost


def get_neighbours(plan: list[dict[int, float]]) -> list[list[dict[int, float]]]:
    neighbours = []
    for t, deliveries in enumerate(plan):
        for u in range(len(plan)):
            if t != u:
                for p_id, q in deliveries.items():
                    plan_copy = copy.deepcopy(plan)
                    plan_copy[u][p_id] = q
                    del plan_copy[t][p_id]

                    neighbours.append(plan_copy)

    return neighbours


def better(plan: list[dict[int, float]], sched: Schedule) -> list[dict[int, float]]:
    for t, deliveries in enumerate(plan):
        for p_id, q in deliveries.items():
            if sched.profitable[p_id] != 0:
                if sched.profitable[p_id] > 0:
                    plan[t][p_id] = q + 1
                else:
                    plan[t][p_id] = q - 1

    return plan


def optimize(plan: list[dict[int, float]], max_capacity, sched) -> list[dict[int, float]]:
    previous = plan
    best = plan
    while count_conflicts(best, max_capacity) == 0:
        best = better(best, sched=sched)
        if count_conflicts(best, max_capacity) != 0:
            return previous
        else:
            previous = copy.deepcopy(best)


def ls_deliveries(instance: Instance, sched: Schedule) -> list[dict[int, float]]:
    max_capacity = min(instance.mine.dailyProduction, instance.Q)
    best = only_mandatory(instance, sched)
    continue_search = True

    while continue_search:
        continue_search = False
        if count_conflicts(best, max_capacity) == 0:
            sched.update_needs()
            best = optimize(best, max_capacity, sched)

        else:
            neighbours = get_neighbours(best)
            for n in neighbours:
                if count_conflicts(n, max_capacity) < count_conflicts(best, max_capacity):
                    best = n
                    continue_search = True
                    break
    return best


inst = Instance(
    "/Users/benjamindjian/Documents/DD PolyMTL/Trimestre Hiver 2024 (H24)/INF6102/Devoirs/Devoir2/code_etudiant/instances/super_easy.txt")
s = Schedule(instance=inst)
best_s = ls_deliveries(instance=inst, sched=s)
print(best_s)


# Table des bornes de qté autorisée dans chaque four (compris entre 0 (si le niveau est au max) et max-min (si le niveau est au min))

