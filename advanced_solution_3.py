import numpy as np

from utils import Instance


class Schedule:
    def __init__(self, instance: Instance, levels: None | np.ndarray = None):
        self.instance = instance
        self.lb = np.array([p.minInventory for p in instance.pizzerias])
        self.ub = np.array([p.maxInventory for p in instance.pizzerias])
        self.profitable = np.array([(instance.mine.inventoryCost - p.inventoryCost) for p in instance.pizzerias])

        if levels:
            self.levels = levels
        else:
            self.levels = np.array(
                [[p.inventoryLevel - i * p.dailyConsumption for p in instance.pizzerias] for i in
                 range(self.instance.T)])

    def deliver(self, t, delivery: np.ndarray):
        assert delivery.shape == (len(self.instance.pizzerias),)
        self.levels[t, :] += delivery

    def get_load_delivery(self, t) -> float:
        diff = sum(self.levels[t, :] - self.levels[t - 1, :])
        return diff + sum([p.dailyConsumption for p in self.instance.pizzerias])

    def count_exceed_individual_cap(self) -> float:
        cap = self.instance.Q

    def count_exceed_cap(self) -> float:
        cap = min(self.instance.mine.dailyProduction, self.instance.Q * self.instance.M)
        return sum([self.get_load_delivery(t) > cap for t in range(1, self.instance.T)])

    def count_exceed_futur_cap(self) -> float:
        count = 0.
        cap = min(self.instance.mine.dailyProduction, self.instance.Q * self.instance.M)
        for i in range(self.instance.T):
            min_load = np.sum([np.maximum(0, self.lb - self.levels[i, :])])
            if min_load > cap:
                count += min_load

        return count

    def count_exceed_bounds(self) -> float:
        count = 0
        for i in range(self.instance.T):
            count += np.count_nonzero(self.levels[i, :] > self.ub)
            count += np.count_nonzero(self.levels[i, :] < self.lb)
        return count

    def get_profits(self) -> float:
        profits = 0
        for i in range(self.instance.T):
            profits += np.sum(np.multiply(self.levels[i, :], self.profitable))
        return profits

    def find_best_delivery(self):
        continue_search = True
        while continue_search:
            continue_search = False
            neighbours = self.get_neighbours()
            for n in neighbours:
                if n.cost() < self.cost():
                    self.levels = n
                    continue_search = True
                    break
        return self.levels

    def cost(self) -> float:
        return 100 * self.count_exceed_cap() + 100 * self.count_exceed_bounds() - self.get_profits()

    def get_neighbours(self) -> list:
        neighbours = []
        n = Schedule(instance=self.instance, levels=self.levels)
        # Faire des livraisons, puis optimiser les quantités tant qu'il n'y a pas de conflits
        return neighbours


inst = Instance(
    "/Users/benjamindjian/Documents/DD PolyMTL/Trimestre Hiver 2024 (H24)/INF6102/Devoirs/Devoir2/code_etudiant/instances/instanceD.txt")
s = Schedule(instance=inst)
print(s.count_exceed_futur_cap())
