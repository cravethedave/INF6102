import os
import imageio
import matplotlib.pyplot as plt
import copy
from typing import List, Tuple, Dict
from math import sqrt
make_universal = lambda x: os.sep.join(x.split('/'))

class Solution:
    @classmethod
    def from_file(cls,filepath:str,npizzerias:int)->'Solution':
        """Reads a solution from a file

        Args:
            filepath (str): the path
            npizzerias (int): the number of pizzerias

        Returns:
            Solution: the encapsulated solution
        """
        raw = []

        with open(filepath, 'r') as file:
            t,trucks = file.readline().split(' ')
            for _ in range(int(t)):
                solution = []
                for __ in range(int(trucks)):
                    truck_line = file.readline().strip()
                    truck_line = list(map(lambda x:x.strip(') '),truck_line.split('(')))
                    n_clients = int(truck_line[0])
                    clients = []
                    for i in range(1,n_clients+1):
                        a,b = truck_line[i].split(', ')
                        a,b = int(a),float(b)
                        clients.append((a,b))
                    solution.append(clients)
                raw.append(solution)
                if not file.readline().strip():  # Skip empty line after each section
                    file.readline()  # Skip additional empty line

        return Solution(npizzerias,raw)

    def __init__(self, npizzerias, raw_solution:List[List[List[Tuple[int,float]]]]):
        """
        Initialize the Solution object with raw solution data.

        Parameters:
            npizzerias (int): Number of pizzerias.
            raw_solution (List[List[List[Tuple[int,float]]]]): Raw solution data\n
                this must be a list of timesteps\n
                each timestep is itself a list of Routes\n
                each route is a list of couples (pizzeria_id, delivered_amount)\n
                e.g [[ [(id1,q1),(id2,q2)] , [(id1,q1),(id2,q2)] ], [ [(id1,q1),(id2,q2),(id3,q3)] , [(id1,q1),(id2,q2)] ]]\n
                DO NOT SPECIFY Luigi's mine

        Attributes:
            npizzerias (int): Number of pizzerias.
            raw (List[List[List[Tuple[int,float]]]]): Raw solution data.
            pizzeria_pov (Dict[int,Dict[int,float]]): Pizzeria point of view data.
                Keys represent pizzeria IDs, values are dictionaries with timestep as key
                and quantity delivered as value.
            mine_pov (Dict[int,int]): Mine point of view data.
                Keys represent timesteps, values represent quantity picked up.
            truck_pov (Dict[int,Dict[int,Dict[int,float]]]): Truck point of view data.
                First level keys represent truck IDs, second level keys represent timesteps,
                and third level keys represent pizzeria IDs with quantity delivered as value.
        """
        self.npizzerias = npizzerias
        self.raw = raw_solution

        self.pizzeria_pov: Dict[int, Dict[int, float]] = {}
        """ pizzeria_id:{timestep:quantity_delivered}"""

        self.mine_pov: Dict[int, int] = {}
        """timestep:quantity_pickedup"""

        self.truck_pov: Dict[int, Dict[int, float]] = {}
        """truck_id:{timestep:{pizzeria_id:quantity_delivered}}"""


        # Initialize dictionaries
        for i in range(1,self.npizzerias+1):
            self.pizzeria_pov[i] = self.pizzeria_pov.get(i, {})

        # Process raw solution data
        for t, timestep in enumerate(raw_solution):
            self.mine_pov[t] = self.mine_pov.get(t, 0)
            for i in range(1,self.npizzerias+1):
                self.pizzeria_pov[i][t] = self.pizzeria_pov[i].get(t, 0)

            for truck_id, truck in enumerate(timestep):
                self.truck_pov[truck_id] = self.truck_pov.get(truck_id, {})
                self.truck_pov[truck_id][t] = self.truck_pov[truck_id].get(t, {})

                for pizzeria_id, pizzeria_quantity in truck:
                    self.truck_pov[truck_id][t][pizzeria_id] = pizzeria_quantity
                    self.mine_pov[t] += pizzeria_quantity
                    self.pizzeria_pov[pizzeria_id][t] += pizzeria_quantity

class Pizzeria:
    def __init__(self, id, x, y, startingLevel, maxInventory, minInventory, dailyConsumption, inventoryCost):
        """
        Initialize a Pizzeria object with given attributes.

        Parameters:
            id (int): Pizzeria ID.
            x (float): X coordinate of the Pizzeria.
            y (float): Y coordinate of the Pizzeria.
            startingLevel (float): Starting inventory level.
            maxInventory (float): Maximum inventory level.
            minInventory (float): Minimum inventory level.
            dailyConsumption (float): Daily consumption rate.
            inventoryCost (float): Cost of maintaining inventory.

        Attributes:
            id (int): Pizzeria ID.
            x (float): X coordinate of the Pizzeria.
            y (float): Y coordinate of the Pizzeria.
            maxInventory (float): Maximum inventory level (also assigned to self.U).
            minInventory (float): Minimum inventory level (also assigned to self.L).
            inventoryLevel (float): Current inventory level (also assigned to self.i).
            dailyConsumption (float): Daily consumption rate (also assigned to self.r).
            inventoryCost (float): Cost of maintaining inventory (also assigned to self.ic).
        """
        self.id = int(id)
        self.x = x
        self.y = y
        self.maxInventory = self.U = maxInventory
        self.minInventory = self.L = minInventory
        self.inventoryLevel = self.i = startingLevel
        self.dailyConsumption = self.r = dailyConsumption
        self.inventoryCost = self.ic = inventoryCost

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, x={self.x}, y={self.y}, maxInventory={self.maxInventory}, minInventory={self.minInventory}, inventoryLevel={self.inventoryLevel}, dailyConsumption={self.dailyConsumption}, inventoryCost={self.inventoryCost})"

    def generate_cost_and_validity(self, solution: Solution) -> Tuple[float, List[float], bool, List[bool], ]:
        """
        Calculate cost and validity for the Pizzeria based on the solution.

        Parameters:
            solution (Solution): Solution object representing delivery plan.


        Args:
            solution (Solution): your solution

        Returns:
            Tuple[float, List[float], bool, List[bool], List[Tuple[float,float,float]]]:\n
                total cost,\n
                cost at each timestep,\n
                global validity,\n
                validity at each timestep,\n
                triples (before_livraison,after_livraison,after_consumption) for each timestep\n
        """
        level = self.inventoryLevel
        validity_for_each_timestep = []
        cost_for_each_timestep = []
        detailed_quantity_trace = []
        for t, quantity_delivered in solution.pizzeria_pov[self.id].items():
            detailed_quantity_trace.append((level, level + quantity_delivered, level + quantity_delivered - self.dailyConsumption))
            level += quantity_delivered - self.dailyConsumption
            validity_for_each_timestep.append(detailed_quantity_trace[-1][-1] >= self.L and detailed_quantity_trace[-1][-2] <= self.U)
            cost_for_each_timestep.append(max(level * self.inventoryCost, 0))
        
        return sum(cost_for_each_timestep), \
               cost_for_each_timestep, \
               sum(validity_for_each_timestep) == len(validity_for_each_timestep), \
               validity_for_each_timestep, \
               detailed_quantity_trace

class Mine:
    def __init__(self, id, x, y, startingLevel, dailyProduction, inventoryCost):
        """
        Initialize a Mine object with given attributes.

        Parameters:
            id (int): Mine ID.
            x (float): X coordinate of the mine.
            y (float): Y coordinate of the mine.
            startingLevel (float): Starting inventory level.
            dailyProduction (float): Daily production rate.
            inventoryCost (float): Cost of maintaining inventory.

        Attributes:
            id (int): Mine ID.
            x (float): X coordinate of the mine.
            y (float): Y coordinate of the mine.
            inventoryLevel (float): Current inventory level (also assigned to self.i).
            dailyProduction (float): Daily production rate (also assigned to self.r).
            inventoryCost (float): Cost of maintaining inventory (also assigned to self.ic).
        """
        self.id = int(id)
        self.x = x
        self.y = y
        self.inventoryLevel = self.i = startingLevel
        self.dailyProduction = self.r = dailyProduction
        self.inventoryCost = self.ic = inventoryCost

    def generate_cost_and_validity(self, solution: Solution) -> Tuple[float, List[float], bool, List[bool], ]:
        """
        Calculate cost and validity for the Pizzeria based on the solution.

        Parameters:
            solution (Solution): Solution object representing delivery plan.


        Args:
            solution (Solution): your solution

        Returns:
            Tuple[float, List[float], bool, List[bool], List[Tuple[float,float,float]]]:\n
                total cost,\n
                cost at each timestep,\n
                global validity,\n
                validity at each timestep,\n
                triples (before_livraison,after_livraison,after_consumption) for each timestep\n
        """
        level = self.inventoryLevel
        validity_for_each_timestep = []
        cost_for_each_timestep = []
        detailed_quantity_trace = []
        for t, quantity_pickedup in solution.mine_pov.items():
            detailed_quantity_trace.append((level, level + self.dailyProduction, level - quantity_pickedup))
            level += self.dailyProduction - quantity_pickedup
            validity_for_each_timestep.append(level >= 0)
            cost_for_each_timestep.append(max(level * self.inventoryCost, 0))
        return sum(cost_for_each_timestep), \
               cost_for_each_timestep, \
               sum(validity_for_each_timestep) == len(validity_for_each_timestep), \
               validity_for_each_timestep, \
               detailed_quantity_trace

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, x={self.x}, y={self.y}, startingLevel={self.inventoryLevel}, dailyProduction={self.dailyProduction}, inventoryCost={self.inventoryCost})"


class Route:
    def __init__(self,mine:Mine,route:List[Tuple[int,float]],pizzeria_dict:Dict):
        """
        Args:
            route (List[Tuple[int,float]]): list of pairs pizzeria_id,quantity delivered without the Mine (0)
            pizzeria_dict (Dict): a dictionary with encapsulated pizzeria instances pizzeria_id:Pizzeria()
        """
        self.mine = mine
        self.pizzeria_path = [pizzeria_dict[cl] for cl,_ in route]
        self.path = [mine]+self.pizzeria_path+[mine]
        # CHANGED HERE
        self.total_goods = round(sum([q for cl,q in route]), 2)
        self.temp = [q for cl,q in route]
        euclidean_distance = lambda a,b:int(sqrt((a.x-b.x)**2+(a.y-b.y)**2)+.5)
        self.distance_cost = 0
        for i in range(len(self.path)-1):
            self.distance_cost+=euclidean_distance(self.path[i],self.path[i+1])

class Instance:
    def __init__(self,filepath: str):
        self.filepath = make_universal(filepath)
        with open(self.filepath) as f:
            lines = list([[float(x.strip()) for x in x.split('\t')] for x in f.readlines()])
            profile,mine,pizzerias=lines[0],lines[1],lines[2:]
            self.npizzerias,self.T,self.Q,self.M = [int(x) for x in profile]
            self.npizzerias-=1
            self.mine = Mine(*mine)
            self.pizzerias = [Pizzeria(*x) for x in pizzerias]
            self.pizzeria_dict = {c.id:c for c in self.pizzerias}
        self.visualization_directory=f'visualization_{self.filepath.split(os.sep)[-1]}'


    def solution_length_is_valid(self,solution:Solution):
        """ Checks the solution's dimensions
        """
        if len(solution.raw)!=self.T:
            print(f"This doesn't look like a correct solution, you must specify {self.T} instead of {len(solution.raw)} timesteps\n")
            return False
        for t,timestep in enumerate(solution.raw):
            if len(timestep)!=self.M:
                print(f"This doesn't look like a correct solution, you must specify {self.M} trucks at timestep {t} instead of {len(timestep)} trucks\nYou can specify trucks with empty routes.")
                return False
        return True

    def solution_pizzeria_cost_and_validity(self,solution:Solution):
        assert self.solution_length_is_valid(solution)
        validity_for_each_timestep = [True]*len(solution.raw)
        cost_for_each_timestep = [0]*len(solution.raw)

        for pizzeria in self.pizzerias:
            c,c_dt,v,v_dt,_ = pizzeria.generate_cost_and_validity(solution)
            for i,valid in enumerate(v_dt):
                validity_for_each_timestep[i]&=valid
            for i,cost in enumerate(c_dt):
                cost_for_each_timestep[i]+=cost

        return  sum(cost_for_each_timestep),\
                cost_for_each_timestep,\
                sum(validity_for_each_timestep)==len(validity_for_each_timestep),\
                validity_for_each_timestep

    def solution_route_cost_and_validity(self, solution: Solution) -> Tuple[float, List[List[float]], bool, List[List[bool]]]:
        """
        Calculate the cost and validity of each route in the given solution.

        Args:
            solution (Solution): The solution to evaluate.

        Returns:
            Tuple[float, List[List[float]], bool, List[List[bool]]]: A tuple containing:
                - The total cost of all routes.
                - The cost for each route at each timestep.
                - A boolean indicating if all routes in the solution are valid.
                - The validity of each route at each timestep.
        """
        assert self.solution_length_is_valid(solution)
        cost_for_each_timestep_route = [[0]*len(solution.raw[0]) for _ in range(len(solution.raw))]
        validity_for_each_timestep_route = [[True]*len(solution.raw[0]) for _ in range(len(solution.raw))]
        for t,routes in enumerate(solution.raw):
            for truck_id,r in enumerate(routes):
                route = Route(self.mine,r,self.pizzeria_dict)
                cost_for_each_timestep_route[t][truck_id]+=route.distance_cost
                validity_for_each_timestep_route[t][truck_id]&=route.total_goods<=self.Q
                # CHANGED HERE
                # if route.total_goods > self.Q:
                #     print(f"[ERROR] Total is {route.total_goods} for values: ")
                #     for x in route.temp:
                #         print(f"\t{x}")
        return  sum([sum(x) for x in cost_for_each_timestep_route]),\
                cost_for_each_timestep_route,\
                sum([sum(x) for x in validity_for_each_timestep_route])==len(validity_for_each_timestep_route)*len(validity_for_each_timestep_route[0]),\
                validity_for_each_timestep_route

    def solution_mine_cost_and_validity(self, solution: Solution) -> Tuple[float, List[float], bool, List[bool]]:
        """
        Calculate the cost and validity of the mine in the given solution.

        Args:
            solution (Solution): The solution to evaluate.

        Returns:
            Tuple[float, List[float], bool, List[bool]]: A tuple containing:
                - The total cost of the mine.
                - The cost for each timestep.
                - A boolean indicating if the mine is valid for all timesteps.
                - The validity of the mine at each timestep.
        """
        assert self.solution_length_is_valid(solution)
        validity_for_each_timestep = [True]*len(solution.raw)
        cost_for_each_timestep = [0]*len(solution.raw)

        c,c_dt,v,v_dt,_ = self.mine.generate_cost_and_validity(solution)
        for i,valid in enumerate(v_dt):
            validity_for_each_timestep[i]&=valid
        for i,cost in enumerate(c_dt):
            cost_for_each_timestep[i]+=cost

        return  sum(cost_for_each_timestep),\
                cost_for_each_timestep,\
                sum(validity_for_each_timestep)==len(validity_for_each_timestep),\
                validity_for_each_timestep

    def solution_timestep_validity(self,solution:Solution)->Tuple[bool,List[bool]]:
        """Checks that routes do not overlap

        Args:
            solution (Solution): your solution
        Returns:
            Tuple[bool,List[bool]]:\n
                True if routes do not overlap,
                A boolean for True each timestep where routes do not overlap

        """
        assert self.solution_length_is_valid(solution)
        validity_for_each_timestep = [True]*len(solution.raw)
        for t,timestep in enumerate(solution.raw):
            bag = set()
            for route in timestep:
                for c,_ in route:
                    validity_for_each_timestep[t]&=not(c in bag)
                    if not(c in bag):
                        bag.add(c)
        return sum(validity_for_each_timestep)==self.T,validity_for_each_timestep


    def solution_cost_and_validity(self,solution:Solution)->Tuple[float,bool]:
        """Checks solution cost and validity

        Args:
            solution (Solution): your solution

        Returns:
            Tuple[float,bool]: cost,validity
        """
        pizzeria_cost,_,pizzeria_validity,_ =  self.solution_pizzeria_cost_and_validity(solution)
        mine_cost,_,mine_validity,_ = self.solution_mine_cost_and_validity(solution)
        route_cost,_,route_validity,_ = self.solution_route_cost_and_validity(solution)
        timestep_validity, _ = self.solution_timestep_validity(solution)

        return pizzeria_cost+mine_cost+route_cost, (pizzeria_validity and route_validity and mine_validity and timestep_validity)

    def visualize_inventory_sets(self,sol:Solution,show=True):
        """Generates the inventory display

        Args:
            sol (Solution): your solution
            show (bool, optional): If it should be displayed in a window. Defaults to True.
        """
        costandv=[c.generate_cost_and_validity(sol) for c in self.pizzerias]
        triples_sets = [c[-1] for c in costandv]
        num_sets = len(triples_sets)
        num_bars = 3  # inventory_before_delivery, inventory_after_delivery, inventory_after_consumption
        bar_width = 0.25
        index = list(range(len(triples_sets[0])))

        fig, axes = plt.subplots(nrows=num_sets, ncols=1, figsize=(15, num_sets*3), sharey=True)

        for i, c in enumerate(zip(triples_sets,self.pizzerias)):
            triples,pizzeria = c
            ax = axes[i]
            for j in range(num_bars):
                ax.bar([z + j * bar_width for z in index], [triple[j] for triple in triples], bar_width, label=['Avant la livraison','Après la livraison','Après la consommation'][j])

            ax.axhline(y=pizzeria.L, color='r', linestyle='--', label=f'Capacité minimale ({pizzeria.L})')
            ax.axhline(y=pizzeria.U, color='g', linestyle='--', label=f'Capacité maximale ({pizzeria.U})')

            ax.set_ylabel('Inventaire')
            ax.set_title(f'Inventaire - Pizzeria {i+1} - $c_{i+1}={pizzeria.ic}$ - $r_{i+1}={pizzeria.r}$',color='black' if costandv[i][-3] else 'red')
            ax.set_xticks([z + bar_width for z in index])
            ax.set_xticklabels([f'T={k}' for k in range(len(triples))])
            ax.legend()
            ax.grid()
        plt.tight_layout()
        plt.savefig(f'pizzerias_inventory_{self.visualization_directory}.png')
        if show:
            plt.show()
        plt.clf()
        plt.close()

    def visualize_mine(self,sol:Solution,show=True):
        costandv=[c.generate_cost_and_validity(sol) for c in [self.mine]]
        triples_sets = [c[-1] for c in costandv]
        num_sets = len(triples_sets)
        num_bars = 3  # inventory_before_delivery, inventory_after_delivery, inventory_after_consumption
        bar_width = 0.25
        index = list(range(len(triples_sets[0])))

        fig, axes = plt.subplots(nrows=num_sets, ncols=1, figsize=(15, num_sets*3), sharey=True)

        for i, c in enumerate(zip(triples_sets,[self.mine])):
            triples,pizzeria = c
            for j in range(num_bars):
                plt.bar([z + j * bar_width for z in index], [triple[j] for triple in triples], bar_width, label=['Avant la livraison','Après la livraison','Après la consommation'][j])

            plt.ylabel('Inventaire')
            plt.title(f'Inventaire - Mine {i+1}',color='black' if costandv[i][-3] else 'red')
            plt.xticks([z + bar_width for z in index],labels=[f'T={k}' for k in range(len(triples))])
            plt.legend()
            plt.grid()
        plt.tight_layout()
        plt.savefig(f'mine_inventory_{self.visualization_directory}.png')
        if show:
            plt.show()
        plt.clf()
        plt.close()

    def generate_setup_visualization(self,sol:Solution,mine_quantity:float,customer_quantities:Dict[int,float],title='',image_id:int=0):
        customers = self.pizzeria_dict
        mines = [self.mine]
        plt.figure(figsize=(10, 6))
        # Plot mines
        for mine in mines:
            plt.plot(mine.x, mine.y, 'bs', markersize=10, label=f'Mine {mine.id}')
            t = plt.text(mine.x-5, mine.y-15, mine_quantity, verticalalignment='bottom', horizontalalignment='left', fontsize=10, color='yellow', fontweight='bold')
            t.set_bbox(dict(facecolor='blue', alpha=0.5, edgecolor='blue'))

        # Plot customers
        for id,customer in customers.items():
            plt.plot(customer.x, customer.y, 'ro', markersize=8,)
            plt.text(customer.x-5, customer.y, str(customer.id), verticalalignment='bottom', horizontalalignment='right', fontsize=16)
            q = customer_quantities[customer.id]
            t = plt.text(customer.x-5, customer.y-20, str(customer.L)+"$\leq$"+str(q)+"$\leq$"+str(customer.U), verticalalignment='center', horizontalalignment='left', fontsize=10, color='yellow' if q/customer.U>.3 else 'black', fontweight='bold')
            t.set_bbox(dict(facecolor='green' if customer.L<=q and q<=customer.U else 'red', alpha=q/customer.U if customer.L<=q and q<=customer.U else 1, edgecolor='green' if customer.L<=q and q<=customer.U else 'red'))
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title(title)
        plt.grid()
        plt.legend()
        os.makedirs(self.visualization_directory, exist_ok=True)
        plt.savefig(os.path.join(self.visualization_directory,f'plot_{image_id}.png'))
        plt.clf()
        plt.close()
        return None

    def visualize_route(self,truck_id:int,route:Route,mine_quantity:float,customer_quantities:Dict[int,float],title="",image_id:int=0):
        customers = self.pizzeria_dict
        mine = self.mine
        route_as_dict = {k:v for k,v in route}
        i=truck_id
        new_customer_quantities = {}

        new_depq = mine_quantity-sum([q for _,q in route])
        # Plot routes
        colors = plt.cm.tab10.colors

        plt.figure(figsize=(10, 6))

        plt.plot(mine.x, mine.y, 'bs', markersize=10, label=f'Mine {mine.id}')
        t = plt.text(mine.x-5, mine.y-15, new_depq, verticalalignment='bottom', horizontalalignment='left', fontsize=10, color='yellow', fontweight='bold')
        t.set_bbox(dict(facecolor='blue', alpha=0.5, edgecolor='blue'))

        # Plot customers
        for id,customer in customers.items():
            plt.plot(customer.x, customer.y, 'ro', markersize=8,)
            plt.text(customer.x-5, customer.y, str(customer.id), verticalalignment='bottom', horizontalalignment='left', fontsize=16)
            q = customer_quantities[customer.id]+route_as_dict.get(customer.id,0)
            new_customer_quantities[customer.id]=q
            t = plt.text(customer.x-5, customer.y-20, str(customer.L)+"$\leq$"+str(q)+"$\leq$"+str(customer.U), verticalalignment='bottom', horizontalalignment='left', fontsize=10, color='yellow' if q/customer.U>.3 else 'black', fontweight='bold')
            t.set_bbox(dict(facecolor='green', alpha=q/customer.U if customer.L<=q and q<=customer.U else 1, edgecolor='green'))


        # Plot arrows for precedence for each route

        arrow_color = colors[i % len(colors)]
        x_values = [mine.x] + [customers[customer_id].x for customer_id,_ in route] + [mine.x]
        y_values = [mine.y] + [customers[customer_id].y for customer_id,_ in route] + [mine.y]
        for a,b in zip(list(zip(x_values,y_values))[1:],list(zip(x_values,y_values))[:-1]):
            plt.arrow(b[0], b[1],
                    a[0] - b[0],
                    a[1] - b[1],
                    shape='full', color=arrow_color, length_includes_head=True, head_width=3, head_length=max(25-i*3,10), alpha=.7)

        #plt.title(f'Route at time T={ts} for truck #{truck}')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.legend()
        route=Route(self.mine,route,self.pizzeria_dict)
        plt.title(f'{title}\n Route cost: {route.distance_cost}, truck load: {route.total_goods}/{self.Q}', color='black' if route.total_goods<=self.Q else 'red')
        plt.grid(True)
        os.makedirs(self.visualization_directory, exist_ok=True)
        plt.savefig(os.path.join(self.visualization_directory,f'plot_{image_id}.png'))
        plt.clf()
        plt.close()
        return new_depq,new_customer_quantities

    def visualize(self,sol):
        inst=self
        idx = 0
        for t in range(inst.T):
            pizzeria_quantities = {c.id:c.generate_cost_and_validity(sol)[-1][t][0] for c in inst.pizzerias}
            mine_quantity_pre,mine_quantity_post,_ = inst.mine.generate_cost_and_validity(sol)[-1][t]
            inst.generate_setup_visualization(sol,mine_quantity_pre,pizzeria_quantities,f'Situation at time t={t} (Pre-production)',idx)
            idx+=1
            inst.generate_setup_visualization(sol,mine_quantity_post,pizzeria_quantities,f'Situation at time t={t} (Post-production)',idx)
            idx+=1
            for truck in range(inst.M):
                mine_quantity_post,pizzeria_quantities=inst.visualize_route(truck,
                            sol.raw[t][truck],
                            mine_quantity_post,
                            pizzeria_quantities,
                            f'Truck #{truck} at time t={t}',
                            idx
                )
                idx+=1
            inst.generate_setup_visualization(sol,mine_quantity_post,{c.id:c.generate_cost_and_validity(sol)[-1][t][-1] for c in inst.pizzerias},f'Situation at time t={t} (Post-consumption)',idx)
            idx+=1
        images = [imageio.imread(os.path.join(inst.visualization_directory, f'plot_{i}.png')) for i in range(idx)]
        imageio.mimsave(f'animation{self.visualization_directory}.gif', images, fps=.5)

        self.visualize_mine(sol)
        self.visualize_inventory_sets(sol)

    def generate_file(self,sol:Solution):
        """Generates the solution file

        Args:
            sol (Solution): your solution
        """
        os.makedirs('solutions',exist_ok=True)
        with open('solutions/'+self.filepath.split('.')[-2].split(os.sep)[-1]+'.txt', 'w') as f:
            f.write(str(self.T)+' '+str(self.M)+'\n')
            for t in range(self.T):
                for truck in range(self.M):
                    if truck<len(sol.raw[t]):
                        f.write(str(len(sol.raw[t][truck]))+' ')
                        for tup in sol.raw[t][truck]:
                            f.write(str(tup)+' ')
                        f.write('\n')
                    else:
                        f.write('0\n')
                f.write('\n\n')

    def __repr__(self):
        s = f'Instance {self.filepath}:\n'
        s += "\t"+self.mine.__repr__()+'\n'
        for z in self.pizzerias:
            s+="\t"+z.__repr__()+'\n'
        return s
