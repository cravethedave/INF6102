#!/home/cravethedave/session/INF6102/devoirs/.venv/bin/python3

import argparse
import solver_naive
import solver_advanced
import ben_solution
import ben_solution_2
import david_first
import david_second
import david_third
import time
from utils import Instance


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--agent', type=str, default='naive')
    parser.add_argument('--infile', type=str, default='super_easy.dat')
    parser.add_argument('--outdir', type=str, default='solutions')
    parser.add_argument('--time', type=int, default=300)
    parser.add_argument('--visualisation_file', type=str, default='visualization')
    parser.add_argument('--viz', action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    instance = Instance(args.infile)

    print("***********************************************************")
    print("[INFO] Start the solving: pizzeria furnaces")
    print("[INFO] input file: %s" % instance.filepath)
    print("[INFO] output directory: %s" % args.outdir)
    print("[INFO] visualization file: %s" % args.visualisation_file)
    print("[INFO] number of timesteps: %s" % (instance.T))
    print("[INFO] number of pizzerias: %s" % (instance.npizzerias))
    print(f"[INFO] number of trucks : {instance.M}")
    print("***********************************************************")

    start_time = time.time()

    # Méthode à implémenter
    if args.agent == "naive":
        # assign a different time slot for each course
        solution = solver_naive.solve(instance)
    elif args.agent == "advanced":
        solution = solver_advanced.solve(instance)
    elif args.agent == "d1":
        solution = david_first.solve(instance)
    elif args.agent == "d2":
        solution = david_second.solve(instance)
    elif args.agent == "d3":
        solution = david_third.solve(instance, args.time)
    elif args.agent == "ben":
        solution = ben_solution.solve(instance)
    elif args.agent == "ben2":
        solution = ben_solution_2.solve(instance, 60)
    else:
        raise Exception("This agent does not exist")


    solving_time = round((time.time() - start_time) / 60,2)

    # You can disable the display if you do not want to generate the visualization
    if args.viz:
        instance.visualize(solution)
    #
    instance.generate_file(solution)
    cost = instance.solution_cost_and_validity(solution)
    pizzeria_cost,pizzeria_cost_matrix,pizzeria_validity,pizzeria_validity_matrix =  instance.solution_pizzeria_cost_and_validity(solution)
    mine_cost,mine_cost_matrix,mine_validity,mine_validity_matrix = instance.solution_mine_cost_and_validity(solution)
    route_cost,route_cost_matrix,route_validity,route_validity_matrix = instance.solution_route_cost_and_validity(solution)
    timestep_validity,_ = instance.solution_timestep_validity(solution)

    print("***********************************************************")
    print("[INFO] Solution obtained")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print(f"[INFO] Cost for Marco : {cost[0]}\n\t Loss due to pizzeria coal inventory : {pizzeria_cost}\n\t Luigi charged : {mine_cost}\n\t Trucks did cost : {route_cost}")
    print(f"[INFO] Sanity check passed : {cost[-1]}\n\t Truck capacity is respected: {route_validity}\n\t Luigi's coal inventory is never negative : {mine_validity}\n\t Pizzerias bounds are respected : {pizzeria_validity}\n\t Routes do not overlap : {timestep_validity}")
    print("***********************************************************")
