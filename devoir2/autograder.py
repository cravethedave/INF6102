import argparse
import solver_naive
import solver_advanced
import time
from utils import Instance
from utils import *



if __name__ == '__main__':
    grade = 0
    for instanceId,optimum,factor,maxgrade in zip(['A','B','C','D','E'],[1640.93,2475.88,2061.27,2911.11,3487.12],[2,2,2,3,3],[1,2,2,2,2]):
        try:
            inst = Instance(f'./instances/instance{instanceId}.txt')
            try:
                sol = Solution.from_file(f'./solutions/instance{instanceId}.txt',inst.npizzerias)
                cost,validity = inst.solution_cost_and_validity(sol)
                grade += (maxgrade-maxgrade*(cost-optimum)/(factor*optimum-optimum))*int(validity)
                print(f'instance{instanceId} : ',round((maxgrade-maxgrade*(cost-optimum)/(factor*optimum-optimum))*int(validity),maxgrade),f'/{maxgrade}')
            except FileNotFoundError as e:
                print(f'instance{instanceId} : ',0,f'/{maxgrade} (file ./solutions/instance{instanceId}.txt not found)')
                grade+=0
        except FileNotFoundError as e:
            print(f'instance{instanceId} : ',0,f'/{maxgrade} (file ./instances/instance{instanceId}.txt not found)')
            grade+=0
    print(f'Total ',round(grade,2),f'/9')
