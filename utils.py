import matplotlib.pyplot as plt
import pygmo as pg
import pandas as pd
import numpy as np
import time

def report_convergence(solution, fevals = None):
        
    logs = solution['log']
    
    plt.plot([l[0] for l in logs],[l[1] for l in logs], label = solution['algorithm']) 

    # We then add details to the plot
    plt.title('Convergence curve for ' + solution['problem'])
    plt.legend() 
    plt.ylabel('objective function')
    plt.xlabel('objective evals')
    plt.grid() 
    
    if fevals == None:
        print("Number of function evaluations: ", logs[-1][0])
    else:
        print("Number of function evaluations: ", fevals)
    print("Final solution vector: ", solution['champion coordinates'] )
    print("Fitness: ", solution['champion solution'])
    
    
def statistics(objective_function, udas, epochs=25, pop_size=1):
    
    prob = pg.problem(objective_function)
    
    stats = []
    for uda in udas: 
        best = []
        evals = []
        start_time = time.time()
        for i in range(epochs):
            algo = pg.algorithm(uda)
            algo.set_verbosity(1) # regulates both screen and log verbosity
            pop = pg.population(prob, pop_size)
            sol = algo.evolve(pop)
            best.append(sol.champion_f)
            evals = sol.problem.get_fevals()

        end_time = time.time()

        parameters = algo.get_extra_info().split('\t')
        parameters = [parameter.replace('\n','') for parameter in parameters]
        parameters = list(filter(lambda x: x != "", parameters))

        stats.append([parameters, (end_time-start_time)/25, evals, np.max(np.array(best)), 
                        np.min(np.array(best)), np.mean(np.array(best)), np.median(np.array(best))])

    df_stats = pd.DataFrame(stats, columns=['parameters', 'avg computational time (sec)', 'avg function evals' , 
                                            'max', 'min', 'avg', 'median'])
    pd.set_option('display.max_colwidth', None)
    return df_stats