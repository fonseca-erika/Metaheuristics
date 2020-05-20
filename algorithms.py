import pygmo as pg
import numpy as np
from shifted_go import * # shifted global optimum

def sa(objective_function, Ts=10.0, Tf=0.1, n_T_adj=10, n_range_adj=10, bin_size=10, 
                                               start_range=1.0, pop_size=15):
    
    """
    Simulated Annealing (Corana’s version)

    Parameters
    - Ts (float) – starting temperature
    - Tf (float) – final temperature
    - n_T_adj (int) – number of temperature adjustments in the annealing schedule
    - n_range_adj (int) – number of adjustments of the search range performed at a constant temperature
    - bin_size (int) – number of mutations that are used to compute the acceptance rate
    - start_range (float) – starting range for mutating the decision vector
    - pop_size (int)  – the number of individuals
    
    """
    logs =[]
    problem = pg.problem(objective_function)
    algorithm = pg.algorithm(pg.simulated_annealing(Ts=Ts, Tf=Tf, n_T_adj=n_T_adj, n_range_adj=n_range_adj, bin_size=bin_size, 
                                               start_range=start_range))
    algorithm.set_verbosity(50)
    population = pg.population(prob=problem, size=pop_size)
    solution = algorithm.evolve(population)

    """
    get_logs output is a list of tuples with the following structure:
    - Fevals (int), number of functions evaluation made
    - Best (float), the best fitness function found so far
    - Current (float), last fitness sampled
    - Mean range (float), the mean search range across the decision vector components (relative to the box bounds width)
    - Temperature (float), the current temperature
    """
    
    logs = np.array(algorithm.extract(pg.simulated_annealing).get_log())[:,(0,1)] # taking only function evaluations and best fitness

    algo_ = algorithm.get_name()
    function_ = objective_function.get_name()

    return {'champion solution': solution.champion_f, 'champion coordinates': solution.champion_x, 
            'log': logs, 'algorithm':algo_, 'problem': function_}


def pso(objective_function, gen=2000, omega= .7, eta1=0.5, eta2=4, max_vel = .05, variant = 6, neighb_type = 2, 
                neighb_param = 4, memory=False, pop_size=15):
    
    """
    Particle Swarm Optimization (generational) is identical to pso, but does update the velocities of each particle 
    before new particle positions are computed (taking into consideration all updated particle velocities). Each particle 
    is thus evaluated on the same seed within a generation as opposed to the standard PSO which evaluates single particle 
    at a time. Consequently, the generational PSO algorithm is suited for stochastic optimization problems.
    
    Parameters: 
    - objective_function - instance of the class of the objective function
    - gen (int) – number of generations
    - omega (float) – inertia weight (or constriction factor)
    - eta1 (float) – social component
    - eta2 (float) – cognitive component
    - max_vel (float) – maximum allowed particle velocities (normalized with respect to the bounds width)
    - variant (int) – algorithmic variant
    - neighb_type (int) – swarm topology (defining each particle’s neighbours)
    - neighb_param (int) – topology parameter (defines how many neighbours to consider)
    - memory (bool) – when true the velocities are not reset between successive calls to the evolve method
    - pop_size (int)  – the number of individuals
    
    """
    logs = []
    problem = pg.problem(objective_function)
    algorithm = pg.algorithm(pg.pso_gen(gen=gen, omega=omega, eta1=eta1, eta2=eta2, max_vel=max_vel, variant=variant, 
                                    neighb_type=neighb_type, neighb_param=neighb_param, memory=memory))
    algorithm.set_verbosity(50)
    solution = pg.population(prob=problem, size=pop_size, b=None, seed=None)
    solution = algorithm.evolve(solution)

    """
    get_logs output is a list of tuples with the following structure:
    - Gen (int), generation number
    - Fevals (int), number of functions evaluation made
    - gbest (float), the best fitness function found so far by the the swarm
    - Mean Vel. (float), the average particle velocity (normalized)
    - Mean lbest (float), the average fitness of the current particle locations
    - Avg. Dist. (float), the average distance between particles (normalized)
    """

    logs = np.array(algorithm.extract(pg.pso_gen).get_log())[:,(1,2)] # taking only function evaluations and best fitness
    algo_ = algorithm.get_name() 
    function_ = objective_function.get_name()

    return {'champion solution': solution.champion_f, 'champion coordinates': solution.champion_x, 
            'log': logs, 'algorithm':algo_, 'problem': function_}


def cmaes(objective_function, gen=1000, cc=- 1, cs=- 1, c1=- 1, cmu=- 1, sigma0=0.5, ftol=1e-06, 
            xtol=1e-06, memory=False, force_bounds=True, pop_size=15):
    
    """
    Covariance Matrix Evolutionary Strategy (CMA-ES)

    Parameters
    - gen (int) – number of generations
    - cc (float) – backward time horizon for the evolution path (by default is automatically assigned)
    - cs (float) – makes partly up for the small variance loss in case the indicator is zero (by default is 
    automatically assigned)
    - c1 (float) – learning rate for the rank-one update of the covariance matrix (by default is automatically assigned)
    - cmu (float) – learning rate for the rank-mu update of the covariance matrix (by default is automatically assigned)
    - sigma0 (float) – initial step-size
    - ftol (float) – stopping criteria on the x tolerance
    - xtol (float) – stopping criteria on the f tolerance
    - memory (bool) – when true the adapted parameters are not reset between successive calls to the evolve method
    - force_bounds (bool) – when true the box bounds are enforced. The fitness will never be called outside the bounds
    but the covariance matrix adaptation mechanism will worsen
    - pop_size (int)  – the number of individuals
    
    """
    logs =[]
    problem = pg.problem(objective_function)
    algorithm = pg.algorithm(pg.cmaes(gen=gen, cc=cc, cs=cs, c1=c1, cmu=cmu, sigma0=sigma0, ftol=ftol, 
                                      xtol=xtol, force_bounds=force_bounds, memory=memory))
    algorithm.set_verbosity(50)
    solution = pg.population(prob=problem, size=pop_size, b=None, seed=None)
    solution = algorithm.evolve(solution)
    
    """
    get_logs output is a list of tuples with the following structure:
    - Gen (int), generation number
    - Fevals (int), number of functions evaluation made
    - Best (float), the best fitness function currently in the population
    - dx (float), the norm of the distance to the population mean of the mutant vectors
    - df (float), the population flatness evaluated as the distance between the fitness of the best and of the worst individual
    - sigma (float), the current step-size
    """
    
    logs = np.array(algorithm.extract(pg.cmaes).get_log())[:,(1,2)] # taking only function evaluations and best fitness

    algo_ = algorithm.get_name()
    function_ = objective_function.get_name()

    return {'champion solution': solution.champion_f, 'champion coordinates': solution.champion_x, 
            'log': logs, 'algorithm':algo_, 'problem': function_}


def sade(objective_function, gen=100, allowed_variants=[2], variant_adptv=1, ftol=1e-06, xtol=1e-06, pop_size=15):
    
    """
    Self-adaptive Differential Evolution, pygmo flavour (pDE)
    Parameters
    - gen (int) – number of generations
    - allowed_variants (array-like object) – allowed mutation variants, each one being a number in [1, 18]
    - variant_adptv (int) – F and CR parameter adaptation scheme to be used (one of 1..2)
    - ftol (float) – stopping criteria on the x tolerance (default is 1e-6)
    - xtol (float) – stopping criteria on the f tolerance (default is 1e-6)
    - memory (bool) – when true the adapted parameters CR anf F are not reset between successive calls to the evolve method
    """
    logs =[]
    problem = pg.problem(objective_function)
    algorithm = pg.algorithm(pg.de1220(gen=gen, allowed_variants=allowed_variants, variant_adptv=variant_adptv, 
                                       ftol=ftol, xtol=xtol))
    algorithm.set_verbosity(50)
    population = pg.population(prob=problem, size=pop_size)
    solution = algorithm.evolve(population)
    
    """
    get_logs output is a list of tuples with the following structure:
    - Gen (int), generation number
    - Fevals (int), number of functions evaluation made
    - Best (float), the best fitness function currently in the population
    - F (float), the value of the adapted paramter F used to create the best so far
    - CR (float), the value of the adapted paramter CR used to create the best so far
    - Variant (int), the mutation variant used to create the best so far
    - dx (float), the norm of the distance to the population mean of the mutant vectors
    - df (float), the population flatness evaluated as the distance between the fitness of the best and of the 
    worst individual
    """
    
    logs = np.array(algorithm.extract(pg.de1220).get_log())[:,(1,2)] # taking only function evaluations and best fitness

    algo_ = algorithm.get_name()
    function_ = objective_function.get_name()

    return {'champion solution': solution.champion_f, 'champion coordinates': solution.champion_x, 
            'log': logs, 'algorithm':algo_, 'problem': function_}


def sga(objective_function, gen=200, cr=0.9, eta_c=1.0, m=0.02, param_m=1.0, param_s=2, 
        crossover='exponential', mutation='polynomial', selection='tournament', pop_size=15):
    
    """
    Parameters
    - gen (int) – number of generations.
    - cr (float) – crossover probability.
    - eta_c (float) – distribution index for sbx crossover. This parameter is inactive if other types of crossover 
    are selected.
    - m (float) – mutation probability.
    - param_m (float) – distribution index (polynomial mutation), gaussian width (gaussian mutation) or 
    inactive (uniform mutation)
    - param_s (float) – the number of best individuals to use in “truncated” selection or the size of the 
    tournament in tournament selection.
    - crossover (str) – the crossover strategy. One of exponential, binomial, single or sbx
    - mutation (str) – the mutation strategy. One of gaussian, polynomial or uniform.
    - selection (str) – the selection strategy. One of tournament, “truncated”.

    """
    logs =[]
    problem = pg.problem(objective_function)
    algorithm = pg.algorithm(pg.sga(gen=gen, cr=cr, eta_c=eta_c, m=m, param_m=param_m, param_s=param_s, 
                            crossover=crossover, mutation=mutation, selection=selection))
    algorithm.set_verbosity(50)
    population = pg.population(prob=problem, size=pop_size)
    solution = algorithm.evolve(population)
    
    """
    get_logs output is a list of tuples with the following structure:
    - Gen (int), generation number
    - Fevals (int), number of functions evaluation made
    - Best (float), the best fitness function currently in the population
    - dx (float), the norm of the distance to the population mean of the mutant vectors
    - df (float), the population flatness evaluated as the distance between the fitness of the best and of the worst individual
    - sigma (float), the current step-size
    """
    
    logs = np.array(algorithm.extract(pg.sga).get_log())[:,(1,2)] # taking only function evaluations and best fitness

    algo_ = algorithm.get_name()
    function_ = objective_function.get_name()

    return {'champion solution': solution.champion_f, 'champion coordinates': solution.champion_x, 
            'log': logs, 'algorithm':algo_, 'problem': function_}