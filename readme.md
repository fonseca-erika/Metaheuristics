# Metaheuristics

According to the [Online Etymology Dictionary](https://www.etymonline.com/word/heuristic) the word heuristic comes from the Greek and means: "serving to discover or find out", while the prefix [meta](https://www.etymonline.com/search?q=meta-) is related to changing places. And by these definitions we can easily understand the goal of metaheurist in the field of optimization is to find good solutions to problems by exploring different areas of the domain of the problem.

Heuristics are typically used to solve complex (large, nonlinear, non-convex (i.e. contain local minima)) multivariate combinatorial optimization problems that are difficult to solve to optimality. It's commonly used when approximate solutions are sufficient to solve a problem.

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Metaheuristics_classification.svg/1024px-Metaheuristics_classification.svg.png "Euler diagram of the different classifications of metaheuristics")

In this project we are going to explore some of the metaheuristics algorithms applying them on continuous and discrete domains.

**Continuous optimization:**
The functions used come from the benchmark of the CEC’2008 Special Session and Competition on Large Scale Global Optimization:

 - F1: Shifted Sphere Function 
 - F2: Shifted Schwefel’s Problem 2.21 
 - F3:  Shifted Rosenbrock’s Function 
 - F4: Shifted Rastrigin’s Function 
 - F5:  Shifted Griewank’s Function 
 - F6 : Shifted Ackley’s Function

**Discrete optimization:**
Here we are going to explore the famous travel salesman problem (TSP) using two datasets:

 - 38 cities from Djibouti
 - 194 cities from Qatar
 

## Algorithms

### Simulated Annealing

Simulated annealing is a stochastic method. It is inspired by the physical process of annealing, in which a solid is first heated to a high enough temperature so that it melts, and then the temperature is decreased slowly; this allows the particles of the solid to arrange themselves in the lowest possible energy state and thus produce a highly structured lattice.
Source: [Hands-On Artificial Intelligence for IoT, Amita Kapoor, Packt Publishing]</i>

<img src="https://www.researchgate.net/profile/Pavan_Pagadala/publication/314924835/figure/fig3/AS:518417200889856@1500611700487/Hill-climbing-ability-of-simulated-annealing.png" width = 50%>

<i>Source: https://www.researchgate.net/profile/Pavan_Pagadala/publication/314924835/figure/fig3/AS:518417200889856@1500611700487/Hill-climbing-ability-of-simulated-annealing.png</i>

Transforming this physical process in an algorithm you define we start at a given temperature and at a random point, and start the iteration looking for improvement in the fitness of the function, if there's no improvement there is a probability p, which depends on the temperature, that allows to move to a worse area trying to bring diversity and not getting stuck at a local minimum.

The package PyGMO implements the Corana version of the simulated annealing algorithm, that is represented below:
   
<img src="https://www.researchgate.net/profile/Harrison_Barrett/publication/249970084/figure/fig10/AS:668982856146958@1536509350609/The-simulated-annealing-algorithm-implemented-by-Corana-et_W640.jpg" width = 50%>
<i>Source: https://www.researchgate.net/profile/Harrison_Barrett/publication/249970084/figure/fig10/AS:668982856146958@1536509350609/The-simulated-annealing-algorithm-implemented-by-Corana-et_W640.jpg</i>

The parameters of the algorithm are:
- Ts (float) – starting temperature
- Tf (float) – final temperature
- n_T_adj (int) – number of temperature adjustments in the annealing schedule
- n_range_adj (int) – number of adjustments of the search range performed at a constant temperature
- bin_size (int) – number of mutations that are used to compute the acceptance rate
- start_range (float) – starting range for mutating the decision vector
    
The number of fitness evaluations will be n_T_adj * n_range_adj * bin_size times the problem dimension.

### Differential Evolution (DE)

Differential evolution is a vector-based metaheuristic algorithm, that uses mutation, crossover, and selection to search for solutions that optimize a given function. The schema of the algorithm is shown below:

<img src='https://esa.github.io/pagmo2/_images/de.png'>
<i>Source: [PaGMO](https://esa.github.io/pagmo2/docs/cpp/algorithms/de.html)</i>

The configuration of a differential evolution is represented by the following code: DE/x/y/z, where x is the mutation scheme (random or best), y is the number of difference vectors, and z is the crossover scheme (binomial or exponential). 

The parameter $F$ is used as the differential weight and $C_r$ controls the probability for crossover. 

<i>Source: [Nature-Inspired Optimization Algorithms, by Xin-She Yang, Elsevier]</i>

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Ackley.gif/220px-Ackley.gif" width = 50%>
Source: [Wikipedia - Differential_evolution](https://en.wikipedia.org/wiki/Differential_evolution)

### PSO

The PSO algorithm searches the space of an objective function by adjusting the trajectories of individual agents, called particles, as the piecewise paths formed by positional vectors in a quasi-stochastic manner. The movement of a swarming particle consists of two major components: a stochastic component and a deterministic component. Each particle is attracted toward the position of the current global best and its own best location in history, while at the same time it has a tendency to move randomly. The algorithm is:
<img src='https://www.researchgate.net/publication/333314611/figure/fig2/AS:761735870939136@1558623392061/The-Particle-Swarm-Optimization-PSO-algorithm.png'>
<i>Source: https://www.researchgate.net/publication/333314611/figure/fig2/AS:761735870939136@1558623392061/The-Particle-Swarm-Optimization-PSO-algorithm.png </i>

The parameters of the algorithm are:
- The particle's current speed and direction of movement – representing inertia 
- The particle's best position found so far (local best) –representing cognitive force 
- The entire group's best position found so far (global best) – representing social force

<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/ParticleSwarmArrowsAnimation.gif/220px-ParticleSwarmArrowsAnimation.gif' width = 50%>

Source: [# Particle swarm optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization)

### Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

The CMA-ES is an evolutionary algorithm for difficult **non-linear non-convex black-box optimisation** problems in continuous domain, it is considered as state-of-the-art in evolutionary computation. The CMA-ES is typically applied to unconstrained or bounded constraint optimization problems, and search space dimensions between three and a hundred. The method is feasible on non-separable and/or badly conditioned problems, and also for **non-smooth** and even non-continuous problems. The flowchart below represents the steps of CMA-ES:

<img src ='https://ascelibrary.org/cms/asset/8490b093-ff12-4043-8072-952fc3c39ff1/figure3.gif'>

<i>Source: [## The CMA Evolution Strategy](http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaesintro.html)</i>

<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/Concept_of_directional_optimization_in_CMA-ES_algorithm.png/400px-Concept_of_directional_optimization_in_CMA-ES_algorithm.png' width = 50%>>

<i>Source: https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/Concept_of_directional_optimization_in_CMA-ES_algorithm.png/400px-Concept_of_directional_optimization_in_CMA-ES_algorithm.png</i>


## References

- [PyGMO Documentation](https://esa.github.io/pygmo2/overview.html)
- [Corana, A., Marchesi, M., Martini, C., & Ridella, S. (1987). Minimizing multimodal functions of continuous variables with the “simulated annealing” algorithm](https://people.sc.fsu.edu/~inavon/5420a/corana.pdf)
- [Nature-Inspired Optimization Algorithms](https://learning.oreilly.com/library/view/nature-inspired-optimization-algorithms/9780124167438/) by Xin-She YangPublished by  [Elsevier](https://learning.oreilly.com/library/publisher/elsevier/), 2014
- [Hands-On Artificial Intelligence for IoT](https://learning.oreilly.com/library/view/hands-on-artificial-intelligence/9781788836067/) by Amita KapoorPublished by  [Packt Publishing](https://learning.oreilly.com/library/publisher/packt-publishing/), 2019