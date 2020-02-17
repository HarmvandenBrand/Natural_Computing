import array
import random
import operator
import math
from deap import creator, base, tools, algorithms, gp
import numpy
import matplotlib.pyplot as plt

# important resources
# https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# https://github.com/DEAP/deap/blob/f86583f2276e9fab127048b2c367a04bd13ee8c5/examples/gp/symbreg.py

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# add operations
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(math.exp, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))

pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# create an array containting the outputs
outputs = [0, -.1629,-0.2624,-.3129,-3264,-.3125,-.2784,-.2289,-.1664,-.0909,0,.1111,.2496,.4251,.6496,.9375,1.3056,1.7731,2.3616,3.0951,4]

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    errors = 0
    for x,y in zip(points, outputs):
        errors = errors + math.fabs(func(x) - y)

    return errors,


toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,11)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    random.seed(318)

    #define population size
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

    # choose what statistics to register
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # run an evolutionary algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb= 0.7, mutpb= 0, ngen= 50, stats=mstats,
                                   halloffame=hof, verbose=True)

    # plot the results
    plt.plot(log.chapters['fitness'].select("gen", "max")[0], log.chapters['fitness'].select("gen", "max")[1])
    plt.ylabel('best fitness')
    plt.xlabel('generation')
    plt.title('best of generation fitness vs generation')
    plt.show()
    plt.plot(log.chapters['size'].select("gen", "max")[0], log.chapters['size'].select("gen", "max")[1])
    plt.ylabel('best size')
    plt.xlabel('generation')
    plt.title('best of generation size vs generation')
    plt.show()
    return pop, log, hof


if __name__ == "__main__":
    main()