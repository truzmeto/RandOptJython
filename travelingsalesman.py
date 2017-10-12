# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes 
# to a file and plot them in your favorite tool.
import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array


"""
Commandline parameter(s):
    none
"""

# set N value.  This is the number of points
N = 40
random = Random()

points = [[0 for x in xrange(2)] for x in xrange(N)]

for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()
#print points

    
ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscretePermutationDistribution(N)
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

nsample = 10
niters = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

#-- R-Hill Climbing
rhc = RandomizedHillClimbing(hcp)
for iters in niters:
    start = time.time()
    fit = FixedIterationTrainer(rhc, iters)
    value = 0
    for isample in range(nsample):
        fit.train()
        value += ef.value(rhc.getOptimal())
    end = time.time()
    clock_time = (end - start)/nsample    
    value = round(value/nsample,2)    
    print "RHC " + str(1/value),   iters,  clock_time

path = []
for x in range(0,N):
    path.append(rhc.getOptimal().getDiscrete(x))
    print "Rout_RHC", path[x] , points[path[x]][0], points[path[x]][1]

#-- Simulated Annealing
sa = SimulatedAnnealing(1E12, .999, hcp)
for iters in niters:
    start = time.time()
    fit = FixedIterationTrainer(sa, iters)
    value = 0
    for isample in range(nsample):
        fit.train()
        value += ef.value(sa.getOptimal())
    end = time.time()
    clock_time = (end - start)/nsample    
    value = round(value/nsample,2)    
    print "SA " + str(1/value),  iters, clock_time

path = []
for x in range(0,N):
    path.append(sa.getOptimal().getDiscrete(x))
    print "Rout_SA", path[x] , points[path[x]][0], points[path[x]][1]

    
#-- Genetic Algorithm
ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
for iters in niters:
    start = time.time()
    fit = FixedIterationTrainer(ga, iters)
    #value = 0
    #for isample in range(nsample):
    fit.train()
    #value += ef.value(ga.getOptimal())
    value = ef.value(ga.getOptimal())
    end = time.time()
    clock_time = (end - start)#/nsample    
    value = round(value,2)    
    print "GA " + str(1/value),  iters, clock_time

path = []
for x in range(0,N):
    path.append(ga.getOptimal().getDiscrete(x))
    print "Rout_GA", path[x] , points[path[x]][0], points[path[x]][1]

#-- MIMIC    
# for mimic we use a sort encoding
ef = TravelingSalesmanSortEvaluationFunction(points);
fill = [N] * N
ranges = array('i', fill)
odd = DiscreteUniformDistribution(ranges);
df = DiscreteDependencyTree(.1, ranges); 
pop = GenericProbabilisticOptimizationProblem(ef, odd, df);

mimic = MIMIC(500, 100, pop)
niters = [50, 100, 200, 500, 600, 700, 800, 1000, 1500, 2000]
for iters in niters:
    start = time.time()
    fit = FixedIterationTrainer(mimic, iters)
    # value = 0
    # for isample in range(nsample):
    fit.train()
    #value += ef.value(mimic.getOptimal())
    value = ef.value(mimic.getOptimal())
    end = time.time()
    clock_time = (end - start)#/nsample    
    value = round(value,2)    
    print "MIMIC " + str(1/value),  iters, clock_time

path = []
optimal = mimic.getOptimal()
fill = [0] * optimal.size()
ddata = array('d', fill)
for i in range(0,len(ddata)):
    ddata[i] = optimal.getContinuous(i)
order = ABAGAILArrays.indices(optimal.size())
ABAGAILArrays.quicksort(ddata, order)
print order
