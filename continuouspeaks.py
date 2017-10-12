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
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array


"""
Commandline parameter(s):
   none
"""

N=60
T=N/10
fill = [2] * N
ranges = array('i', fill)

ef = ContinuousPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

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
      # print "RHC " + str(ef.value(rhc.getOptimal())), isample
        value += ef.value(rhc.getOptimal())
    end = time.time()
    clock_time = (end - start)/nsample    
    value = round(value/nsample,2)    
    print "RHC " + str(value),  iters,  clock_time

   
#-- Simulated Annealing    
sa = SimulatedAnnealing(1E11, .95, hcp)
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
    print "SA " + str(value),  iters, clock_time

    
#-- Genetic Algorithm
ga = StandardGeneticAlgorithm(200, 100, 10, gap)
for iters in niters:
    start = time.time()
    fit = FixedIterationTrainer(ga, iters)
    value = 0
    for isample in range(nsample):
        fit.train()
        value += ef.value(ga.getOptimal())
    end = time.time()
    clock_time = (end - start)/nsample    
    value = round(value/nsample,2)    
    print "GA " + str(value),  iters, clock_time

#-- MIMIC 
mimic = MIMIC(200, 20, pop)
niters = [50, 100, 200, 500, 600, 700, 800, 1000, 1500, 2000]
for iters in niters:
    start = time.time()
    fit = FixedIterationTrainer(mimic, iters)
    value = 0
    for isample in range(nsample):
        fit.train()
        value += ef.value(mimic.getOptimal())
    end = time.time()
    clock_time = (end - start)/nsample    
    value = round(value/nsample,2)    
    print "MIMIC " + str(value),  iters, clock_time
