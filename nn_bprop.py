"""
Backprop NN training on Adult data (Feature selection complete)
"""

from __future__ import with_statement
import os
import csv
import time
import sys
sys.path.append("../ABAGAIL/ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import  LogisticSigmoid #-------------------------------------------------------

# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 66
HIDDEN_LAYER = 5
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 1000


def initialize_instances(infile):
    """Read the m_trg.csv CSV data into a list of instances."""
    instances = []

    # Read in the CSV file
    with open(infile, "r") as dat:
        reader = csv.reader(dat)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) < 0 else 1))
            instances.append(instance)

    return instances
	

def errorOnDataSet(network,ds,measure):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0
    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()
        actual = instance.getLabel().getContinuous()
        predicted = network.getOutputValues().get(0)
        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
    MSE = error/float(N)
    acc = correct/float(correct+incorrect)
    return MSE,acc
	
	
def train(oa, network, oaName, training_ints,testing_ints, measure):
    """Train a given network on a set of instances.
    """
    print "\nError results for %s\n---------------------------" % (oaName,)
    times = [0]
    for iteration in xrange(TRAINING_ITERATIONS):
        start = time.clock()
        oa.train()
        elapsed = time.clock()-start
    	times.append(times[-1]+elapsed)
        if iteration % 10 == 0:
    	    MSE_trg, acc_trg = errorOnDataSet(network,training_ints,measure)
            MSE_tst, acc_tst = errorOnDataSet(network,testing_ints,measure)
            print iteration,MSE_trg,MSE_tst,acc_trg,acc_tst,times[-1]

def main():
    """Run this experiment"""
    training_ints = initialize_instances('./clean_data/adult_train.txt')
    testing_ints = initialize_instances('./clean_data/adult_test.txt')
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    logunit = LogisticSigmoid() #---------------------------------------------------------------
    rule = RPROPUpdateRule()
    oa_names = ["Backprop"]
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER],logunit)
    train(BatchBackPropagationTrainer(data_set,classification_network,measure,rule), classification_network, 'Backprop', training_ints,testing_ints, measure)
        

if __name__ == "__main__":
    main()
