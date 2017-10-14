"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""
from __future__ import with_statement

import os
import csv
import time

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem


import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm


INPUT_FILE_train = os.path.join(".","clean_data","adult_train.txt")
INPUT_FILE_test = os.path.join(".","clean_data","adult_test.txt")


INPUT_LAYER = 11
HIDDEN_LAYER = 5
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 10

def initialize_instances(file_path):
    """Read the abalone.txt CSV data into a list of instances."""
    instances = []
    
    # Read in the adult_train.txt CSV file
    with open(file_path, "r") as adult:
        reader = csv.reader(adult)
           
        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            # my data was already preprocessed, so this basically does nothing but appends my data to instances
            instance.setLabel(Instance(0 if float(row[-1]) < 1 else 1))
            instances.append(instance)
                       
    return instances


def train(oa, network, oaName, instances, measure):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    print "\nError results for %s\n---------------------------" % (oaName,)

    for iteration in xrange(TRAINING_ITERATIONS):
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        print  round(error,3)


def main():
    """Run algorithms on the abalone dataset."""
    instances = initialize_instances(INPUT_FILE_train)
    instances_test = initialize_instances(INPUT_FILE_test)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(instances)

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    oa_names = ["RHC", "SA", "GA"]
    results = ""

    for name in oa_names:
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

    oa.append(RandomizedHillClimbing(nnop[0]))
    oa.append(SimulatedAnnealing(1E11, .95, nnop[1]))
    oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))

    for i, name in enumerate(oa_names):
        start = time.time()
        correct = 0
        incorrect = 0

        train(oa[i], networks[i], oa_names[i], instances, measure)
        end = time.time()
        training_time = end - start

        optimal_instance = oa[i].getOptimal()
        networks[i].setWeights(optimal_instance.getData())

        # here we make make prediction on training set-------------------------------------------------------------------------------------------------
        start = time.time()
        for instance in instances:
            networks[i].setInputValues(instance.getData())
            networks[i].run()
            ## swapped actual and predicted from original version
            actual = instance.getLabel().getContinuous()
            predicted = networks[i].getOutputValues().get(0)
            if abs(predicted - actual) < 0.5:
                correct += 1
            else:
                incorrect += 1

        end = time.time()
        testing_time = end - start

        results += "------------------------------- reporting on training set ---------------------------"
        results += "\nResults for %s: \nCorrectly classified %d instances." % (name, correct)
        results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, float(correct)/(correct+incorrect)*100.0)
        results += "\nTraining time: %0.03f seconds" % (training_time,)
        results += "\nTesting time: %0.03f seconds\n" % (testing_time,)
        
        # here we make make prediction on test set-----------------------------------------------------------------------------------------------------
        start = time.time()
        correct = 0
        incorrect = 0
        for instance in instances_test:
            networks[i].setInputValues(instance.getData())
            networks[i].run()
            ## swapped actual and predicted from original version
            actual = instance.getLabel().getContinuous()
            predicted = networks[i].getOutputValues().get(0)
            if abs(predicted - actual) < 0.5:
                correct += 1
            else:
                incorrect += 1
                
        end = time.time()
        testing_time = end - start

        results += "------------------------------- reporting on testing set ---------------------------"
        results += "\nResults for %s: \nCorrectly classified %d instances." % (name, correct)
        results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, float(correct)/(correct+incorrect)*100.0)
        results += "\nTesting time: %0.03f seconds\n" % (testing_time,)

        
    print results


if __name__ == "__main__":
    main()

