import sys
from os.path import dirname
import logging

#Add src folder to python path
import os

src = dirname(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(src)

from experiment import Experiment
from runners.simple_reweighting_runner import run_solving_simple as simple
from runners.random_reweighting_runner import run_solving_random as random


class ExperimentRunner:
    def __init__(self):
        self.experiments = []

    def add_experiment(self, experiment):
        self.experiments.append(experiment)

    def run_experiments(self, experiment):
        experiment.run_experiment()

    def run_all_experiments(self):
        for experiment in self.experiments:
            print("-------------\nRunning experiment",self.experiments.index(experiment)+1)
            experiment.run_experiment()

if __name__ == "__main__":
    # Create instance of Experiment with different reweighting functions
    experiment1 = Experiment(5,2,[simple,random],2)
    experiment2 = Experiment(10,1,[simple,random],100)

    #Create an ExperimentRunner
    runner = ExperimentRunner()

    #Add experiments
    runner.add_experiment(experiment1)
    runner.add_experiment(experiment2)

    #Run all experiments
    runner.run_all_experiments()

