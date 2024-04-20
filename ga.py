from data import DataManager
import argparse
import numpy as np
import matplotlib.pyplot
import pickle
import pygad
import os
import json
from datetime import datetime
from ml import logistic_regression, random_forest

parser = argparse.ArgumentParser()
parser.add_argument("-g", 
                    "--generations",
                    help="Number of generations", 
                    default=10,
                    type=int)
parser.add_argument("-t", 
                    "--threads",
                    help="Number of parallel processes", 
                    default=8,
                    type=int)
parser.add_argument("-o", 
                    "--outcome",
                    help="Outcome to check: 'm30', 'm90', 'm180', 'm365'", 
                    default="m30",
                    type=str)
parser.add_argument("-f", 
                    "--folder",
                    help="Folder to save results", 
                    default="default",
                    type=str)
parser.add_argument("-p", 
                    "--procedure",
                    help="To include procedure as parameter", 
                    action=argparse.BooleanOptionalAction,
                    type=bool)
parser.add_argument("-d", 
                    "--data",
                    help="Data *.csv", 
                    default="full.csv",
                    type=str)
parser.add_argument("-a", 
                    "--algorithm",
                    help="Wich algorithm to use 'lr' or 'rf", 
                    default="lr",
                    type=str)
args = parser.parse_args()

dm = DataManager(args.data)
dm.set_feature(args.outcome)

if args.procedure:
    dm.switch_procedure()

if not os.path.exists(args.algorithm):
    os.mkdir(args.algorithm)

RESULTS_FOLDER = os.path.join(args.algorithm, args.folder)
if not os.path.exists(RESULTS_FOLDER):
    os.mkdir(RESULTS_FOLDER)

CURRENT_BEST_SOLUTION = {
    "solution": [],
    "validation": 0,
}

columns = dm.get_tested_features()
gene_space = [0, 1]

def on_gen(ga_instance):
    global columns, CURRENT_BEST_SOLUTION
    best_solution_data = ga_instance.best_solution()
    if not np.array_equal(best_solution_data[0], np.array(CURRENT_BEST_SOLUTION["solution"])) and CURRENT_BEST_SOLUTION["validation"] < best_solution_data[1]:
        CURRENT_BEST_SOLUTION["solution"] = best_solution_data[0]
        validation_auc = best_solution_data[1]
        CURRENT_BEST_SOLUTION["validation"] = validation_auc

    with open(os.path.join(RESULTS_FOLDER, "state"), "w") as f:
        f.write("Time : %s\r\n" % (datetime.now()))
        f.write("Generation : %s\r\n" % (ga_instance.generations_completed))
        f.write("Fitness of the best solution : %s\r\n" % (CURRENT_BEST_SOLUTION["validation"]))
        f.write(json.dumps([columns[i] for i, s in enumerate(CURRENT_BEST_SOLUTION["solution"]) if s]))   

def fitness_func(ga_instance, solution, solution_idx):
    global columns, dm
    selected_columns = [columns[i] for i, s in enumerate(solution) if s]
    dm.set_tested_features(selected_columns)
    
    if args.algorithm == 'lr':
        return logistic_regression(dm)
    elif args.algorithm == 'rf':
        return random_forest(dm)

ga_instance = pygad.GA(num_generations=args.generations,
                       sol_per_pop=100,
                       parallel_processing=args.threads,
                       num_parents_mating=5,
                       on_generation=on_gen,
                       keep_parents = 1,
                       num_genes=len(columns),
                       fitness_func=fitness_func,
                       mutation_percent_genes=20,
                       gene_space=gene_space,
                       gene_type=int)

ga_instance.run()

matplotlib.pyplot.figure()
plt = matplotlib.pyplot.plot(ga_instance.best_solutions_fitness)
matplotlib.pyplot.savefig(os.path.join(RESULTS_FOLDER, "fitness.jpg"))
pickle.dump(plt, open(os.path.join(RESULTS_FOLDER, "fitness.pickle"), "wb"))