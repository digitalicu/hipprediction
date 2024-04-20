from data import DataManager
import argparse
import numpy as np
import matplotlib.pyplot
import pickle
import pygad
import os
import json
from datetime import datetime
from ml import mlp_classifier

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
parser.add_argument("-ln", 
                    "--layers_number",
                    help="Number of layers", 
                    default=5,
                    type=int)
parser.add_argument("-ls", 
                    "--layers_size",
                    help="Size of layers", 
                    default=10,
                    type=int)
args = parser.parse_args()

dm = DataManager(args.data)
dm.set_feature(args.outcome)

if args.procedure:
    dm.switch_procedure()

if not os.path.exists("mlp"):
    os.mkdir("mlp")

RESULTS_FOLDER = os.path.join("mlp", args.folder)
if not os.path.exists(RESULTS_FOLDER):
    os.mkdir(RESULTS_FOLDER)

CURRENT_BEST_SOLUTION = {
    "solution": [],
    "validation": 0,
}

layers_info = [[m for m in range(args.layers_size + 1)] for _ in range(args.layers_number)]

def on_gen(ga_instance):
    global CURRENT_BEST_SOLUTION, RESULTS_FOLDER
    best_solution_data = list(ga_instance.best_solution())
    best_solution_data[0] = [s for s in best_solution_data[0] if s > 0]
    if not np.array_equal(best_solution_data[0], np.array(CURRENT_BEST_SOLUTION["solution"])) and CURRENT_BEST_SOLUTION["validation"] < best_solution_data[1]:
        CURRENT_BEST_SOLUTION["solution"] = best_solution_data[0]
        CURRENT_BEST_SOLUTION["validation"] = best_solution_data[1]

    with open(os.path.join(RESULTS_FOLDER, "state"), "w") as f:
        f.write("Time : %s\r\n" % (datetime.now()))
        f.write("Generation : %s\r\n" % (ga_instance.generations_completed))
        f.write("Fitness of the best solution : %s\r\n" % (CURRENT_BEST_SOLUTION["validation"]))
        f.write(json.dumps(list([int(s) for s in CURRENT_BEST_SOLUTION["solution"] if s > 0])))

def fitness_func(ga_instance, solution, solution_idx):
    global dm
    real_solution = [s for s in solution if s > 0]
    return mlp_classifier(dm, hidden_layers=real_solution)

ga_instance = pygad.GA(num_generations=args.generations,
                       sol_per_pop=50,
                       parallel_processing=16,
                       num_parents_mating=5,
                       on_generation=on_gen,
                       keep_parents = 1,
                       num_genes=len(layers_info),
                       fitness_func=fitness_func,
                       mutation_percent_genes=20,
                       gene_space=layers_info,
                       gene_type=int)

ga_instance.run()

matplotlib.pyplot.figure()
plt = matplotlib.pyplot.plot(ga_instance.best_solutions_fitness)
matplotlib.pyplot.savefig(os.path.join(RESULTS_FOLDER, "fitness.jpg"))
pickle.dump(plt, open(os.path.join(RESULTS_FOLDER, "fitness.pickle"), "wb"))