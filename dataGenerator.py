from antOptimisation import AntOptimisationAlgo
from foodMaze import Maze
import matplotlib.pyplot as plt
import numpy as np


NUM_NODES = 4
GRID_SIZE = 2 * NUM_NODES - 1
NUM_TRIALS = 1
COMPLETED_LEVEL = "Level completed"
def run_test(decay_rate, alpha, beta, q0, q, swarm_size, iterations, num_pos):

    start = (GRID_SIZE // 2, GRID_SIZE // 2)
    num_food = iterations * swarm_size
    maze = Maze(NUM_NODES, num_food, num_pos)
    maze.maze[start[1]][start[0]] = 1
    #maze.maze = np.ones((GRID_SIZE, GRID_SIZE))

    antAlgo = AntOptimisationAlgo(swarm_size,
                                  GRID_SIZE,
                                  maze,
                                  start,
                                  decay_rate,
                                  alpha,
                                  beta,
                                  iterations,
                                  q0,
                                  q)

    results = []
    for _ in range(NUM_TRIALS):
        results.append(antAlgo.run())
        print("working")

    print(results)

    #return antAlgo.run()


