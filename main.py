# # # This is a sample Python script.
# #
# # # Press ⌃R to execute it or replace it with your code.
# # # Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# #
# import random
#
# import numpy as np
# import matplotlib.pyplot as plt
# import swarmAlgos
# from antColonyOptimisation import AntOptimisationAlgo
# from maze import Maze
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     grid_size = 10
#     food_positions = [(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))]
#     swarm_size = 10
#     iterations = 10
#
#     fig, ax = plt.subplots()
#     start = np.array([0, 0])
#     decay_rate = 0.8
#     alpha = 10
#     beta = 20
#     q0 = 0.6
#     q = 3
#
#     antAlgorithm = AntOptimisationAlgo(swarm_size, grid_size, food_positions, (0, 0), decay_rate, alpha,
#                                        beta,
#                                        iterations, q0, q, ax)
#
#     antAlgorithm.run()
#
#     # grid_size = 50
#     # target_position = np.random.random_integers(grid_size, size=2)
#     # target_position = np.array([grid_size - 1, grid_size - 1])
#     #
#     # swarm_size = 10
#     # iterations = 50
#     #
#     # iterations_pso = 100
#     #
#     # c1 = 2.05
#     # c2 = 2.05
#     # w = 1
#     #
#     # v_min = -5 / 1000 * grid_size
#     # v_max = 5 / 1000 * grid_size
#     #
#     # fig, ax = plt.subplots()
#     # # fig2, ax2 = plt.subplots()
#     #
#     # start = [0, 0]
#     # decay_rate = 0.8
#     # alpha = 3
#     # beta = 20
#     # q0 = 0.6
#     # q = 3
#     #
#     #
#     # maze = Maze(grid_size, 0.5, start, target_position)
#     # pso = swarmAlgos.ParticleSwarmAlgoFast(swarm_size, grid_size, target_position, v_min, v_max, w, c1, c2, iterations_pso,
#     #                                        ax)
#     #
#     #
#     # print("hello")
#     # antAlgorithm = swarmAlgos.AntOptimisationAlgo(swarm_size, grid_size, target_position, (0, 0), decay_rate, alpha,
#     #                                               beta,
#     #                                               iterations, q0, q, ax, maze)
#     #
#     # antAlgorithm.run()
#     # ax.clear()
#     # pso.run()
#
import matplotlib.pyplot as plt

from foodMaze import Maze
from antOptimisation import AntOptimisationAlgo

decay_rate = 0.8
alpha = 5
beta = 1
q0 = 0.7
q = 3
swarm_size = 4
iterations = 10

num_nodes = 4
num_food = iterations * swarm_size
num_pos = 3
grid_size = 2 * num_nodes - 1
start = (grid_size // 2, grid_size // 2)

maze = Maze(num_nodes, num_food, num_pos)
maze.maze[start[1]][start[0]] = 1

fig, ax = plt.subplots()
maze.show_maze(ax)
plt.show()
#maze.show_maze(a)
print(maze.get_food_pos())
antAlgo = AntOptimisationAlgo(swarm_size, grid_size, maze, start, decay_rate, alpha, beta, iterations, q0, q)

# antAlgo.show_probabilities()

antAlgo.run()

# # # See PyCharm help at https://www.jetbrains.com/help/pycharm/
