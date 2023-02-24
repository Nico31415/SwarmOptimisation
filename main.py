from foodMaze import Maze
from randomWalk import RandomWalk
from antOptimisation import AntOptimisationAlgo
import matplotlib.pyplot as plt
import numpy as np

decay_rate = 0.5
alpha = 1
beta = 0
q0 = 1
q = 20
swarm_size = 1
iterations = 1000

num_nodes = 7
# num_food = iterations * swarm_size
num_pos = 1
grid_size = 2 * num_nodes - 1
start = (grid_size // 2, grid_size // 2)







# walk.run()
# antAlgoPheromones.run()
# antAlgoHeuristics.run()
# antAlgoPheromonesHeuristics.run()

pheromone_times = []
heuristic_times = []
randomwalk_times = []

graph_iterations = 1

maze = Maze(num_nodes=num_nodes, num_food=iterations * swarm_size, num_pos=2)
maze.maze[start[0]][start[1]] = 1
antAlgoPheromones = AntOptimisationAlgo(swarm_size, grid_size, maze, start, decay_rate, 0, beta, iterations, q0, q)
walk = RandomWalk(swarm_size, iterations, maze, start=start, grid_size=grid_size)

print(walk.run())
antAlgoPheromones.run()



# for i in range(graph_iterations, graph_iterations + 1):
#
#     random_time = 0
#     pheromone_time = 0
#
#     for k in range(1):
#         maze = Maze(num_nodes=num_nodes, num_food=iterations * swarm_size, num_pos=i)
#         maze.maze[start[0]][start[1]] = 1
#         # for pos in maze.get_food_pos():
#         #     plt.scatter(pos[0], pos[1])
#         # maze.show_maze(ax)
#
#         #antAlgo = AntOptimisationAlgo(swarm_size, grid_size, maze, start, decay_rate, 10, 0, iterations, q0, q)
#
#         antAlgoPheromones = AntOptimisationAlgo(swarm_size, grid_size, maze, start, decay_rate, alpha, beta, iterations, q0, q)
#         #antAlgoHeuristics = AntOptimisationAlgo(swarm_size, grid_size, maze, start, decay_rate, 0, 1, iterations, q0, q)
#         #walk = RandomWalk(swarm_size, iterations, maze, start=start, grid_size=grid_size)
#
#         #random_time += walk.run() / 10
#         pheromone_time += antAlgoPheromones.run() / 10
#         #randomwalk_times.append(walk.run())
#         pheromone_times.append(antAlgoPheromones.run())
#
#     #randomwalk_times.append(random_time)
#     pheromone_times.append(pheromone_time)
#     # heuristic_times.append(antAlgoHeuristics.run())

#plt.plot(pheromone_times, label="pheromone")
#plt.plot(heuristic_times)
#plt.plot(randomwalk_times, label="random")
#plt.legend()
#plt.show()

# # # See PyCharm help at https://www.jetbrains.com/help/pycharm/
