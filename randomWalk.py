import random
import numpy as np
import matplotlib.pyplot as plt

class RandomWalk:

    def __init__(self,
                 num_ants,
                 iterations,
                 food_maze,
                 start,
                 grid_size):
        self.grid_size = grid_size
        self.start = start
        self.num_ants = num_ants
        self.iterations = iterations
        self.food_maze = food_maze

    def move_ant(self, ant):
        curr_pos = ant[-1]
        (x, y) = curr_pos

        possible_moves = []

        if (x - 1) in range(self.grid_size) and self.food_maze.maze[y][x- 1] != 0:
            possible_moves.append((x - 1, y))
        if (x + 1) in range(self.grid_size) and self.food_maze.maze[y][x+ 1] != 0:
            possible_moves.append((x + 1, y))
        if (y - 1) in range(self.grid_size) and self.food_maze.maze[y - 1][x] != 0:
            possible_moves.append((x, y - 1))
        if (y + 1) in range(self.grid_size) and self.food_maze.maze[y + 1][x] != 0:
            possible_moves.append((x, y + 1))

        next_move = random.choice(possible_moves)

        ant.append(next_move)

        return ant

    def run(self):
        all_ants = []
        for i in range(self.iterations):
            ants = [[self.start] for _ in range(self.num_ants)]

            for ant in ants:
                while ant[-1] not in self.food_maze.get_food_pos():
                    ant = self.move_ant(ant)
                    #print(ant)
            all_ants = all_ants + ants
        lengths = list(map(lambda x : len(x), all_ants))

        print("Mean length random walk: ", np.mean(lengths))

        #plt.hist(lengths)
        #plt.show()
        return np.mean(lengths)

