import numpy as np
import matplotlib.pyplot as plt
import random


class AntOptimisationAlgo:
    def __init__(self,
                 num_ants,
                 grid_size,
                 food_maze,
                 start,
                 decay_rate,
                 alpha,
                 beta,
                 iterations,
                 q0,
                 q
                 ):

        self.num_ants = num_ants
        self.grid_size = grid_size
        self.start = start
        self.decay_rate = decay_rate
        self.food_maze = food_maze
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.q0 = q0
        self.q = q

        self.heuristics = np.zeros((grid_size, grid_size))
        self.compute_heuristics()
        self.pheromones = np.ones((grid_size, grid_size)) / (1 / grid_size ** 2)
        self.probabilities = np.ones((grid_size, grid_size))
        self.compute_probabilities()

    def compute_heuristics(self):

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for pos in self.food_maze.get_food_pos():
                    self.heuristics[j][i] += 1 / np.exp(1 + np.linalg.norm(np.array((i, j)) - np.array(pos)))

        self.heuristics = self.heuristics / self.heuristics.sum()

    def compute_probabilities(self):
        self.probabilities = np.power(self.pheromones, self.alpha) * np.power(self.heuristics, self.beta)
        self.probabilities = np.multiply(self.probabilities, self.food_maze.maze)
        self.probabilities = self.probabilities / self.probabilities.sum()

    def show_heuristics(self):
        fig, ax = plt.subplots()
        ax.imshow(self.heuristics, cmap='hot')
        plt.show()

    def show_probabilities(self):
        fig, ax = plt.subplots()
        ax.imshow(self.probabilities, cmap='hot')
        plt.show()

    def update_pheromones(self, ants):
        self.pheromones = self.pheromones * self.decay_rate
        for ant in ants:
            for (i, j) in ant:
                self.pheromones[j][i] += self.q * 1 / len(ant)
        self.pheromones = self.pheromones / self.pheromones.sum()

    def move_ant(self, ant):
        curr_pos = ant[-1]
        (x, y) = curr_pos
        next_move = curr_pos
        curr_prob = 0

        if random.random() < self.q0:
            possible_moves = []
            curr_prob = 0

            if (x - 1) in range(self.grid_size):
                if self.probabilities[y][x - 1] > curr_prob:
                    next_move = (x - 1, y)
                    curr_prob = self.probabilities[y][x - 1]

            if (x + 1) in range(self.grid_size):
                if self.probabilities[y][x + 1] > curr_prob:
                    next_move = (x + 1, y)
                    curr_prob = self.probabilities[y][x + 1]

            if (y + 1) in range(self.grid_size):
                if self.probabilities[y + 1][x] > curr_prob:
                    next_move = (x, y + 1)
                    curr_prob = self.probabilities[y + 1][x]

            if (y - 1) in range(self.grid_size):
                if self.probabilities[y - 1][x] > curr_prob:
                    next_move = (x, y - 1)
                    curr_prob = self.probabilities[y - 1][x]

        if next_move == curr_pos:
            # pick one of the adjacent squares with equal probability
            choices = []
            weights = []
            if (x - 1) in range(self.grid_size) and self.food_maze.maze[y][x -1] != 0:
                choices.append((x - 1, y))
                weights.append(self.probabilities[y][x - 1])
            if (x + 1) in range(self.grid_size) and self.food_maze.maze[y][x + 1] != 0:
                choices.append((x + 1, y))
                weights.append(self.probabilities[y][x + 1])
            if (y - 1) in range(self.grid_size) and self.food_maze.maze[y-1][x] != 0:
                choices.append((x, y - 1))
                weights.append(self.probabilities[y - 1][x])
            if (y + 1) in range(self.grid_size) and self.food_maze.maze[y+1][x] != 0:
                choices.append((x, y + 1))
                weights.append(self.probabilities[y + 1][x])
            if len(choices) > 0:
                next_move = random.choices(choices, k=1)[0]

        (a, b) = next_move
        print(next_move)
        print(self.probabilities[b][a])
        ant.append(next_move)
        return ant

    #

    def run(self):
        # an ant can do a max number of moves before it dies.
        # Is the colony able to survive in a certain environment?
        dead_ants = 0
        max_moves = 10**10
        fig, ax = plt.subplots()
        for i in range(self.iterations):
            ants = [[self.start] for _ in range(self.num_ants - dead_ants)]
            no_more_food = False
            for ant in ants:
                while ant[-1] not in self.food_maze.get_food_pos():

                    if len(ant) > max_moves:
                        dead_ants += 1
                        break
                    ant = self.move_ant(ant)
                    # if ant[-1] in self.food_maze.food_pos:
                    #     self.food_maze.food_pos_and_amounts[ant[-1]] -= 1
                    #     if
                self.food_maze.update_foods(ant[-1])
                if len(self.food_maze.get_food_pos()) == 0:
                    print("Level completed")
                    break
            if len(ants) == 0:
                print("Level not completed")
                break
            # print(self.food_maze.food_pos)
            # print(ants)
            # print(self.food_maze.maze)
            self.update_pheromones(ants)
            self.compute_probabilities()
            self.compute_heuristics()
            print(self.probabilities)



            ax.clear()

            for i, ant in enumerate(ants):
                antx = [p[0] for p in ant]
                anty = [p[1] for p in ant]

                label = 'ant' + str(i)
                #ax.imshow(self.probabilities, cmap='hot', interpolation='nearest')
                self.food_maze.show_maze(ax)
                ax.plot(antx, anty, color=np.random.rand(3, ), label=label)
                ax.legend()
                ax.scatter(self.start[0], self.start[1], color='red', marker='x', s=30)
                for (pos, amount) in self.food_maze.food_pos_and_amounts.items():
                    ax.scatter(pos[0], pos[1], color='green', marker='o', s=np.log(amount)+1)

            plt.pause(1)
            if len(self.food_maze.get_food_pos()) == 0:
                print("Level completed")
                break
        plt.show()
