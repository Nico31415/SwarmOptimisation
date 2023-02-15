import matplotlib.pyplot as plt
import random
import numpy as np


# this class contains the main logic for the Ant Colony Optimization Algorithm
# (self, swarm_size, grid_size, target_position, v_min, v_max, w, c1, c2, iterations, ax)
class AntOptimisationAlgo:
    def __init__(self, num_ants,
                 grid_size,
                 food_positions,
                 start, decay_rate,
                 alpha,
                 beta,
                 iterations,
                 q0,
                 q,
                 ax
                 ):
        self.num_ants = num_ants
        self.grid_size = grid_size
        self.food_positions = food_positions
        self.start = start
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.ax = ax
        self.q0 = q0
        self.q = q

        self.heuristics = np.ones((grid_size, grid_size))
        self.compute_heuristics()
        self.pheromones = np.ones((grid_size, grid_size)) / (1 / grid_size ** 2)
        self.probabilities = np.ones((grid_size, grid_size))
        self.compute_probabilities()

    def compute_heuristics(self):
        print(self.food_positions)
        for i in range(self.grid_size):
           for j in range(self.grid_size):
               for pos in self.food_positions:
                   self.heuristics[i][j] += 1 / np.exp(1 + np.linalg.norm(np.array((i, j)) - np.array(pos)))
               # self.heuristics[i][j] = 1 / np.exp(1 + np.linalg.norm(np.array((i, j)) - np.array(self.food_positions)))
        # self.heuristics[i][j] = 1 / 1 + np.linalg.norm((i, j) - self.target_position)


        #for this experiment, we are initialising heuristic to be 0. Ant will only be guided by pheromones.
        self.heuristics = self.heuristics / self.heuristics.sum()

    def compute_probabilities(self):
        self.probabilities = np.power(self.pheromones, self.alpha) * np.power(self.heuristics, self.beta)
        #self.probabilities = np.multiply(self.probabilities, self.maze.maze)
        self.probabilities = self.probabilities / self.probabilities.sum()

    def move_ant(self, ant):
        curr_pos = ant[-1]
        (x, y) = curr_pos
        next_move = curr_pos
        curr_prob = 0

        if random.random() < self.q0:
            possible_moves = []
            curr_prob = 0
            # go to adjacent square with highest probability
            if (x - 1) in range(self.grid_size):
                if self.probabilities[x - 1][y] > curr_prob:
                    if curr_prob < self.probabilities[x - 1][y]:
                        possible_moves = [(x -1, y)]
                    elif curr_prob == self.probabilities[x-1][y]:
                        possible_moves.append((x-1, y))
                    curr_prob = self.probabilities[x - 1][y]
                    #possible_moves.append((x-1, y))
                    #next_move = (x - 1, y)
            if (x + 1) in range(self.grid_size):
                if self.probabilities[x + 1][y] > curr_prob:
                    if curr_prob < self.probabilities[x + 1][y]:
                        possible_moves = [(x +1, y)]
                    elif curr_prob == self.probabilities[x+1][y]:
                        possible_moves.append((x+1, y))
                    curr_prob = self.probabilities[x + 1][y]
                    #next_move = (x + 1, y)
            if (y - 1) in range(self.grid_size):
                if self.probabilities[x][y - 1] > curr_prob:
                    if curr_prob < self.probabilities[x][y -1]:
                        possible_moves = [(x, y-1)]
                    elif curr_prob == self.probabilities[x][y-1]:
                        possible_moves.append((x , y-1))
                    curr_prob = self.probabilities[x][y-1]
                    #next_move = (x, y - 1)
            if (y + 1) in range(self.grid_size):
                if self.probabilities[x][y + 1] > curr_prob:
                    if curr_prob < self.probabilities[x][y +1]:
                        possible_moves = [(x, y+1)]
                    elif curr_prob == self.probabilities[x][y+1]:
                        possible_moves.append((x , y+1))
                    curr_prob = self.probabilities[x][y+1]

                    #next_move = (x, y + 1)
            next_move = random.choice(possible_moves)

        if next_move == curr_pos:
            # pick one of the adjacent squares with equal probability
            choices = []
            weights = []
            if (x - 1) in range(self.grid_size) and self.probabilities[x - 1][y] != 0:
                choices.append((x - 1, y))
                weights.append(self.probabilities[x - 1][y])
            if (x + 1) in range(self.grid_size) and self.probabilities[x + 1][y] != 0:
                choices.append((x + 1, y))
                weights.append(self.probabilities[x + 1][y])
            if (y - 1) in range(self.grid_size) and self.probabilities[x][y - 1] != 0:
                choices.append((x, y - 1))
                weights.append(self.probabilities[x][y - 1])
            if (y + 1) in range(self.grid_size) and self.probabilities[x][y + 1] != 0:
                choices.append((x, y + 1))
                weights.append(self.probabilities[x][y + 1])
            next_move = random.choices(choices, k=1)[0]

        ant.append(next_move)
        return ant

    def update_pheromones(self, ants):
        self.pheromones = self.pheromones * self.decay_rate
        for ant in ants:
            for (i, j) in ant:
                self.pheromones[i][j] += self.q * 1 / len(ant)

        self.pheromones = self.pheromones / self.pheromones.sum()

    def run(self):
        self.ax.imshow(np.log(self.probabilities), cmap='hot')
        plt.pause(5)
        # )
        print("running")
        dead_ants = 0
        for i in range(self.iterations):
            ants = [[self.start] for _ in range(self.num_ants - dead_ants)]
            print("ants: ", ants)
            for ant in ants:
                print("ant first:, ", ant)
                count = 0
                while ant[-1] not in self.food_positions:
                    count += 1
                    if count > 10000:
                        dead_ants += 1
                        break
                    ant = self.move_ant(ant)
                    #print('ant: ', ant)
                    print('food pos: ', self.food_positions)
            print('ants: ', ants)
            print('food pos: ', self.food_positions)
            self.update_pheromones(ants)
            self.compute_probabilities()

            self.ax.clear()

            for i, ant in enumerate(ants):
                antx = [p[1] for p in ant]
                anty = [p[0] for p in ant]

                label = 'ant' + str(i)
                self.ax.imshow(np.log(self.probabilities), cmap='hot', interpolation='nearest')
                self.ax.plot(antx, anty, color=np.random.rand(3, ), label=label)
                self.ax.legend()
                for pos in self.food_positions:
                    self.ax.scatter(pos[1], pos[0], color='red', marker='x', s=100)

            plt.pause(1)
        print(ants)
        # print('almost done')
        # path_lengths = [len(ant) for ant in ants]
        # best_ant = ants[np.argmin(path_lengths)]
        # print(best_ant)
        # antx = [p[1] for p in best_ant]
        # anty = [p[0] for p in best_ant]
        # self.ax.clear()
        # self.ax.imshow(np.log(self.probabilities), cmap='hot', interpolation='nearest')
        # self.ax.plot(antx, anty, color=np.random.rand(3, ), label='best ant')
        # self.ax.legend()
        # self.ax.scatter(self.food_position[0], self.food_position[1], color='red', marker='x', s=100)

        plt.savefig("output.png")
        plt.show()

        self.ax.imshow(np.log(self.probabilities.T + 10 ** (- 500)), cmap='hot', interpolation='nearest')
        self.ax.imshow(self.heuristics, cmap='hot', interpolation='nearest')
        self.ax.imshow(self.pheromones, cmap='hot', interpolation='nearest')
        plt.show()
        return ants
