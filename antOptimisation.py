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

        self.MAX_MOVES_MAP = {3: 11.11, 5: 45.07, 7: 111.80, 9: 218.31, 11: 369.02,
                              13: 567.66, 15: 817.28, 17: 1120.52, 19: 1479.58}

    def compute_heuristics(self):

        self.heuristics = np.ones((self.grid_size, self.grid_size))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for pos in self.food_maze.get_food_pos():
                    self.heuristics[j][i] += 1 / np.exp(1 + np.sum(abs(np.array((i, j)) - np.array(pos))))
                    # self.heuristics[j][i] += 1 / 1 + np.linalg.norm(np.array((i, j)) - np.array(pos))

        # self.heuristics = self.heuristics / self.heuristics.sum()

    def compute_probabilities(self):
        self.probabilities = np.power(self.pheromones, self.alpha) * np.power(self.heuristics, self.beta)
        self.probabilities = np.multiply(self.probabilities, self.food_maze.maze)
        # self.probabilities = self.probabilities / self.probabilities.sum()

    def show_heuristics(self):
        fig, ax = plt.subplots()
        ax.imshow(self.heuristics, cmap='hot')
        plt.show()

    def show_pheromones(self):
        fig, ax = plt.subplots()
        ax.imshow(self.pheromones, cmap='hot')
        plt.show()

    def show_probabilities(self):
        fig, ax = plt.subplots()
        ax.imshow(self.probabilities, cmap='hot')
        plt.show()

    def update_pheromones(self, ants):
        self.pheromones = self.pheromones * self.decay_rate

        ants_to_update = list(filter(lambda x : len(x) > 0, ants))
        if len(ants_to_update) > 0:
            ants_to_update = [min(ants_to_update, key=len)]

        for ant in ants_to_update:
            for (i, j) in set(ant):
                self.pheromones[j][i] += self.q * 1 / 1 + len(set(ant))
        # self.pheromones = self.pheromones / self.pheromones.sum()

    def move_ant(self, ant):
        curr_pos = ant[-1]
        (x, y) = curr_pos
        next_move = curr_pos
        curr_prob = 0
        # moves = []

        curr_prob_all = 0
        curr_prob_unseen = 0
        moves = []
        good_moves = []
        possible_moves = []
        possible_good_moves = []
        weights = []
        good_weights = []
        if random.random() < self.q0:

            if (x - 1) in range(self.grid_size) and self.food_maze.maze[y][x - 1] != 0:
                possible_moves.append((x - 1, y))
                weights.append(self.probabilities[y][x - 1])
                if (x - 1, y) not in ant:
                    possible_good_moves.append((x - 1, y))
                    good_weights.append(self.probabilities[y][x - 1])
            if (x + 1) in range(self.grid_size) and self.food_maze.maze[y][x + 1] != 0:
                possible_moves.append((x + 1, y))
                weights.append(self.probabilities[y][x + 1])
                if (x + 1, y) not in ant:
                    possible_good_moves.append((x + 1, y))
                    good_weights.append(self.probabilities[y][x + 1])
            if (y - 1) in range(self.grid_size) and self.food_maze.maze[y - 1][x] != 0:
                possible_moves.append((x, y - 1))
                weights.append(self.probabilities[y - 1][x])
                if (x, y - 1) not in ant:
                    possible_good_moves.append((x, y - 1))
                    good_weights.append(self.probabilities[y - 1][x])
            if (y + 1) in range(self.grid_size) and self.food_maze.maze[y + 1][x] != 0:
                possible_moves.append((x, y + 1))
                weights.append(self.probabilities[y + 1][x])
                if (x, y + 1) not in ant:
                    possible_good_moves.append((x, y + 1))
                    good_weights.append(self.probabilities[y + 1][x])

            weights = weights / np.sum(weights)
            #print(weights)
            #print(possible_moves)
            if len(possible_good_moves) > 0:
                next_move = random.choices(possible_good_moves, good_weights, k=1)[0]
            else:
                next_move = random.choices(possible_moves, weights, k=1)[0]

        ant.append(next_move)
        return ant


    def run(self):
        # an ant can do a max number of moves before it dies.
        # Is the colony able to survive in a certain environment?
        dead_ants = 0
        # max_moves = self.MAX_MOVES_MAP[self.grid_size] * 2
        max_moves = 10 ** 10
        all_ants = []
        # print(self.food_maze.maze)
        #plt.imshow(self.probabilities)
        changed = False
        for pos in self.food_maze.get_food_pos():
            plt.scatter(pos[0], pos[1])
        for i in range(self.iterations):
            print('iteration: ', i)
            ants = [[self.start] for _ in range(self.num_ants - dead_ants)]
            for pos in self.food_maze.get_food_pos():
                plt.scatter(pos[0], pos[1])

            #plt.imshow(self.probabilities / np.sum(self.probabilities), cmap="hot")
            #plt.show()
            for ant in ants:
                #print(self.food_maze.get_food_pos())
                while ant[-1] not in self.food_maze.get_food_pos():

                    #print(ant)
                    ant = self.move_ant(ant)
                #self.food_maze.update_foods(ant[-1])
                if self.food_maze.update_foods(ant[-1]):
                    print("changing pheromones")
                    self.pheromones = np.ones((self.grid_size, self.grid_size))
                    self.compute_heuristics()
                    self.compute_probabilities()
                    #print(self.probabilities)

                print(ant[-1])
                print("here")
                # if len(self.food_maze.get_food_pos()) == 0:
                #     print("bear")
                #     mean_length = np.median((list(map(lambda x: len(x), ants))))
                #     lengths = list(map(lambda x: len(x), all_ants))
                #     print("mean length: ", mean_length)
                #
                #     return mean_length
            all_ants = all_ants + ants
            print(self.food_maze.food_pos_and_amounts)

            ants_to_update = [ant if ant[-1] in self.food_maze.get_food_pos() else [] for ant in ants]
            self.update_pheromones(ants_to_update)
            self.compute_heuristics()
            self.compute_probabilities()

            if len(self.food_maze.get_food_pos()) == 0:
                print("moose")
                lengths = list(map(lambda x: len(x), all_ants))

                mean_length = np.median((list(map(lambda x: len(x), all_ants))))
                print("mean length: ", mean_length)
                return mean_length
                return "Level completed: " + str(mean_length)
                print("Level completed")

        lengths = list(map(lambda x: len(x), all_ants))
        return np.median((list(map(lambda x: len(x), all_ants))))
        # plt.hist(lengths)
        # plt.show()
