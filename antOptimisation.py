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
                    self.heuristics[j][i] += 1 / (1 + np.sum(abs(np.array((i, j)) - np.array(pos))))
                    # self.heuristics[j][i] += 1 / 1 + np.linalg.norm(np.array((i, j)) - np.array(pos))

        # self.heuristics = self.heuristics / self.heuristics.sum()

    def compute_probabilities(self):
        self.probabilities = np.power(self.pheromones, self.alpha) * np.power(self.heuristics, self.beta)
        self.probabilities = np.multiply(self.probabilities, self.food_maze.maze)
        #self.probabilities = self.probabilities / self.probabilities.sum()

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
        for ant in ants:
            for (i, j) in set(ant):
                self.pheromones[j][i] += self.q * 1 / (1 + len(set(ant)))
        # self.pheromones = self.pheromones / self.pheromones.sum()

    def move_ant(self, ant):
        curr_pos = ant[-1]
        (x, y) = curr_pos
        next_move = curr_pos
        curr_prob = 0
        moves = []

        if random.random() < self.q0:

            if (x - 1) in range(self.grid_size):
                if self.probabilities[y][x - 1] >= curr_prob and self.food_maze.maze[y][x-1] != 0:
                    moves.append((x - 1, y))
                    curr_prob = self.probabilities[y][x - 1]
            if (x + 1) in range(self.grid_size):
                if self.probabilities[y][x + 1] >= curr_prob and self.food_maze.maze[y][x+1] != 0:
                    moves.append((x + 1, y))
                    curr_prob = self.probabilities[y][x + 1]
            if (y - 1) in range(self.grid_size):
                if self.probabilities[y - 1][x] >= curr_prob and self.food_maze.maze[y-1][x] != 0:
                    moves.append((x, y - 1))
                    curr_prob = self.probabilities[y - 1][x]
            if (y + 1) in range(self.grid_size):
                if self.probabilities[y + 1][x] >= curr_prob and self.food_maze.maze[y+1][x] != 0:
                    moves.append((x, y + 1))
                    curr_prob = self.probabilities[y + 1][x]

            good_moves = list(filter(lambda x: x not in ant, moves))


            if (self.pheromones != np.ones((self.grid_size, self.grid_size))).all():
                if len(good_moves) > 0:
                    next_move = random.choice(good_moves)
                elif len(moves) > 0:
                    next_move = random.choice(moves)
            else:
                if len(moves) > 0:
                    next_move = random.choice(moves)
            #
            # if len(moves) > 0:
            #     next_move = random.choice(moves)

        if next_move == curr_pos:
            #print("in second section")
            # pick one of the adjacent squares with equal probability
            choices = []
            weights = []
            if (x - 1) in range(self.grid_size) and self.food_maze.maze[y][x - 1] != 0:
                choices.append((x - 1, y))
                weights.append(self.probabilities[y][x - 1])
            if (x + 1) in range(self.grid_size) and self.food_maze.maze[y][x + 1] != 0:
                choices.append((x + 1, y))
                weights.append(self.probabilities[y][x + 1])
            if (y - 1) in range(self.grid_size) and self.food_maze.maze[y - 1][x] != 0:
                choices.append((x, y - 1))
                weights.append(self.probabilities[y - 1][x])
            if (y + 1) in range(self.grid_size) and self.food_maze.maze[y + 1][x] != 0:
                choices.append((x, y + 1))
                weights.append(self.probabilities[y + 1][x])
            if len(choices) > 0:
                next_move = random.choices(choices, k=1)[0]

        (a, b) = next_move
        #print("next move: ", next_move)
        #print("possible moves: ", moves)
        #print("probability: ", self.probabilities[b][a])
        #print("maze: ", self.food_maze.maze[b][a])
        ant.append(next_move)
        return ant

    #

    def run(self):
        # an ant can do a max number of moves before it dies.
        # Is the colony able to survive in a certain environment?
        dead_ants = 0
        # max_moves = self.MAX_MOVES_MAP[self.grid_size] * 2
        max_moves = 10 ** 10
        fig, ax = plt.subplots()
        all_ants = []
        #print(self.food_maze.maze)
        plt.imshow(self.food_maze.maze)
        for pos in self.food_maze.get_food_pos():
            plt.scatter(pos[0], pos[1])
        #plt.show()
        for i in range(self.iterations):

            ants = [[self.start] for _ in range(self.num_ants - dead_ants)]
            no_more_food = False
            for ant in ants:
                plt.imshow(self.probabilities / np.sum(self.probabilities), cmap="hot")
                for pos in self.food_maze.get_food_pos():
                    plt.scatter(pos[0], pos[1])


                while ant[-1] not in self.food_maze.get_food_pos():
                    #print(ant)
                    # print(ant)
                    if len(ant) > max_moves:
                        dead_ants += 1
                        # return "Couldnt complete level"
                        break
                    ant = self.move_ant(ant)
                    # if ant[-1] in self.food_maze.food_pos:
                    #     self.food_maze.food_pos_and_amounts[ant[-1]] -= 1
                    #     if
                #print(len(ant), ant[-1])
                # we should update the pheromone matrix and probability
                # matrix once each ant reaches the goal instead of
                # aiting for all of them to do this

                # self.compute_probabilities()
                if self.food_maze.update_foods(ant[-1]):
                    self.pheromones = np.ones((self.grid_size, self.grid_size))
                    #print("removed: ", ant[-1])
                else:
                    self.update_pheromones([ant])


                self.compute_heuristics()
                self.compute_probabilities()

                # self.show_heuristics()
                # self.show_pheromones()
                # fig, ax = plt.subplots()

                # ax.imshow(np.log(self.probabilities), cmap='hot')
                #print(self.probabilities)
                # self.show_probabilities()
                # for (pos, amount) in self.food_maze.food_pos_and_amounts.items():
                #    ax.scatter(pos[0], pos[1], color='green', marker='o', s=np.log(amount)+1)
                # plt.show()
                # print(ant[-1])
                # print('Path length: ', len(ant))

                # if self.food_maze.update_foods(ant[-1]):
                #     plt.imshow(self.probabilities)
                #
                #     plt.show()
                #     self.pheromones = np.ones((self.grid_size, self.grid_size))
                #     self.compute_probabilities()
                #
                #     print("removed")
                if len(self.food_maze.get_food_pos()) == 0:
                    mean_length = np.median((list(map(lambda x: len(x), ants))))
                    # print(max_length)
                    lengths = list(map(lambda x: len(x), all_ants))

                    print("mean length: ", mean_length)
                    #plt.hist(lengths)
                    #plt.show()
                    return mean_length
                    # print("Level completed")
                    # break
           # plt.show()
            all_ants = all_ants + ants
            if len(ants) == 0:
                return "Couldnt complete level"
            #   print("Level not completed")
            #   break
            # print(self.food_maze.food_pos)
            # print(ants)
            # print(self.food_maze.maze)
            # self.update_pheromones(ants)
            # self.compute_probabilities()
            # self.compute_heuristics()
            # print(self.probabilities)

            # ax.clear()

            # for i, ant in enumerate(ants):
            #     antx = [p[0] for p in ant]
            #     anty = [p[1] for p in ant]
            #
            #     label = 'ant' + str(i)
            #     #ax.imshow(self.probabilities, cmap='hot', interpolation='nearest')
            #     self.food_maze.show_maze(ax)
            #     ax.plot(antx, anty, color=np.random.rand(3, ), label=label)
            #     ax.legend()
            #     ax.scatter(self.start[0], self.start[1], color='red', marker='x', s=30)
            #     for (pos, amount) in self.food_maze.food_pos_and_amounts.items():
            #         ax.scatter(pos[0], pos[1], color='green', marker='o', s=np.log(amount)+1)
            #
            # plt.pause(1)
            if len(self.food_maze.get_food_pos()) == 0:
                # max_length = max(list(map(lambda x: len(x), ants)))
                # print(max_length)
                lengths = list(map(lambda x: len(x), all_ants))

                #plt.hist(lengths)
                #plt.show()
                mean_length = np.median((list(map(lambda x: len(x), all_ants))))
                print("mean length: ", mean_length)
                return mean_length
                return "Level completed: " + str(mean_length)
                print("Level completed")
                break

        lengths = list(map(lambda x: len(x), all_ants))

        #plt.hist(lengths)
        #plt.show()
