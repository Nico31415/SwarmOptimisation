import random

import matplotlib.pyplot as plt
import numpy as np


class Maze:
    def __init__(self, num_nodes, num_food, block_size, num_pos):
        assert num_food >= num_pos
        self.block_size = block_size
        self.grid_size = 2 * num_nodes - 1
        self.maze = np.ones((self.grid_size, self.grid_size))
        #self.maze = np.zeros((self.grid_size, self.grid_size))
        self.num_food = num_food
        self.num_pos = num_pos

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if j % 2 == 0:
                    if i % 2 != 0:
                        self.maze[i][j] = 0
                else:
                    self.maze[i][j] = 0
        #print(self.maze)

        self.validate_maze()
        print(self.maze)
        self.blockify(self.block_size)
        print(self.maze)
        self.grid_size = len(self.maze)

        self.empty_positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.maze[j][i] == 1:
                    self.empty_positions.append((i, j))

        self.food_pos_and_amounts = dict()
        self.add_food()

    def update_foods(self, pos):
        try:
            self.food_pos_and_amounts[pos] -= 1
            if self.food_pos_and_amounts[pos] == 0:
                self.food_pos_and_amounts.pop(pos)
                print("one less focal point: ", pos)
                return True
            else:
                return False
        except:
            return False

    def get_food_pos(self):
        #print("In maze code: ", set(self.food_pos_and_amounts.keys()))
        #print("Dict: ", self.food_pos_and_amounts)
        return set(self.food_pos_and_amounts.keys())

    def add_food(self):

        # density = points / area
        # food ratio = food amount / number of ants

        # so basically create n points using density,
        # then with food amount remaining, add a random amount to each point
        # food is distributed between num_food points. Each point has at least 1

        # takes in density of maze as parameter, means
        possible_pos = random.choices(self.empty_positions, k=self.num_pos)


        for pos in possible_pos:
            self.food_pos_and_amounts[pos] = 1
        food_left = self.num_food - self.num_pos
        while food_left > 0:
            amount = random.randint(1, food_left)
            position = random.choice(possible_pos)
            self.food_pos_and_amounts[position] += amount
            food_left -= amount

        self.food_pos = self.food_pos_and_amounts.keys()


        for pos in possible_pos:
            self.food_pos_and_amounts[pos] = self.num_food // self.num_pos
        print(self.food_pos_and_amounts)

    def validate_maze(self):
        start = (0, 0)
        visited = set(start)
        stack = [start]
        while stack:
            choices = []
            (x, y) = stack[-1]
            if (x + 2) in range(self.grid_size) and (x + 2, y) not in visited:
                choices.append((x + 2, y))
            if (x - 2) in range(self.grid_size) and (x - 2, y) not in visited:
                choices.append((x - 2, y))
            if (y + 2) in range(self.grid_size) and (x, y + 2) not in visited:
                choices.append((x, y + 2))
            if (y - 2) in range(self.grid_size) and (x, y - 2) not in visited:
                choices.append((x, y - 2))
            #print(choices)
            if len(choices) > 0:
                (x2, y2) = random.choice(choices)
                visited.add((x2, y2))
                xwall, ywall = (int((x2 + x) / 2), int((y2 + y) / 2))
                self.maze[xwall][ywall] = 1
                stack.append((x2, y2))
            else:
                stack = stack[:-1]

    def blockify(self, n):
        m = len(self.maze)
        n_cols = len(self.maze[0])
        new_size = n_cols + (n_cols // 2 + 1) * (n - 1)
        result = [[0] * (new_size) for _ in range(new_size)]
        print(result)
        for i in range(m):
            for j in range(n_cols):
                if self.maze[i][j] == 1:
                    if i % 2 == 1:
                        if j % 2 == 0:
                            print("here")
                            start_i = (i // 2) * (n + 1) + n
                            start_j = (j // 2) * (n + 1)
                            for p in range(1):
                                for q in range(n):
                                    result[start_i + p][start_j + q] = 1
                    else:

                        if j % 2 == 0:
                            start_i = (i // 2) * (n + 1)
                            start_j = (j // 2) * (n + 1)
                            for p in range(n):
                                for q in range(n):
                                    result[start_i + p][start_j + q] = 1
                        else:
                            start_i = (i // 2) * (n + 1)
                            start_j = (j // 2) * (n + 1) + n
                            for p in range(n):
                                result[start_i + p][start_j] = 1

        self.maze = result
    def show_maze(self):
        fig, ax = plt.subplots()
        ax.imshow(self.maze, cmap='binary_r')

        for (pos, amount) in self.food_pos_and_amounts.items():
            ax.scatter(pos[0], pos[1], color='green', marker='o', s=amount)
        plt.show()

    def show_maze(self, ax):
        #print(self.food_pos_and_amounts)
        ax.imshow(self.maze, cmap='binary_r')

        for (pos, amount) in self.food_pos_and_amounts.items():
            ax.scatter(pos[0], pos[1], color='green', marker='o', s=amount)
        # plt.show()
