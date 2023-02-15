import random

import numpy as np
import matplotlib.pyplot as plt


class FoodMap:
    def __init__(self, grid_size, food_ratio):
        self.grid_size = grid_size
        self.food_ratio = food_ratio

        self.num_food = int(food_ratio * grid_size)

        self.food_grid = np.zeros((grid_size, grid_size))
        for _ in range(self.num_food):
            self.food_grid[random.randint(0, self.grid_size - 1)][random.randint(0, self.grid_size- 1)] = 1

    def show_food(self):
        plt.imshow(self.food_grid, cmap='binary')
        plt.show()



foodMap = FoodMap(100, 0.3)
foodMap.show_food()