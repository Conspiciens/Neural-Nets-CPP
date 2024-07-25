import numpy as np  
import matplotlib.pyplot as plt
import random
import math

# Working on RRT Alogrithim 
# https://github.com/nimRobotics/RRT/blob/master/rrt.py
class Simulator: 
    def __init__(self): 
        self.start_pos = [0, 0]
        self.end_pos = [99, 99]
        self.space = np.zeros((100, 100))
        self.total_steps = 10
        self.path = [start_pos]
        self.isPathFinished = False
    
    def start_path_finding(self): 
        while self.isPathFinished == False:
            rand_space_num = random_conf() 
            q_near = self.nearest_vertex(rand_space_num)

    def random_conf(self): 
        while (check_collision(random.randint(0, 100), random.randint(0, 100))): 
            pass 

    def check_collision(self, x_pos, y_pos) -> bool: 
        if self.space[x_pos][y_pos] == 1: 
            return True 
        return False 

    def update_path(self): 
        pass

    def nearest_vertex(self, rand_space_num): 
        ''' 
            Find the distance between two points
        ''' 
        x_pos = rand_space_num[0] - self.path[-1][0] 
        y_pos = rand_space_num[1] - self.path[-1][1]

        x_pos *= x_pos 
        y_pos *= y_pos 

        return sqrt(x_pos - y_pos)
    
    def dist_and_angle(self, x_pos, y_pos)
        pass 
        

    def init(self): 
        # Randomly assign values every fifth value
        for i in range(0, 100, 5): 
            for j in range(0, 100, 5): 
                self.space[i][j] = np.random.rand()
        

        plt.plot(self.space, marker=".", linestyle="none")
        plt.show()

    
sim = Simulator() 
sim.init()