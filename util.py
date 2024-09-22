import time

import numpy as np
import math
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

from scipy.sparse.linalg import lsqr

get_x = lambda state: state[0]
get_y = lambda state: state[1]
get_theta = lambda state: state[2]
get_nanos = lambda state: state[3]
get_msg = lambda x: x[0]


class Car:
    def __init__(self, wheel_circ, L):
        self.wheel_circ = wheel_circ
        self.L = L
        self.state_hist = [[0, 0, 0, 424000000]]  # x, y, theta. history for all updates
        self.slam_hist = np.array([[0, 0, 0]])  # stores the slam state updates
        self.prev_nanos = 424000000

        self.np_runtime_data = []  # edges, dt
        self.lil_runtime_data = []
        self.csr_runtime_data = []
        self.csc_runtime_data = []

        self.distinct_cones = []
        self.distinct_cone_colors = []  # skull emoji
        self.cone_detections = []
        self.red_cones = []
        self.blue_cones = []
        self.yellow_cones = []  # each element is [avg cone x, avg cone y, num assoc points]

        self.x_vel = 0
        self.y_vel = 0
        self.theta_vel = 0

        # SLAM matrices
        self.state_indices = [0]  # slam update indices in A
        self.cone_indices = []

        self.zero = np.zeros((1000000, 1000000))
        self.A_rows = 2
        self.A_cols = 2
        self.zero[0, 0], self.zero[1, 1] = 1, 1
        self.A = self.zero[:self.A_rows, :self.A_cols]

        self.b = np.array([[0, 0]]).T

    def predict(self, nanos):  # adds new state onto state history array
        prev_state = self.state_hist[-1]
        dt = (nanos - get_nanos(prev_state)) / 10 ** 9
        new_x = get_x(prev_state) + dt * self.x_vel
        new_y = get_y(prev_state) + dt * self.y_vel
        new_theta = get_theta(prev_state) + dt * self.theta_vel

        new_state = [new_x, new_y, new_theta, nanos]
        self.state_hist.append(new_state)
        self.prev_nanos = nanos

    def addUSLAM(self):
        prev_state = self.slam_hist[-1]
        latest_state = self.state_hist[-1]
        self.zero[self.A_rows, self.state_indices[-1]] = -1  # x part
        self.zero[self.A_rows, self.A_cols] = 1  # x part
        self.zero[self.A_rows + 1, self.state_indices[-1] + 1] = -1  # y part
        self.zero[self.A_rows + 1, self.A_cols + 1] = 1  # y part

        self.state_indices.append(self.A_cols)
        self.slam_hist = np.append(self.slam_hist,
                                   [[get_x(latest_state), get_y(latest_state), get_theta(latest_state)]], axis=0)

        self.A_rows += 2
        self.A_cols += 2
        self.A = self.zero[:self.A_rows, :self.A_cols]
        self.b = np.vstack((self.b, np.array([[get_x(latest_state) - get_x(prev_state)],
                                              [get_y(latest_state) - get_y(prev_state)]])))

    def solve(self):
        # calculate optimal x vector (contains cones and states)
        A_sparse = csc_matrix(self.A)
        start = time.perf_counter()
        x = lsqr(A_sparse, self.b)[0]
        self.csc_runtime_data.append([self.A_rows, time.perf_counter() - start])

        # A_sparse = csr_matrix(self.A)
        # start = time.perf_counter()
        # x = lsqr(A_sparse, self.b)[0]
        # self.csr_runtime_data.append([self.A_rows, time.perf_counter() - start])
        #
        # start = time.perf_counter()
        # x, uno, dos, tres = np.linalg.lstsq(self.A, self.b)
        # self.np_runtime_data.append([self.A_rows, time.perf_counter() - start])
        #
        # A_sparse = lil_matrix(self.A)
        # start = time.perf_counter()
        # x = lsqr(A_sparse, self.b)[0]
        # self.lil_runtime_data.append([self.A_rows, time.perf_counter() - start])

        # update the slam_hist
        solved_states = []
        solved_cones = []
        for index in self.state_indices:
            solved_states.append([x[index], x[index + 1]])
        for index in self.cone_indices:
            solved_cones.append([x[index], x[index + 1]])
        self.slam_hist[:, :2] = solved_states
        self.distinct_cones = solved_cones
        return solved_states

    def cone(self, r, theta, color):
        x, y, angle = self.slam_hist[-1]
        cone_x = x + r * np.cos(theta + angle)
        cone_y = y + r * np.sin(theta + angle)
        cone_pos = [cone_x, cone_y]
        self.b = np.vstack((self.b, np.array([[cone_x - x],
                                              [cone_y - y]])))

        cone_number, new_cone = self.dataAssoc(cone_pos, color)
        if new_cone:
            self.zero[self.A_rows, self.state_indices[-1]] = -1
            self.zero[self.A_rows, self.A_cols] = 1
            self.zero[self.A_rows + 1, self.state_indices[-1] + 1] = -1
            self.zero[self.A_rows + 1, self.A_cols + 1] = 1
            self.cone_indices.append(self.A_cols)

            self.A_rows += 2
            self.A_cols += 2
            self.A = self.zero[:self.A_rows, :self.A_cols]
        else:
            self.zero[self.A_rows, self.state_indices[-1]], self.zero[
                self.A_rows, self.cone_indices[cone_number]] = -1, 1
            self.zero[self.A_rows + 1, self.state_indices[-1] + 1], self.zero[
                self.A_rows + 1, self.cone_indices[cone_number] + 1] = -1, 1

            self.A_rows += 2
            self.A = self.zero[:self.A_rows, :self.A_cols]

        self.cone_detections.append(cone_pos)
        if color == 0:
            self.red_cones.append(cone_pos)
        if color == 1:
            self.yellow_cones.append(cone_pos)
        if color == 2:
            self.blue_cones.append(cone_pos)

    def dataAssoc(self, cone_pos, color):
        threshold = 1.25  # cone threshold
        if len(self.distinct_cones) == 0:
            self.distinct_cones.append([get_x(cone_pos), get_y(cone_pos)])
            self.distinct_cone_colors.append(color)
            cone_number = 0
            new_cone = True
        else:
            min_dis = 999999999  # min distance to existing distinct landmark
            closest = 0  # index of the closest cone
            for i in range(len(self.distinct_cones)):
                cone = [get_x(self.distinct_cones[i]), get_y(self.distinct_cones[i])]
                if math.dist(cone, cone_pos) < min_dis:
                    min_dis = math.dist(cone, cone_pos)
                    closest = i
            if min_dis > threshold:
                self.distinct_cones.append([get_x(cone_pos), get_y(cone_pos)])
                self.distinct_cone_colors.append(color)
                cone_number = len(self.distinct_cones) - 1
                new_cone = True
            else:
                avg = np.array([self.distinct_cones[closest][0], self.distinct_cones[closest][1]])
                result_x, result_y = avg
                self.distinct_cones[closest][0], self.distinct_cones[closest][1] = result_x, result_y
                cone_number = closest
                new_cone = False
        return cone_number, new_cone

    def setVel(self, steer, bLeftRPM, bRightRPM):  # calculate new velocity values
        v = (bLeftRPM * self.wheel_circ / 60.0 + bRightRPM * self.wheel_circ / 60.0) / 2.0  # meters per second
        state = self.state_hist[-1]
        self.x_vel = v * np.cos(get_theta(state))
        self.y_vel = v * np.sin(get_theta(state))
        self.theta_vel = v * np.tan(steer) / self.L
