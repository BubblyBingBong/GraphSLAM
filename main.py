from util import *
import csv
from matplotlib import pyplot as plt
import time

data = csv.reader(open('movement.csv'), delimiter=',')
racecar = Car(wheel_circ=0.505 * np.pi, L=1.52)
imu = []
line_num = 0
total_batches = 0  # total number of batches
previous_is_cone = False  # use to determine when a new batch of cones comes
start_time = time.perf_counter()
prev_time = time.perf_counter()
edges = []
dts = []
for line in data:
    print("line", line_num)
    line_num += 1
    if get_msg(line) == 'MessageType':
        continue  # skip the non-data line

    nanos = float(line[2]) + float(line[1]) * 1000000000
    racecar.predict(nanos)
    if get_msg(line) == 'WheelSpeeds':
        previous_is_cone = False
        steer, bLeftRPM, bRightRPM = float(line[15]), float(line[17]), float(line[19])
        racecar.setVel(steer, bLeftRPM, bRightRPM)
    if get_msg(line) == "Imu":
        previous_is_cone = False
        x, y, z, w = float(line[8]), float(line[9]), float(line[10]), float(line[11])
        t3 = 2.0 * (w * z + x * y)
        t4 = 1 - 2.0 * (y * y + z * z)
        angle = math.atan2(t3, t4)
        racecar.state_hist[-1][2] = angle
        imu.append(angle)
    if get_msg(line) == 'Cones':
        if not previous_is_cone:
            total_batches += 1
        if total_batches % 2 == 1 and not previous_is_cone:  # solve SLAM every other batch
            racecar.addUSLAM()  # add the latest state to slam history
            racecar.solve()
        previous_is_cone = True
        r, theta, color = float(line[3]), float(line[4]), float(line[5])
        racecar.cone(r, theta, color)  # this uses the latest slam history update
print("total seconds taken", time.perf_counter() - start_time)


def convert_color(n):
    if n == 0:
        return 'red'
    if n == 1:
        return 'yellow'
    if n == 2:
        return 'blue'


states = racecar.slam_hist
cones = racecar.distinct_cones

plt.figure(1)
np_runtime = racecar.np_runtime_data
csr_runtime = racecar.csr_runtime_data
csc_runtime = racecar.csc_runtime_data
lil_runtime = racecar.lil_runtime_data

# plt.title("Edges vs Dt")
# plt.xlabel("Edges")  # it's technically not edges its just how many rows in A
# plt.ylabel("dt")
# plt.plot(np.array(np_runtime)[:, 0], np.array(np_runtime)[:, 1])
# plt.plot(np.array(lil_runtime)[:, 0], np.array(lil_runtime)[:, 1])
# plt.plot(np.array(csc_runtime)[:, 0], np.array(csc_runtime)[:, 1])
# plt.plot(np.array(csr_runtime)[:, 0], np.array(csr_runtime)[:, 1])
# plt.legend(['numpy', 'lil', 'csc', 'csr'])

plt.figure(2)
plt.title("Cone Data Association")
plt.plot(np.array(states)[:, 0], np.array(states)[:, 1])  # plot the states
plt.plot(np.array(racecar.yellow_cones)[:, 0], np.array(racecar.yellow_cones)[:, 1], 'o', color='yellow', markersize=1)
plt.plot(np.array(racecar.blue_cones)[:, 0], np.array(racecar.blue_cones)[:, 1], 'o', color='blue', markersize=1)
plt.plot(np.array(racecar.red_cones)[:, 0], np.array(racecar.red_cones)[:, 1], 'o', color='red', markersize=1)
plt.plot(np.array(racecar.distinct_cones)[:, 0], np.array(racecar.distinct_cones)[:, 1], 'o', color='black', markersize=3)

plt.figure(3)
plt.title("Car Position and Cones")
plt.plot(np.array(states)[:, 0], np.array(states)[:, 1])  # plot the states
for i in range(len(cones)):
    cone = cones[i]
    plt.scatter(cone[0], cone[1], s=16, facecolor=convert_color(racecar.distinct_cone_colors[i]), edgecolor='black')

plt.figure(4)
plt.title("Car Angle")
plt.plot(np.array(states)[:, 2])
plt.show()
