import numpy as np
from numpy.linalg import norm


def steer(x_nearest, x_rand, eta):
    x_new = np.zeros(2)
    if norm(x_nearest - x_rand) <= eta:
        x_new = x_rand
    else:
        m = (x_rand[1] - x_nearest[1]) / (x_rand[0] - x_nearest[0])
        x_new[0] = np.sign(x_rand[0] - x_nearest[0]) * np.sqrt(pow(eta, 2) / (pow(m, 2) + 1)) + x_nearest[0]
        x_new[1] = m * (x_new[0] - x_nearest[0]) + x_nearest[1]
        if x_rand[0] == x_nearest[0]:
            x_new[0] = x_nearest[0]
            x_new[1] = x_nearest[1] + eta
    return x_new


def nearest(V, x):
    min_dist = np.inf
    min_v = None  # assuming V is not empty
    for v in V:
        dist = norm(v - x)
        if dist < min_dist:
            min_dist = dist
            min_v = v
    return min_v


def grid_value(map_data, point):
    resolution = map_data.info.resolution
    m_startx = map_data.info.origin.position.x
    m_starty = map_data.info.origin.position.y
    width = map_data.info.width
    idx = (np.floor((point[1] - m_starty) / resolution) * width) + np.floor((point[0] - m_startx) / resolution)
    return map_data.data[int(idx)]


def obstacle_free(x_nearest, x_new, map_data):
    rez = float(map_data.info.resolution) * 0.2
    steps = int(np.ceil(norm(x_new - x_nearest)) / rez)
    xi = x_nearest
    obs = 0
    unknown = 0

    for i in range(steps):
        xi = steer(xi, x_new, rez)
        grid_v = grid_value(map_data, xi)  # grid_v: 0:free, -1:unknown, 100:obstacle
        if grid_v > 80:
            obs = 1
        elif grid_v == -1:
            unknown = 1
            break

    # out: -1: unknown; 0: obstacle; 1: free
    if unknown == 1:
        out = -1
    elif obs == 1:
        out = 0
    else:
        out = 1
    return out, xi


def cal_dist(input_loc, dest_loc):
    return np.sqrt(np.power(input_loc[0] - dest_loc[0], 2) + np.power(input_loc[1] - dest_loc[1], 2))


def grid_value_merged_map(mapData, Xp, distance=2):
    resolution = mapData.info.resolution
    xstartx = mapData.info.origin.position.x
    xstarty = mapData.info.origin.position.y

    width = mapData.info.width
    Data = mapData.data
    # returns grid value at "Xp" location
    # map data:  100 occupied      -1 unknown       0 free
    index = np.floor((Xp[1] - xstarty) / resolution) * width + np.floor((Xp[0] - xstartx) / resolution)
    outData = square_area_check(Data, index, width, distance)
    if len(outData) > 1:
        if 100 not in outData:
            if max(outData, key=outData.count) == -1 and max(outData) == 0:
                return -1
            elif max(outData, key=outData.count) == -1 and max(outData) == -1 and 0 not in outData:
                return 100
            elif max(outData, key=outData.count) == -1 and max(outData) == -1 and 0 in outData:
                return -1
            elif max(outData, key=outData.count) == 0 and -1 in outData:
                return -1
            else:
                return -1
        else:
            return 100
    else:
        return 100


# ________________________________________________________________________________
def square_area_check(data, index, width, distance=2):
    # now using the data to perform a square area check on the data for removing the invalid point
    dataOutList = []
    for j in range(-1 * distance, distance + 1):
        for i in range(int(index) + ((width * j) - distance), int(index) + ((width * j) + distance)):
            if i < len(data):
                dataOutList.append(data[int(i)])
    return dataOutList


def information_gain(mapData, point, r):
    infoGain = 0.0
    index = index_of_point(mapData, point)
    r_region = int(r / mapData.info.resolution)
    init_index = index - r_region * (mapData.info.width + 1)
    for n in range(0, 2 * r_region + 1):
        start = n * mapData.info.width + init_index
        end = start + 2 * r_region
        limit = ((start / mapData.info.width) + 2) * mapData.info.width
        for i in range(start, end + 1):
            if (i >= 0 and i < limit and i < len(mapData.data)):
                if (mapData.data[i] == -1 and norm(np.array(point) - point_of_index(mapData, i)) <= r):
                    infoGain = infoGain + 1.0
    return infoGain * (mapData.info.resolution ** 2)


def index_of_point(mapData, Xp):
    resolution = mapData.info.resolution
    Xstartx = mapData.info.origin.position.x
    Xstarty = mapData.info.origin.position.y
    width = mapData.info.width
    Data = mapData.data
    index = int((np.floor((Xp[1] - Xstarty) / resolution) * width) + (np.floor((Xp[0] - Xstartx) / resolution)))
    return index


def point_of_index(mapData, i):
    y = mapData.info.origin.position.y + (i / mapData.info.width) * mapData.info.resolution
    x = mapData.info.origin.position.x + float(
        i - (int(i / mapData.info.width) * mapData.info.width)) * mapData.info.resolution
    # modified for certain python version might mismatch the int and float conversion
    return np.array([x, y])
