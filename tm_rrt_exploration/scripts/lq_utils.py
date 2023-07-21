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
    min_p = np.inf
    for v in V:
        if norm(v - x) < min_p:
            min_p = v
    return min_p


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
        grid_v = grid_value(map_data, xi)
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
    return xi, out
