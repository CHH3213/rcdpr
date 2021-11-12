import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


total_steps = 3000
radius = 0.1
c0 = [0.1,0]
save_data_dir = "../scripts/data"

def generate_ideal_trajectory():
    theta = 0
    trajectory_x = []
    trajectory_y = []
    for i in range(total_steps):
        theta += 2 * np.pi / (total_steps)
        # # print('theat',theta)
        dist_x = radius * np.cos(np.pi - theta) + c0[0]
        dist_y = radius * np.sin(np.pi - theta) + c0[1]
        trajectory_x.append(dist_x)
        trajectory_y.append(dist_y)
    return [trajectory_x, trajectory_y]

def generate_real_trajectory(Filename):
    trajectory_data = sio.loadmat(
        save_data_dir+Filename+'.mat')
    platform_pos = np.array(trajectory_data['platform_pos']).T
    return [platform_pos[0], platform_pos[1]]

def compare(Filename):
    [ideal_x, ideal_y] = generate_ideal_trajectory()
    [real_x, real_y] = generate_real_trajectory(Filename)

    dist = []
    for i in range(len(ideal_x)):
        dist.append(np.linalg.norm(
            np.array([ideal_x[i]-real_x[i], ideal_y[i] - real_y[i]])))
    
    sio.savemat(save_data_dir+'/dist_'+Filenames+'.mat', dict(dist=dist))

    fig,ax = plt.subplots(2,1)
    ax[0].plot(ideal_x, ideal_y, label='ideal')
    ax[0].plot(real_x, real_y, label='real')
    ax[1].plot(dist, label='distance')
    # plt.axis('equal')

    ax[0].set_title('trajectory')
    ax[1].set_title('distance of each point')
    
    ax[0].legend()
    ax[1].legend()

    ax[0].axis('equal')
    plt.show()


if __name__ == '__main__':
    Filenames = ['static_01_trajectry_2021-11-10-21:23:11']
    for filename in Filenames:
        compare(filename)
