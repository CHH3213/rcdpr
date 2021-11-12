import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from torch.autograd import Variable
import random
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from numpy.core.fromnumeric import shape
import torch
import scipy.io as sio
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

from value_function import Value_function

save_data_dir = "../scripts/data/"

class Net(nn.Module):
    def __init__(self, n_input=42, n_hidden1=32, n_hidden2=32, n_hidden3=16, n_output=1):
        super(Net, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(n_input, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_hidden3),
            nn.ReLU(),
            nn.Linear(n_hidden3, n_output),
        )

    def forward(self, input):
        return self.nn(input)

def real_value_func(value_input):
    cable_one_side = value_input['CABLE_ONE_SIDE']
    cable_other_side = value_input['CABLE_OTHER_SIDE']
    pose_0 = value_input['POSE_0']
    rot_center = value_input['ROTATION_CENTER']
    point_end_effector = value_input['point_end_effector']
    cable_length = np.array([5 for _ in range(7)])  # FIXME: 改为5， 原来是4.5

    Value.set_jacobian_param(point_end_effector, pose_0, rot_center)
    v1 = Value.cost_feasible_points(
        cable_one_side, cable_other_side, cable_length)
    v2 = Value.cost_cable_interference(cable_one_side, cable_other_side)
    # v3 = Value.cost_cable_length(cable_one_side, cable_other_side)
    r1 = Value.r_t_AW(cable_one_side, cable_other_side)
    r2 = Value.r_r_AW(cable_one_side, cable_other_side, rot_center)
    v4 = r1+r2
    y_value = v1+v2+v4


def compare(net,filename ):
    value_input = sio.loadmat(save_data_dir+filename+'.mat')
    


if __name__ == '__main__':
    Value = Value_function()
    net = Net(14, 512, 512, 256, 1).to(device)
    net.load_state_dict(torch.load(
        '/home/firefly/chh_ws/src/plan_cdpr/scripts/model/model_14dim/2021-11-10-17-44-env.pt'))
    
    Filenames = []
    for filename in Filenames:
        compare(net,filename)
