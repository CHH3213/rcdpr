# import struct
import numpy as np
import math
from time import sleep
from time import time


# 位置式
class PID_posi:
    def __init__(self, kp, ki, kd, target, upper=1., lower=-1.):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.err = 0
        self.err_last = 0
        self.err_all = 0
        self.target = target
        self.upper = upper
        self.lower = lower
        self.value = 0

    def increase(self, state):
        self.err = self.target - state
        # self.err =state-self.target
        self.value = self.kp * self.err + self.ki * self.err_all + self.kd * (self.err - self.err_last)
        self.update()

    def update(self):
        self.err_last = self.err
        self.err_all = self.err_all + self.err
        if self.value > self.upper:
            self.value = self.upper
        elif self.value < self.lower:
            self.value = self.lower

    def auto_adjust(self, Kpc, Tc):
        self.kp = Kpc * 0.6
        self.ki = self.kp / (0.5 * Tc)
        self.kd = self.kp * (0.125 * Tc)
        return self.kp, self.ki, self.kd

    def set_pid(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def reset(self):
        self.err = 0
        self.err_last = 0
        self.err_all = 0

    def set_target(self, target):
        self.target = target


# 增量式
class PID_inc:
    def __init__(self, kp, ki, kd, target, upper=1., lower=-1.):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.err = 0
        self.err_last = 0
        self.err_ll = 0
        self.target = target
        self.upper = upper
        self.lower = lower
        self.value = 0
        self.inc = 0

    def increase(self, state):
        self.err = self.target - state
        self.inc = self.kp * (self.err - self.err_last) + self.ki * self.err + self.kd * (
                    self.err - 2 * self.err_last + self.err_ll)
        self.update()
        return self.value

    def update(self):
        self.err_last = self.err
        self.err_ll = self.err_last
        self.value = self.value + self.inc
        if self.value > self.upper:
            self.value = self.upper
        elif self.value < self.lower:
            self.value = self.lower

