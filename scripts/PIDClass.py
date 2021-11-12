# -*-coding:utf-8-*-

"""
位置式
"""


class PID:
    def __init__(self, k = [1.,0.,0.], target=1.0, upper=1.0, lower=-1.0) -> None:
        self.kp, self.ki, self.kd = k

        self.e = 0  # error
        self.pre_e = 0  # previous error
        self.sum_e = 0  # sum of error

        self.target = target # target
        self.upper_bound = upper    # upper bound of output
        self.lower_bound = lower    # lower bound of output

    def set_target(self,target):
        self.target = target

    def set_k(self, k):
        self.kp, self.ki, self.kd = k
    
    def set_bound(self,upper, lower):
        self.upper_bound = upper
        self.lower_bound = lower
    
    def cal_output(self, obs):   # calculate output
        self.e = self.target - obs
        
        u = self.e * self.kp + self.sum_e * self.ki + (self.e - self.pre_e)* self.kd
        if u < self.lower_bound:
            u = self.lower_bound
        elif u > self.upper_bound:
            u = self.upper_bound

        self.pre_e = self.e
        self.sum_e += self.e
        return u
    
    def reset(self):
        # self.kp = 0
        # self.ki = 0
        # self.kd = 0

        self.e = 0
        self.pre_e = 0
        self.sum_e = 0
        # self.target = 0

if __name__ == '__main__':
    pass