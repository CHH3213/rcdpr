# -*-coding:utf-8-*-
import random
import numpy as np


class ReplayBuffer:
    """
    a ring buffer for storing positive tensions and sampling for negative tensions
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, cable_tensions):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = cable_tensions
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, BATCH_SIZE):
        cable_tensions = random.sample(self.buffer, BATCH_SIZE)
        # print(cable_tensions)
        cable_tensions = np.sum(cable_tensions,axis=0)/BATCH_SIZE
        # print(batch)
        return cable_tensions

    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    rb = ReplayBuffer(10)
    for i in range(10):
        cable_tensions = np.array([i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7])
        rb.push(cable_tensions)
    print(rb.buffer)
    tensions = rb.sample(2)
    print(tensions)

