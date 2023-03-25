import os
from time import time
from tqdm import tqdm


# make path (recursively)
def mkp(p):
    d = os.path.dirname(p)
    if d not in ['', '.', '..'] and not os.path.exists(d):
        mkp(d)
        os.mkdir(d)


# running cost
class RunningCost:
    def __init__(self, stage_count=5):
        self.stage_count = stage_count
        self.running_cost = [None for i in enumerate(range(self.stage_count + 1))]
        self.hint = [None for i in enumerate(range(self.stage_count))]
        self.position = 0

    def tic(self, hint=None):
        if self.position < self.stage_count:
            t = time()
            self.running_cost[self.position] = t
            self.hint[self.position]=hint
            self.position += 1

    def cost(self):
        print('-'*20)
        for stage_, (i, j) in enumerate(zip(self.running_cost, self.running_cost[1:])):
            if j is not None:
                if self.hint[stage_+1] is not None:
                    print(f'stage {self.hint[stage_+1]} cost time: {j - i}')
                else:
                    print(f'stage {stage_+1} cost time: {j - i}')

        print('-' * 20)





class AvailableMethods:
    def __init__(self, s):
        self.methodsSet=s
        self.__dict__.update({i:i for i in self.methodsSet})

    def __iter__(self):
        return self.methodsSet.__iter__()
