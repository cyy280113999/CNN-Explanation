from BaseSingleLineAnalyser import *
import torch


class ProbChangeAnalyser(BaseSingleLineAnalyser):
    def setting(self):
        self.setWindowTitle('Prob Change Analyser')
        self.filename = '../datas/pclog.txt'
        self.x = torch.arange(0.1, 1, 0.1) # L <= v < R


if __name__ == '__main__':
    analyse_main(ProbChangeAnalyser)
