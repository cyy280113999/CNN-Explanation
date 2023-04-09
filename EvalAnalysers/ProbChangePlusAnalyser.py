from BaseSingleLineAnalyser import *
import torch


class ProbChangePlusAnalyser(BaseSingleLineAnalyser):
    def setting(self):
        self.setWindowTitle('Prob Change Plus Analyser')
        self.filename = '../datas/pcplog.txt'
        self.x = torch.arange(0.1, 1, 0.1) # L <= v < R


if __name__ == '__main__':
    analyse_main(ProbChangePlusAnalyser)
