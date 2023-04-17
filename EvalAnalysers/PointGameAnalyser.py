from BaseSingleLineAnalyser import *
import torch


class PointGameAnalyser(BaseSingleLineAnalyser):
    def setting(self):
        self.setWindowTitle('Point Game Analyser')
        self.filename = '../datas/pglog.txt'
        self.x = torch.arange(0.05, 1, 0.05)  # L <= v < R


if __name__ == '__main__':
    analyse_main(PointGameAnalyser)



