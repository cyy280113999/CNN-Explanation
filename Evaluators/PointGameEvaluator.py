import torch

from Evaluators.BaseEvaluator import *
import pyqtgraph as pg
import numpy as np
from utils import RunningCost

# def getBBoxScore(model, data, hm,threshood):
#     global counter
#     net_fun = lambda x: nf.softmax(model(x), 1)[0, y]
#     x,y,bboxs = data
#     cam = hm(x, y)
#     cam = normalize_R(cam)
#     cam = 1*cam>=threshood
#
#     score = 0
#     for b in bboxs:
#         # [xmin, ymin, xmax, ymax]
#         score+=cam[b[1]:b[3],b[0]:b[2]].sum()
#     score/=cam.sum()
#
#     return score


class PointGameEvaluator(BaseEvaluator):
    "requires Bounding Box Dataset"
    ONE_BBOX=True
    def __init__(self, ds_name, ds, dl, model_name, model, hm_name, heatmap_method):
        super().__init__(ds_name, ds, dl, model_name, model, hm_name, heatmap_method)

        self.log_name = "datas/pglog.txt"
        self.remain_ratios = torch.arange(0.05, 1, 0.05)  # L <= v < R
        self.scores = torch.zeros(len(self.remain_ratios), len(ds)).cuda()

    def eval_once(self, raw_inputs):
        x, y, bboxs=raw_inputs
        hm=self.heatmap_method(x,y).clip(min=0).cpu().detach()
        hm/=hm.max()
        energy = hm.flatten()
        energy.sort()
        cum_energy = energy.cumsum(0)
        energy_sum=cum_energy[-1]
        for row, ratio in enumerate(self.remain_ratios):
            i=torch.searchsorted(cum_energy, ratio * energy_sum)
            if i==len(cum_energy):i-=1
            threshood = energy[i]
            bin_cam = (hm >= threshood).int()
            # hit ratio
            score = 0
            nonzero = bin_cam.count_nonzero()
            if nonzero==0:
                continue
            for b in bboxs:
                xmin, ymin, xmax, ymax = b
                score += bin_cam[0, 0, ymin:ymax, xmin:xmax].count_nonzero()
                if self.ONE_BBOX:
                    break  # can not handle multi bbox
            score = score / nonzero
            self.scores[row, self.counter] = score
        self.counter += 1

    def save_str(self):
        main_info=[
            self.ds_name,
            self.model_name,
            self.hm_name,
        ]
        scores = self.scores.cpu().detach()
        # # show e-s plot
        # pw=pg.plot(title='e-s plot')
        # for row in range(self.energys.shape[0]):
        #     pw.plot(self.energys.numpy()[row],scores.numpy()[row], pen=None, symbol='o')# no line, dot marker.
        # pg.exec()
        if self.counter != self.ds_len:
            print("not full dataset evaluated")
            scores = scores[:self.counter]
        score = scores.mean(1)
        append_info=[
            str(s) for s in score.cpu().detach().tolist()
        ]
        save_str = ','.join(main_info+append_info) + '\n'
        return save_str