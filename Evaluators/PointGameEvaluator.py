import torch

from Evaluators.BaseEvaluator import *
import pyqtgraph as pg
import numpy as np

from utils.image_dataset_plot import invStd
from utils.func import RunningCost
from utils.plot import toPlot, lrp_lut, plotItemDefaultConfig
from pyqtgraph import mkPen


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
    def __init__(self, ds_n, ds, dl, md_n, md, hm_n, hm_m,
                 eval_vis_check):
        super().__init__(ds_n, ds, dl, md_n, md, hm_n, hm_m,
                         eval_vis_check=eval_vis_check)
        self.log_name = "datas/pglog.txt"
        self.remain_ratios = torch.arange(0.05, 1, 0.05)  # L <= v < R
        self.scores = torch.zeros(len(self.remain_ratios), len(ds)).cuda()

    def eval_once(self, raw_inputs):
        x, y, bboxs = raw_inputs
        x = x.cuda()
        with torch.enable_grad():
            hm = self.heatmap_method(x, y).clip(min=0).cpu().detach().squeeze(0).squeeze(0)
        with torch.no_grad():
            energys = hm.flatten()
            energys = energys.sort()[0]
            cum_energy = energys.cumsum(0)
            energy_sum = cum_energy[-1]
            bbox_mat = torch.zeros(hm.shape[-2:], dtype=torch.bool)  # multi bbox overlapped
            for b in bboxs:
                xmin, ymin, xmax, ymax = b
                bbox_mat[ymin:ymax + 1, xmin:xmax + 1] = True
            for row, ratio in enumerate(self.remain_ratios):
                i = torch.searchsorted(cum_energy, ratio * energy_sum)
                if i == len(cum_energy):
                    i -= 1
                threshood = energys[i]
                bin_cam: torch.Tensor = hm >= threshood
                nonzero = bin_cam.count_nonzero()
                if nonzero == 0:
                    continue
                score = bin_cam.bitwise_and(bbox_mat).count_nonzero() / nonzero
                self.scores[row, self.counter] = score
        self.counter += 1

    def save_str(self):
        main_info = [
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
        append_info = [
            str(s) for s in score.cpu().detach().tolist()
        ]
        save_str = ','.join(main_info + append_info) + '\n'
        return save_str

    def evc_once(self, vc):
        x, y, bboxs = vc.raw_inputs
        x = x.unsqueeze(0)
        hm = self.heatmap_method(x.cuda(), y).clip(min=0).cpu().detach()
        vc.imageCanvas.pglw.clear()
        pi = vc.imageCanvas.pglw.addPlot()
        # 1
        ii = pg.ImageItem(toPlot(invStd(x)))
        pi.addItem(ii)
        # 2
        hm = toPlot(hm)
        ii = pg.ImageItem(hm, levels=[-1, 1], lut=lrp_lut, opacity=0.7)
        pi.addItem(ii)
        # 3
        for bbox in bboxs:
            xmin, ymin, xmax, ymax = bbox
            boxpdi = pg.PlotDataItem(x=[xmin, xmax, xmax, xmin, xmin],
                                     y=[ymin, ymin, ymax, ymax, ymin],
                                     pen=mkPen(color=(0, 127, 0), width=3), opacity=0.7)
            pi.addItem(boxpdi)
        plotItemDefaultConfig(pi)
