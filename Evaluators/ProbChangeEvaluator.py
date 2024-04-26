from PyQt5.QtCore import QTimer

from Evaluators.BaseEvaluator import *
from utils import binarize, invStd, toPlot, lrp_lut, plotItemDefaultConfig, RunningCost


class ProbChangeEvaluator(BaseEvaluator):
    def __init__(self, ds_n, ds, dl, md_n, md, hm_n, hm_m,
                 eval_vis_check):
        super().__init__(ds_n, ds, dl, md_n, md, hm_n, hm_m,
                         eval_vis_check=eval_vis_check)
        self.ratios = torch.arange(0.1, 1, 0.1, device='cuda')
        self.scores = torch.zeros((self.ds_len, len(self.ratios)), device='cuda')
        self.log_name = "datas/pclog.txt"

    def eval_once(self, raw_inputs):
        x, y = raw_inputs
        net_fun = lambda x: nf.softmax(self.model(x), 1)[0, y]
        x = x.cuda().requires_grad_()
        with torch.enable_grad():
            hm = self.heatmap_method(x, y)
        with torch.no_grad():
            before = net_fun(x)
            for i, p in enumerate(self.ratios):
                masked_input = binarize(hm, sparsity=p) * x
                after = net_fun(masked_input)
                # pc = prob_change(yc, oc)
                self.scores[self.counter, i] = after - before
        self.counter += 1

    def save_str(self):
        main_info = [
            self.ds_name,
            self.model_name,
            self.hm_name,
        ]
        scores = self.scores
        if self.counter != self.ds_len:
            print("not full dataset evaluated")
            scores = scores[:self.counter]
        score = scores.mean(0)
        append_info = [
            str(s) for s in score.detach().cpu().tolist()
        ]
        save_str = ','.join(main_info + append_info) + '\n'
        return save_str

    def evc_once(self, vc):
        if hasattr(self,'timer') and self.timer is not None:
            self.timer.stop()
        x, y = vc.raw_inputs
        x = x.unsqueeze(0)
        hm = self.heatmap_method(x.cuda(), y).detach().cpu()
        vc.imageCanvas.pglw.clear()
        pi = vc.imageCanvas.pglw.addPlot(0,0)
        plotItemDefaultConfig(pi)
        pi.addItem(pg.ImageItem(toPlot(invStd(x))))
        pi.addItem(pg.ImageItem(toPlot(hm)), levels=[-1,1], lut=lrp_lut, opacity=0.7)
        # 2
        pi = vc.imageCanvas.pglw.addPlot(0, 1)
        plotItemDefaultConfig(pi)
        self.update_p = 0
        masked_input = binarize(hm, sparsity=self.ratios[self.update_p]) * x
        self.update_i=pg.ImageItem(toPlot(masked_input), levels=[-1, 1], lut=lrp_lut, opacity=0.7)
        pi.addItem(self.update_i)
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.update(x,hm))
        self.timer.start(300)

    def update(self,x,hm):
        self.update_p = (self.update_p + 1) % len(self.ratios)
        masked_input = binarize(hm, sparsity=self.ratios[self.update_p]) * x
        self.update_i.setImage(toPlot(masked_input))




