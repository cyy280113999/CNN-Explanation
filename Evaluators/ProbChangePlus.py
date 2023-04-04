from Evaluators.BaseEvaluator import *
from utils.plot import heatmapNormalizeR, toPlot, lrp_lut, plotItemDefaultConfig
from utils.image_dataset_plot import invStd
from utils import binarize
prob_change = lambda p1, p2: p2 - p1


class ProbChangePlusEvaluator(BaseEvaluator):
    def __init__(self, ds_n, ds, dl, md_n, md, hm_n, hm_m,
                 eval_vis_check):
        super().__init__(ds_n, ds, dl, md_n, md, hm_n, hm_m,
                         eval_vis_check=eval_vis_check)
        self.ratios = torch.arange(0.1, 1, 0.1, device='cuda')
        self.scores = torch.zeros((self.ds_len, len(self.ratios)), device='cuda')
        self.log_name = "datas/pcplog.txt"

    def eval_once(self, raw_inputs):
        x, y = raw_inputs
        net_fun = lambda x: nf.softmax(self.model(x), 1)[0, y]
        x = x.cuda()
        hm = self.heatmap_method(x, y)
        for i, p in enumerate(self.ratios):
            with torch.no_grad():
                yc = net_fun(x)
                masked_input = binarize(hm, sparsity=p) * x
                oc = net_fun(masked_input)
                pc = prob_change(yc, oc)
            self.scores[self.counter, i] = pc
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
            str(s) for s in score.cpu().detach().tolist()
        ]
        save_str = ','.join(main_info + append_info) + '\n'
        return save_str

    def evc_once(self, vc):
        x, y = vc.raw_inputs
        x = x.unsqueeze(0)
        hm = self.heatmap_method(x.cuda(), y)
        vc.imageCanvas.pglw.clear()
        pi:pg.PlotItem = vc.imageCanvas.pglw.addPlot(0,0)
        plotItemDefaultConfig(pi)
        ii = pg.ImageItem(toPlot(invStd(x)))
        pi.addItem(ii)
        hm = toPlot(hm.cpu().detach())
        ii = pg.ImageItem(hm, levels=[-1, 1], lut=lrp_lut, opacity=0.7)
        pi.addItem(ii)

