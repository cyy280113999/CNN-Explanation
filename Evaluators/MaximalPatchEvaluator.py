import torch

from Evaluators.BaseEvaluator import *
from utils import invStd, toPlot, lrp_lut, plotItemDefaultConfig, maximalPatch,maximalLoc,patch


prob_change = lambda p1, p2: p2 - p1


class MaximalPatchEvaluator(BaseEvaluator):
    def __init__(self, ds_n, ds, dl, md_n, md, hm_n, hm_m,
                 eval_vis_check):
        super().__init__(ds_n, ds, dl, md_n, md, hm_n, hm_m,
                         eval_vis_check=eval_vis_check)
        # -----------masking methods
        r_range = [1, 2, 3, 5, 10, 20]
        self.masks = [(True, r) for r in r_range]  # max, top=True
        self.masks +=[(False, r) for r in r_range]  # min
        # masks.update({f"min_{r}": partial(maximalPatch, top=False, r=r) for r in r_range})

        self.scores=torch.zeros(self.num_samples,len(self.masks),device='cuda')
        self.log_name=f"datas/mplog.txt"

    def eval_once(self, raw_inputs):
        x,y=raw_inputs
        net_fun = lambda x: nf.softmax(self.model(x), 1)[0, y]
        x = x.cuda().requires_grad_()
        with torch.enable_grad():
            hm=self.heatmap_method(x,y)
        with torch.no_grad():
            yc = net_fun(x)
            maxloc=maximalLoc(hm,True)
            minloc=maximalLoc(hm,False)
            for i, (top,r) in enumerate(self.masks):
                mask=patch(hm, maxloc if top else minloc, r)
                masked_input = mask * x
                oc = net_fun(masked_input)
                self.scores[self.counter,i] = prob_change(yc, oc)
        self.counter += 1

    def save_str(self):
        main_info=[
            self.ds_name,
            self.model_name,
            self.hm_name,
        ]
        scores = self.scores
        if self.counter != self.ds_len:
            print("not full dataset evaluated")
            scores = scores[:self.counter]
        score = scores.mean(0)
        append_info=[
            str(s) for s in score.detach().cpu().tolist()
        ]
        save_str = ','.join(main_info+append_info) + '\n'
        return save_str

    def evc_once(self, vc):
        x, y = vc.raw_inputs
        x = x.unsqueeze(0)
        with torch.enable_grad():
            hm = self.heatmap_method(x.cuda(), y).detach().cpu()
        maxloc = maximalLoc(hm, True)
        minloc = maximalLoc(hm, False)
        vc.imageCanvas.pglw.clear()
        # 1
        pi = vc.imageCanvas.pglw.addPlot()
        plotItemDefaultConfig(pi)
        pi.addItem(pg.ImageItem(toPlot(invStd(x)), levels=[0, 1], lut=lrp_lut, opacity=0.7))
        pi.addItem(pg.ImageItem(toPlot(hm), levels=[-1, 1], lut=lrp_lut, opacity=0.7))
        # 2
        masked_input = x
        masked_input = masked_input * patch(hm, maxloc, r=10)
        masked_input = masked_input * patch(hm, minloc, r=10)
        pi = vc.imageCanvas.pglw.addPlot()
        plotItemDefaultConfig(pi)
        pi.addItem(pg.ImageItem(toPlot(masked_input), levels=[-1, 1], lut=lrp_lut, opacity=0.7))

