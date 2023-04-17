from Evaluators.BaseEvaluator import *
from utils import binarize

prob_change = lambda p1, p2: p2 - p1

class PixelExpEvaluator(BaseEvaluator):
    "requires pixel-level heatmap"
    def __init__(self, ds_n, ds, dl, md_n, md, hm_n, hm_m,
                 eval_vis_check):
        super().__init__(ds_n, ds, dl, md_n, md, hm_n, hm_m,
                         eval_vis_check=eval_vis_check)
        self.scores=torch.zeros(self.ds_len,device='cuda')
        self.log_name="datas/pxlog.txt"

    def eval_once(self, raw_inputs):
        x,y=raw_inputs
        net_fun = lambda x: nf.softmax(self.model(x), 1)[0, y]
        x = x.cuda()
        hm=self.heatmap_method(x,y)
        with torch.no_grad():
            yc = net_fun(x)
            masked_input = binarize(hm, sparsity=0.9) * x
            oc = net_fun(masked_input)
            pc = prob_change(yc, oc)
        self.scores[self.counter] = pc
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
        score = scores.mean()
        std = scores.std()
        append_info=[
            f'{score}',
            f'{std}',
        ]
        save_str = ','.join(main_info+append_info) + '\n'
        return save_str
