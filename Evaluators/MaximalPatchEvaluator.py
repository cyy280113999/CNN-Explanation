from Evaluators.BaseEvaluator import *
from utils import maximalPatch,maximalLoc,patch

prob_change = lambda p1, p2: p2 - p1

class MaximalPatchEvaluator(BaseEvaluator):
    def __init__(self, ds_name, ds, dl, model_name, model, hm_name, heatmap_method):
        super().__init__(ds_name, ds, dl, model_name, model, hm_name, heatmap_method)

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
        x = x.cuda()
        hm=self.heatmap_method(x,y)
        with torch.no_grad():
            yc = net_fun(x)
            maxloc=maximalLoc(hm,True)
            minloc=maximalLoc(hm,False)
            for i, (top,r) in enumerate(self.masks):
                mask=patch(hm.shape[-2:], maxloc if top else minloc, r)
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
            str(s) for s in score.cpu().detach().tolist()
        ]
        save_str = ','.join(main_info+append_info) + '\n'
        return save_str