from Evaluators.BaseEvaluator import *


class YourEvaluator(BaseEvaluator):
    def __init__(self, ds_name, ds, dl, model_name, model, hm_name, heatmap_method):
        super().__init__(ds_name, ds, dl, model_name, model, hm_name, heatmap_method)

        # === config this settings
        self.log_name=f"datas/yourlog.txt"
        # === create empty space to store scores
        # self.scores=torch.zeros(self.num_samples,len(self.masks),device='cuda')

    def eval_once(self, raw_inputs):
        # ===== your evaluation on one sample
        # -your dataset unpacking
        # x,y=raw_inputs
        # x = x.cuda()
        # -heatmap generating
        # hm=self.heatmap_method(x,y)
        # -your score saving
        # self.scores[self.counter, i]=
        # =========
        self.counter += 1

    def save_str(self):
        main_info=[
            self.ds_name,
            self.model_name,
            self.hm_name,
        ]
        # ====== your statistic result
        # after all samples evaluated, make statistic for saving.
        # if self.counter != self.ds_len:
        #     print("not full dataset evaluated")
        #     self.scores = self.scores[:self.counter]
        # score = scores.mean(0)
        # ========
        append_info=[
            # Your evaluation result
        ]
        save_str = ','.join(main_info+append_info) + '\n'
        return save_str