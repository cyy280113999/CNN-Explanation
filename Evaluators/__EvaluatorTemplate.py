from Evaluators.BaseEvaluator import *
from utils import invStd, toPlot, lrp_lut, plotItemDefaultConfig


class YourEvaluator(BaseEvaluator):
    def __init__(self, ds_n, ds, dl, md_n, md, hm_n, hm_m,
                 eval_vis_check):
        super().__init__(ds_n, ds, dl, md_n, md, hm_n, hm_m,
                         eval_vis_check=eval_vis_check)
        # === config this settings
        self.log_name=f"datas/yourlog.txt"
        # === create empty space to store scores
        # self.scores=torch.zeros(self.num_samples,device='cuda')

    def eval_once(self, raw_inputs):
        # ===== your evaluation on one sample
        # -your dataset unpacking
        x,y=raw_inputs
        # -heatmap generating
        hm=self.heatmap_method(x.cuda(),y)
        # -your score saving
        # self.scores[self.counter, i]=
        # =========
        self.counter += 1

    # this is how eval called
    # def eval(self):
    #     if self.eval_vis_check:
    #         self.eval_visualization_check()
    #     else:
    #         for raw_inputs in tqdm(self.dataLoader):
    #             self.eval_once(raw_inputs)
    #         self.save()

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

    # ---- visual check. visualize every evaluated sample to check.
    def evc_once(self, vc):
        x, y = vc.raw_inputs
        x = x.unsqueeze(0)
        hm = self.heatmap_method(x.cuda(), y).clip(min=0).detach().cpu()
        vc.imageCanvas.pglw.clear()
        pi = vc.imageCanvas.pglw.addPlot()
        plotItemDefaultConfig(pi)
        pi.addItem(pg.ImageItem(toPlot(invStd(x))))
        # 2
        pi.addItem(pg.ImageItem(toPlot(hm), levels=[-1, 1], lut=lrp_lut, opacity=0.7))


    #  this is how window created.
    # def eval_visualization_check(self):
    #     qapp = QApplication.instance()
    #     if qapp is None:
    #         qapp=QApplication(sys.argv)
    #     self.vc=BaseDatasetTravellingVisualizer(self.dataset, imageChangeCallBack=self.evc_once)
    #     self.vc.setWindowTitle(self.hm_name)
    #     self.vc.show()
    #     qapp.exec_()