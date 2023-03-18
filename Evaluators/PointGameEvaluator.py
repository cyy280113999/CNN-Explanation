from Evaluators.BaseEvaluator import *
import pyqtgraph as pg
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
    def __init__(self, ds_name, ds, dl, model_name, model, hm_name, heatmap_method):
        super().__init__(ds_name, ds, dl, model_name, model, hm_name, heatmap_method)

        self.log_name = "datas/pclog.txt"
        self.threshoods = torch.arange(0.05, 1, 0.05)
        self.energys = torch.zeros(len(self.threshoods), len(ds)).cuda()
        self.scores = torch.zeros(len(self.threshoods), len(ds)).cuda()

    def eval_once(self, raw_inputs):
        x, y, bboxs=raw_inputs
        net_fun = lambda x: nf.softmax(self.model(x), 1)[0, y]
        hm=self.heatmap_method(x,y).clip(min=0).cpu().detach()
        hm/=hm.max()
        energy_base = hm.count_nonzero()
        for row, threshood in enumerate(self.threshoods):
            bin_cam = (hm >= threshood).int()
            non_zero = bin_cam.count_nonzero()
            energy = non_zero / energy_base
            score = 0
            for b in bboxs:
                xmin, ymin, xmax, ymax = b
                score += bin_cam[0, 0, ymin:ymax, xmin:xmax].count_nonzero()
                continue  # take in one bbox
            score = score / bin_cam.count_nonzero()
            self.energys[row, self.counter] = energy
            self.scores[row, self.counter] = score
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


        self.energys = self.energys.cpu().detach()
        self.scores = self.scores.cpu().detach()
        # # show e-s plot
        pw=pg.plot(title='e-s plot')
        for row in range(self.energys.shape[0]):
            pw.plot(self.energys.numpy()[row],scores.numpy()[row], pen=None, symbol='o')# no line, dot marker.
        pg.exec()
        score = scores.mean(1)
        score_std = scores.std(1)

        save_str = ''.join(f"{self.hm_name},{threshood.item()},"
                           f"{score[row].item()},{score_std[row].item()}\n" for row, threshood in enumerate(threshoods))

        print(f'name:{self.hm_name}')
        print(f'prob change :{score:6f}+-{score_std:6f}')


        return save_str