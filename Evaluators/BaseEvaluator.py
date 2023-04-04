import sys

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication
from tqdm import tqdm
from functools import partial
import torch
import torch.nn.functional as nf
import torch.utils.data as TD
from utils.func import mkp
from utils.window_tools import BaseDatasetTravellingVisualizer, windowMain
import pyqtgraph as pg

def write(str, file):
    mkp(file)
    with open(file, 'a') as f:
        f.write(str)


class BaseEvaluator:
    def __init__(self, dataset_name, dataset, dataloader,
                 model_name, model,
                 heatmap_name, heatmap_method,
                 eval_vis_check=False):
        self.ds_name = dataset_name
        self.dataset = dataset
        self.ds_len = len(dataset)
        self.dataLoader = dataloader
        self.num_samples = len(self.dataLoader)

        self.model_name=model_name
        self.model = model

        self.hm_name=heatmap_name
        self.heatmap_method=heatmap_method

        self.eval_vis_check = eval_vis_check

        self.counter = 0

        # customized param
        # self.scores = torch.zeros(len(ds)).cuda()
        self.log_name=None

    # -------------evaluating
    def eval(self):
        if self.eval_vis_check:
            self.eval_visualization_check()
        else:
            for raw_inputs in tqdm(self.dataLoader):
                self.eval_once(raw_inputs)
            self.save()

    def eval_once(self,raw_inputs):
        raise NotImplementedError()

    def save(self):
        ans=self.save_str()
        print(ans)
        assert self.log_name is not None
        write(ans, self.log_name)

    def save_str(self):
        raise NotImplementedError()

    # ---- visual check.
    def eval_visualization_check(self):
        qapp = QApplication.instance()
        if qapp is None:
            qapp=QApplication(sys.argv)
        self.vc=BaseDatasetTravellingVisualizer(self.dataset, imageChangeCallBack=self.evc_once)
        self.vc.setWindowTitle(self.hm_name)
        self.vc.show()
        qapp.exec_()

    def evc_once(self, vc):
        raise NotImplementedError()


# class VisualizationChecker:
#     []