import re
import json
from itertools import product
from functools import partial
import os

from tqdm import tqdm
import torch
import torch.utils.data as TD
import torch.nn.functional as nf
import torchvision
import numpy as np

# user
from utils import *
from EvalSettings import dataset_callers,models,eval_heatmap_methods, ds_name, model_name, EvalClass, eval_vis_check


class EvaluatorSetter:
    def __init__(self, dataset_callers, models, eval_heatmap_methods):
        """
        one dataset with one model at a time.
        """
        self.dataset_callers = dataset_callers
        self.models = models
        self.heatmap_methods = eval_heatmap_methods

    def presetting(self,  dataset_name, model_name, eval_vis_check):
        self.dataset_name = dataset_name
        self.dataset = self.dataset_callers[self.dataset_name]()
        self.dataloader = TD.DataLoader(self.dataset, batch_size=1, pin_memory=True, num_workers=2,
                                        persistent_workers=True)

        self.model_name = model_name
        self.model = self.models[self.model_name]()

        # ---eval explaining methods
        self.eval_vis_check = eval_vis_check

    def eval(self, heatmap_name, SubEvalClass):
        heatmap_method = self.heatmap_methods[heatmap_name](self.model)
        evaluator = SubEvalClass(self.dataset_name, self.dataset, self.dataloader,
                                      self.model_name, self.model,
                                      heatmap_name, heatmap_method,
                                      self.eval_vis_check)

        evaluator.eval()

    def eval_all_hm(self, SubEvalClass):
        for hm_name in self.heatmap_methods:
            self.eval(hm_name, SubEvalClass)


qapp = None
if __name__ == '__main__':
    print('utf8 chinese test: 中文测试')
    if eval_vis_check:
        qapp = QApplication(sys.argv)
    # notice which called by loop.
    mainEvaluator = EvaluatorSetter(dataset_callers,models,eval_heatmap_methods)
    mainEvaluator.presetting(ds_name, model_name,eval_vis_check)
    mainEvaluator.eval_all_hm(EvalClass)


