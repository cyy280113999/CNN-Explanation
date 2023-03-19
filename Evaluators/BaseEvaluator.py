from tqdm import tqdm
from functools import partial
import torch
import torch.nn.functional as nf
import torch.utils.data as TD
from utils import mkp


def write(str, file):
    mkp(file)
    with open(file, 'a') as f:
        f.write(str)


class BaseEvaluator:
    def __init__(self, ds_name, dataset, dataloader, model_name, model, hm_name, heatmap_method):
        self.ds_name=ds_name
        self.dataset = dataset
        self.ds_len = len(dataset)
        self.dataLoader = dataloader
        self.num_samples = len(self.dataLoader)

        self.model_name=model_name
        self.model = model

        self.hm_name=hm_name
        self.heatmap_method=heatmap_method

        self.counter = 0

        # customized param
        # self.scores = torch.zeros(len(ds)).cuda()
        self.log_name=None

    # -------------evaluating
    def eval(self):
        for raw_inputs in tqdm(self.dataLoader):
            self.eval_once(raw_inputs)

    def eval_once(self,raw_inputs):
        raise NotImplementedError()
        self.counter+=1

    def save(self):
        ans=self.save_str()
        print(ans)
        assert self.log_name is not None
        write(ans, self.log_name)

    def save_str(self):
        raise NotImplementedError()
        scores = self.scores
        if self.counter != self.ds_len:
            print("not full dataset evaluated")
            scores = scores[:self.counter]
        save_str = f"{self.ds_name},{self.model_name},{self.hm_name}\n"

