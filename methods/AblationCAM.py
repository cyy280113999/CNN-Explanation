
# wc_k= (yc - yc_without_k)/yc

import torch.nn.functional as nf
from utils import *


class AblationCAM:
    def __init__(self, model, layer=None):
        self.model = model

        self.features = list(self.model.features)
        self.flatten = torch.nn.Flatten(1)
        self.classifier = self.model.classifier

        # def backward_hook(module, grad_input, grad_output):
        #     self.gradients = grad_output[0]
        #     return None

        # def forward_hook(module, x, output):
        #     self.activations = output
        #     return None

        layer = auto_find_layer_index(self.model,layer) - 1
        self.features1 = self.features[:layer+1]
        self.features2 = self.features1[layer+1:]
        # self.features[layer].register_forward_hook(forward_hook)
        # self.features[layer].register_backward_hook(backward_hook)

        # config with your gpu
        self.parallel_batch = 128
        # 1     100%|██████████| 10/10 [00:06<00:00,  1.49it/s]
        # 2     100%|██████████| 10/10 [00:03<00:00,  2.80it/s]
        # 8     100%|██████████| 10/10 [00:01<00:00,  9.59it/s]
        # 16    100%|██████████| 10/10 [00:00<00:00, 18.18it/s]
        # 32    100%|██████████| 10/10 [00:00<00:00, 31.89it/s]
        # 64    100%|██████████| 10/10 [00:00<00:00, 38.53it/s]
        #       100%|██████████| 100/100 [00:02<00:00, 38.78it/s]
        # 128   100%|██████████| 100/100 [00:02<00:00, 46.65it/s]
        # 192
        # 256   100%|██████████| 100/100 [00:02<00:00, 40.52it/s]


        # short cut ratio
        self.top_percent = 1.  #

    def forward1(self, x):
        for l in self.features1:
            x = l(x)
        return x

    def forward2(self, x):
        for l in self.features2:
            x = l(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def __call__(self, x, class_idx=None, relu=True):
        with torch.no_grad():
            x=x.cuda()
            h, w = x.shape[2:]

            # predication on raw x
            activations = self.forward1(x)
            prob = self.forward2(activations).softmax(1)
            if class_idx is None:
                predicted_class = prob.argmax(1)
            elif isinstance(class_idx, int):
                predicted_class = torch.LongTensor([class_idx]).cuda()
            elif isinstance(class_idx, torch.Tensor):
                predicted_class=class_idx
            else:
                raise Exception()
            pc = prob[0, predicted_class]
            net_fun = lambda x: self.forward2(x).softmax(1)[:, predicted_class]

            # cut useless activations
            top_count = int(self.top_percent * activations.shape[1])
            channel_scores = activations.mean(axis=[2, 3], keepdim=False).flatten()
            top_indice = channel_scores.argsort(0,True)[:top_count]

            sub_scores = torch.zeros((top_count,)).cuda()
            for batch_start in range(0, top_count, self.parallel_batch):
                batch_stop_exclude = batch_start + self.parallel_batch
                if batch_stop_exclude > top_count:
                    batch_stop_exclude = top_count
                sub_indice = top_indice[batch_start:batch_stop_exclude]
                batch_ablation = torch.repeat_interleave(activations, len(sub_indice), 0)
                batch_ablation[:, sub_indice] = 0

                sub_scores[batch_start:batch_stop_exclude] = (pc-net_fun(batch_ablation).flatten())/pc
            abcam = (sub_scores.reshape(1,-1,1,1) * activations[:,top_indice]).sum(1, keepdim=True)
            abcam = nf.interpolate(abcam, size=(h, w), mode='bilinear', align_corners=False)
            if relu:
                abcam = nf.relu(abcam)
                abcam = heatmapNormalizeP(abcam)
            else:
                abcam = heatmapNormalizeR(abcam)

        return abcam

