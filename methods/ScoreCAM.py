"""
score cam
sg=True, norm=False, relu=True

parallel batch
1     100%|██████████| 10/10 [00:28<00:00,  2.89s/it]
2     100%|██████████| 10/10 [00:22<00:00,  2.28s/it]
8     100%|██████████| 10/10 [00:17<00:00,  1.72s/it]
32    100%|██████████| 10/10 [00:15<00:00,  1.51s/it]
64    100%|██████████| 10/10 [00:13<00:00,  1.36s/it]
128   100%|██████████| 10/10 [00:14<00:00,  1.44s/it]

short cut ratio 0.1, 0.3, 0.5: finally 0.3
sub_imgnt,vgg16,ScoreCAM-1,-0.025340337306261063,-0.03987511619925499,-0.05381786823272705,-0.07178227603435516,-0.09759528189897537,-0.1367368996143341,-0.19820384681224823,-0.29197919368743896,-0.4410395920276642
sub_imgnt,vgg16,ScoreCAM-3,-0.025449801236391068,-0.03700661659240723,-0.05120148882269859,-0.06861212849617004,-0.09272456914186478,-0.1310550719499588,-0.19331781566143036,-0.2878183424472809,-0.43646150827407837
sub_imgnt,vgg16,ScoreCAM-5,-0.02595943585038185,-0.03754677250981331,-0.05140040069818497,-0.06853514164686203,-0.09233172982931137,-0.12974369525909424,-0.19212815165519714,-0.28568318486213684,-0.43570637702941895
"""

import torch.nn.functional as nf
from torchvision.models import VGG, AlexNet, ResNet, GoogLeNet

from utils import *


class ScoreCAM:
    def __init__(self, model, layer_names=None,
                 post_softmax=True, norm=False, relu=True, pp=False, top_percent=0.3,**kwargs):
        self.model = model
        self.layers = auto_hook(model, layer_names)
        self.relu=relu
        self.post_softmax=post_softmax
        self.norm=norm
        self.pp=pp
        # config with your gpu
        self.parallel_batch = 64

        # shortcut ratio
        self.top_percent = top_percent

    def __call__(self, x, yc):
        with torch.no_grad():
            if self.post_softmax:
                net_fun = lambda x: nf.softmax(self.model(x), 1)[:, yc]
            else:
                net_fun = lambda x: self.model(x)[:, yc]
            out = net_fun(x)
            hms = []
            for layer in self.layers:
                # net_fun(x)
                # use activation as masks
                activations = layer.activation
                # cut useless activations to sub-mask by sorted mean activation
                top_count = int(self.top_percent * activations.shape[1])
                channel_scores = activations.mean(axis=[2, 3], keepdim=False).flatten()
                # print(channel_scores.cpu().histogram())
                top_indice = channel_scores.argsort(0, descending=True)[:top_count]
                sub_masks = activations[:, top_indice]  # only these will be computed
                sub_masks = sub_masks.permute(1, 0, 2, 3)  # run as mini batch
                sub_scores = torch.zeros((top_count,)).cuda()
                for batch_start in range(0, top_count, self.parallel_batch):
                    batch_stop_excluded = batch_start + self.parallel_batch
                    if batch_stop_excluded > top_count:
                        batch_stop_excluded = top_count
                    batch_mask = sub_masks[batch_start:batch_stop_excluded]
                    # upsampling
                    batch_mask = nf.interpolate(batch_mask, size=x.shape[2:], mode='bilinear', align_corners=False)
                    # normalize
                    batch_mask = heatmapNormalizeR_ForEach(batch_mask)
                    # save the score
                    sub_scores[batch_start:batch_stop_excluded] = net_fun(x * batch_mask).flatten()
                if self.norm:
                    sub_scores = nf.softmax(sub_scores, 0)
                elif self.pp:
                    sub_scores = sub_scores/sub_scores.abs().max()
                cam = (sub_scores.reshape(-1, 1, 1, 1) * sub_masks).sum(0, keepdim=True)
                # cam = nf.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
                if self.relu:
                    cam = nf.relu(cam)
                hms.append(cam)
            cam = multi_interpolate(hms)
        return cam

    def __del__(self):
        # clear hooks
        for layer in self.layers:
            layer.activation = None
            layer.gradient = None
        self.model.zero_grad(set_to_none=True)
        clearHooks(self.model)

