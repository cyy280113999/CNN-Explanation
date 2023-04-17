import argparse
from functools import partial
from tqdm import tqdm
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as nf
import torch.utils.data as td

from utils import *
from methods.LRP import LRP_Generator
from methods.cam.gradcam import GradCAM



def main(model_name = 'vgg16',heatmap_name = 'gradcam'):
    models = {
        # "None": lambda: None,
        "vgg16": lambda: tv.models.vgg16(weights=tv.models.VGG16_Weights.DEFAULT),
        "alexnet": lambda: tv.models.alexnet(weights=tv.models.AlexNet_Weights.DEFAULT),
        "googlenet": lambda: tv.models.googlenet(weights=tv.models.GoogLeNet_Weights.DEFAULT),
        "resnet18": lambda: tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT),
        "resnet50": lambda: tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT),
    }
    cam_model_dict_by_layer = lambda layer: {'type': model_name, 'arch': model, 'layer_name': f'{layer}',
                                             'input_size': (224, 224)}
    interpolate_to_imgsize = lambda x: nf.interpolate(x.sum(1, True), 224, mode='bilinear')

    heatmap_methods = {
        "gradcam": lambda: partial(GradCAM(cam_model_dict_by_layer(-1)).__call__,
                                     sg=False, relu=False),
        "sggradcam": lambda: partial(GradCAM(cam_model_dict_by_layer(-1)).__call__,
                                        sg=True, relu=False),
        "sig0lrpc": lambda: lambda x, y: interpolate_to_imgsize(
            LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=-1)),
    }

    ds = getImageNet('val')

    # #  17%|█▋        | 8357/50000 [01:01<04:47, 144.69it/s]
    # for img_path,img_label in tqdm(ds.imgs):
    #     class_name, img_name = img_path.split('\\')[-2:]
    #     img_name, img_suffix = img_name.split('.')
    #     x = pilOpen(img_path)
    #     x = default_transform(x)
    #     x=x.cuda()
    #     pass

    # dl = td.DataLoader(ds, batch_size=1, shuffle=False, num_workers=8,
    #                    pin_memory=True)
    # # dl
    # 0:
    # #   3%|▎         | 1320/50000 [00:10<07:06, 114.17it/s]
    # 2:
    #  15%|█▌        | 7556/50000 [00:46<02:33, 275.93it/s]
    # 8:
    #  57%|█████▋    | 28278/50000 [01:44<00:42, 516.35it/s]

    # for x,y in tqdm(dl):
    #     x = x.cuda()
    #     pass

    model = models[model_name]().eval().cuda()

    heatmap_method = heatmap_methods[heatmap_name]()

    save_pre_dir = 'F:/DataSet/imagenet/heatmaps/'
    save_dir = save_pre_dir + heatmap_name +'_'+ model_name +'\\'

    for img_path, img_label in tqdm(ds.imgs):
        # _____rc_____ = RunningCost(10)
        # _____rc_____.tic()
        class_name, img_name = img_path.split('\\')[-2:]
        img_name, img_suffix = img_name.split('.')
        x = pilOpen(img_path)
        x = default_transform(x).unsqueeze(0)
        x=x.cuda()
        y=img_label
        # _____rc_____.tic('load to gpu')
        hm=heatmap_method(x,y)
        hm=nf.relu(hm)
        hm=heatmapNormalizeR(hm)
        hm=hm.squeeze(0).squeeze(0)
        # _____rc_____.tic('get heatmap')
        hm_dir = save_dir+class_name+'\\'
        mkp(hm_dir)
        np.save(hm_dir +img_name+'.npy',hm.detach().cpu().numpy())
        # _____rc_____.tic('saved')
        # _____rc_____.cost()
        pass

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--method')
    args=ap.parse_args()
    main(args.model,args.method)
