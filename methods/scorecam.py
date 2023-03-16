import torch.nn.functional as nf
from utils import *


class ScoreCAM:
    """
        ScoreCAM is not CAM
    """
    def __init__(self,model, layer=None):
        self.model = model

        # self.features = list(self.model.features)
        # self.flatten = torch.nn.Flatten(1)
        # self.classifier = self.model.classifier

        # def backward_hook(module, grad_input, grad_output):
        #     self.gradients = grad_output[0]
        #     return None

        def forward_hook(module, input, output):
            self.activations = output
            return None

        layer = auto_find_layer(self.model,layer)
        layer.register_forward_hook(forward_hook)
        # self.features[layer].register_backward_hook(backward_hook)

        # config with your gpu
        self.parallel_batch = 64
        # 1     100%|██████████| 10/10 [00:28<00:00,  2.89s/it]
        # 2     100%|██████████| 10/10 [00:22<00:00,  2.28s/it]
        # 8     100%|██████████| 10/10 [00:17<00:00,  1.72s/it]
        # 32    100%|██████████| 10/10 [00:15<00:00,  1.51s/it]
        # 64    100%|██████████| 10/10 [00:13<00:00,  1.36s/it]
        # 128   100%|██████████| 10/10 [00:14<00:00,  1.44s/it]

        # short cut ratio
        self.top_percent = 0.1
        # 10% same quality

    def forward(self, x):
        return self.model(x)

    def __call__(self, x, class_idx=None,
                 sg=True, norm=False, relu=True):
        with torch.no_grad():
            x=x.cuda()
            h, w = x.shape[2:]

            # predication on raw x
            logit = self.model(x.cuda())
            if class_idx is None:
                predicted_class = logit.argmax(1)
            elif isinstance(class_idx, int):
                predicted_class = torch.LongTensor([class_idx]).cuda()
            elif isinstance(class_idx, torch.Tensor):
                predicted_class=class_idx
            else:
                raise Exception()

            # origin version
            if not sg:
                net_fun = lambda x: self.model(x)[:, predicted_class]
            # new version , softmax
            else:
                net_fun = lambda x: nf.softmax(self.model(x), 1)[:, predicted_class]

            # net_fun(x)
            # use activation as masks
            activations = self.activations

            # cut useless activations to sub-mask by sorted mean activation
            top_count = int(self.top_percent * activations.shape[1])
            channel_scores = activations.mean(axis=[2, 3], keepdim=False).flatten()
            # print(channel_scores.cpu().histogram())
            top_indice = channel_scores.argsort(0,descending=True)[:top_count]
            sub_masks = activations[:, top_indice]  # only these will be computed
            sub_masks = sub_masks.permute(1,0,2,3)  # run as mini batch

            sub_scores = torch.zeros((top_count,)).cuda()
            for batch_start in range(0, top_count, self.parallel_batch):
                batch_stop_excluded = batch_start + self.parallel_batch
                if batch_stop_excluded > top_count:
                    batch_stop_excluded = top_count
                batch_mask = sub_masks[batch_start:batch_stop_excluded]
                # upsampling
                batch_mask = nf.interpolate(batch_mask, size=(h, w), mode='bilinear', align_corners=False)
                # normalize
                batch_mask = normalize(batch_mask)
                # save the score
                sub_scores[batch_start:batch_stop_excluded] = net_fun(x * batch_mask).flatten()
            if norm:
                sub_scores = nf.softmax(sub_scores, 0)
            score_cam = (sub_scores.reshape(-1,1,1,1) * sub_masks).sum(0, keepdim=True)
            score_cam = nf.interpolate(score_cam, size=(h, w), mode='bilinear', align_corners=False)
            if relu:
                score_cam = nf.relu(score_cam)
                score_cam = normalize(score_cam)
            else:
                score_cam = normalize_R(score_cam)

        return score_cam
