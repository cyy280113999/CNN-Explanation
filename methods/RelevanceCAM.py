from .LRP import *

"""
    "RelevanceCAM-24": lambda model: lambda x, y: interpolate_to_imgsize(
        RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer_num=24)),
    "RelevanceCAM-17": lambda model: lambda x, y: interpolate_to_imgsize(
        RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer_num=17)),
    "RelevanceCAM-10": lambda model: lambda x, y: interpolate_to_imgsize(
        RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer_num=10)),
    "RelevanceCAM-5": lambda model: lambda x, y: interpolate_to_imgsize(
        RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer_num=5)),
"""

class RelevanceCAM(LRP_Generator):
    def __call__(self, x, yc=None, backward_init='c', method='lrpzp', layer_num=None, device=device):
        # ___________runningCost___________= RunningCost(50)
        layer_num = auto_find_layer_index(self.model, layer_num)
        # forward
        activations = [None] * self.layerlen
        x = x.requires_grad_().to(device)
        activations[0] = x
        for i in range(1, self.layerlen):
            activations[i] = self.layers[i](activations[i - 1])


        logits = activations[self.layerlen - 1]
        if yc is None:
            yc = logits.max(1)[1]
        elif isinstance(yc, int):
            yc = torch.LongTensor([yc]).detach().to(device)
        elif isinstance(yc, torch.Tensor):
            yc=yc.to(device)
        else:
            raise Exception()

        # Gradient backward if required
        target_onehot = nf.one_hot(yc, logits.shape[1]).float().detach()

        # LRP backward
        R = [None] * self.layerlen  # register memory
        if backward_init is None or backward_init == self.available_backward_init.normal:
            R[self.layerlen - 1] = target_onehot * activations[self.layerlen - 1]
        elif backward_init == self.available_backward_init.target_one_hot:
            R[self.layerlen - 1] = target_onehot  # 1 else 0
        elif backward_init == self.available_backward_init.c:
            # (N-1)/N else -1/N
            R[self.layerlen - 1] = target_onehot + torch.full_like(logits, -1 / logits.shape[1])
        elif backward_init == self.available_backward_init.sg:
            # (1-p_t)p else -p_t*p
            prob = nf.softmax(logits, 1)
            prob_t = prob[0, yc]
            R[self.layerlen - 1] = (target_onehot - prob_t) * prob
        elif backward_init == self.available_backward_init.st:
            prob = nf.softmax(logits, 1)
            prob_t = prob[0, yc]
            R[self.layerlen - 1] = activations[self.layerlen - 1] * (target_onehot - prob_t) * prob
        elif backward_init == self.available_backward_init.sig0:
            sig = torch.zeros_like(logits)
            sample_scale = 0.1
            dx = sample_scale * logits
            lin_samples = sample_scale + torch.arange(0, 1, sample_scale, device=device)
            # lines = []
            # fig=plt.figure()
            # axe=fig.add_subplot()
            for scale_multiplier in lin_samples:
                lin_point = scale_multiplier * logits
                prob = nf.softmax(lin_point, 1)
                prob_t = prob[0, yc]
                sgc = (target_onehot - prob_t) * prob
                sig += sgc * dx
            #     lines.append(sig.cpu().clone().detach())
            # lines = torch.vstack(lines).numpy().T
            # axe.plot(lines)
            # axe.set_title(yc.item())
            R[self.layerlen - 1] = sig
        elif backward_init == self.available_backward_init.sigp:
            sig = torch.zeros_like(logits)
            sample_scale = 0.1
            dx = torch.zeros_like(logits)
            dx[0, yc] = logits[0, yc] * sample_scale
            lin_samples = sample_scale + torch.arange(0, 1, sample_scale, device=device)
            for scale_multiplier in lin_samples:
                lin_point = logits.clone()
                lin_point[0, yc] *= scale_multiplier
                prob = nf.softmax(lin_point, 1)
                prob_t = prob[0, yc]
                sgc = (target_onehot - prob_t) * prob
                # sig += sgc * dx
                sig += sgc * sample_scale
            R[self.layerlen - 1] = sig
        else:
            raise Exception(f'Not Valid Method {backward_init}')
        # ___________runningCost___________.tic('last layer')
        if layer_num is None:
            _stop_at = 0
        else:
            _stop_at = layer_num
        for i in range(_stop_at + 1, self.layerlen)[::-1]:
            if method == self.available_layer_method.lrp0:
                xs, funs = lrp0(activations[i - 1])
            elif method == self.available_layer_method.lrpz:
                xs, funs = lrpz(activations[i - 1])
            elif method == self.available_layer_method.lrpc:
                xs, funs = lrpc(i, activations[i - 1], flat_loc=self.flat_loc)
            elif method == self.available_layer_method.lrpzp:
                xs, funs = lrpzp(activations[i - 1])
            elif method == self.available_layer_method.slrp:
                xs, funs = lrpzp(activations[i - 1])
            else:
                raise Exception
            R[i - 1] = LRP_layer(self.layers[i], R[i], xs, funs)
        return (R[layer_num].sum([2, 3], True) * activations[layer_num]).sum(1, True)