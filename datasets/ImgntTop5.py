from tqdm import tqdm
import torch.utils.data as TD
from utils import *

"""
imgnt subset where samples are predicted failed by specific model but in top5 list.
"""


class ImgntTop5(SubSetFromIndices):
    def __init__(self, model_type=''):
        self.imgnt=getImageNet('val')
        self.indices=None
        fn='ImgntTop5_'+model_type+'.npy'
        fn = generate_abs_filename(__file__, fn)
        self.loadIndices(fn)
        super().__init__(self.imgnt,self.indices)

    def loadIndices(self, fn):
        if os.path.exists(fn):
            self.indices = np.load(fn)
        else:
            self.model=model
            self.indices = self.createIndices()
            np.save(fn,self.indices)

    def createIndices(self):
        assert self.model is not None
        indices=[]
        for i in tqdm(range(len(self.imgnt)), desc='create indices'):
            x, y = self.imgnt[i]
            out=self.model(x.unsqueeze(0).to(device))
            out = out.argsort(dim=1,descending=True)[0, 1:5].tolist()
            if y in out:
                indices.append(i)
        indices=np.asarray(indices)
        tqdm.write(f'len: {len(indices)}')
        return indices





if __name__ == '__main__':
    model_type='vgg16'  # len 9482
    model=get_model(model_type)
    ds=ImgntTop5(model,model_type)
    print(len(ds))
