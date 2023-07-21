import torch.utils.data as TD
import os
import sys
sys.path.append('../')
from utils import *
from tqdm import tqdm

"""
imgnt subset where samples are predicted failed by specific model but in top5 list.
"""
def generate_abs_filename(fn):
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    file_to_read = os.path.join(current_directory, fn)
    return file_to_read
class ImgntTop5(TD.Dataset):
    def __init__(self, model=None, model_type=''):
        self.imgnt = getImageNet('val')
        fn='ImgntTop5_'+model_type+'.npy'
        fn = generate_abs_filename(fn)
        if os.path.exists(fn):
            self.indices = np.load(fn)
        else:
            self.model=model
            self.indices = self.createIndex()
            np.save(fn,self.indices)

    def createIndex(self):
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

    def __getitem__(self, i):
        return self.imgnt[self.indices[i]]

    def __len__(self):
        return len(self.indices)

if __name__ == '__main__':
    model_type='vgg16'  # len 9482
    model=get_model(model_type)
    ds=ImgntTop5(model,model_type)
    print(len(ds))
