import argparse
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import os.path as osp
import torch

from tqdm import tqdm
from libs.models import ResNet50 as model
# from libs.models import ResNet18 as model
from libs.datasets import *


parser = argparse.ArgumentParser(description='Get Features')
parser.add_argument('--dataset', default='cub200')
parser.add_argument('--ckpt', default='')
parser.add_argument('--batchsize', default=128)
parser.add_argument('--device', default='cuda', help='cpu / cuda')
parser.add_argument('--input_size', default=(224, 224), help='input image size')
args = parser.parse_args()


if __name__ == '__main__':

    means = [0.485, 0.456, 0.406]    # imagenet means
    stds = [0.229, 0.224, 0.225]     # imagenet stds

    transform = transforms.Compose([transforms.Resize(size=args.input_size), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(means, stds)])
       
    names = []
    feats = []

    save_dir = 'features'

    if args.dataset == 'cub200':
        root_dir = 'Datasets/cub200/images'
        dataset = Cub200Dataset(root_dir, transform, is_validation=True)

    elif args.dataset == 'cars196':
        root_dir = 'Datasets/cars196/images'
        dataset = Cars196Dataset(root_dir, transform, is_validation=True)

    elif args.dataset == 'imagenet2012':
        root_dir = 'Datasets/imagenet2012/images'
        dataset = Imagenet2012Dataset(root_dir, transform)

    else:
        print('You need to define your dataset.')
        sys.exit()

    model = model()

    if os.path.exists(args.ckpt):
        print('Loading checkpoints from {} ...'.format(args.ckpt))
        state_dict = torch.load(args.ckpt)['state_dict']
        model.load_state_dict(state_dict)
        print('done')
    elif args.ckpt != '':
        print('No model checkpoint at {}!'.format(args.ckpt))

    model = model.to(args.device)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)
    data_iterator = tqdm(dataloader, desc='Inference')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model.eval()
    for (path, img) in data_iterator:
        img = img.to(args.device)

        with torch.no_grad():
            l2_feats = model(img).squeeze()
        
        names.extend(path)
        feats.extend(l2_feats.cpu().numpy())

    np.save(osp.join(save_dir, 'feats_{}.npy'.format(args.dataset)), feats)
    np.save(osp.join(save_dir, 'names_{}.npy'.format(args.dataset)), names)


