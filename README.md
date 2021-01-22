# ImageRetrievalGUI

---

## UPDATE

2021/01/22 : Support customize model and customize dataset.

---

## Introuction

This is a lightweight GUI for visualizing the Image Retrieval results, and would be convenient for verifying the results and groundtruth. 

[![alt tag](./demo/demo1.jpg)](https://www.youtube.com/watch?v=xUfO5MMAR5M)
[![alt tag](./demo/demo2.jpg)](https://www.youtube.com/watch?v=xUfO5MMAR5M)
[![alt tag](./demo/demo3.gif)](https://www.youtube.com/watch?v=xUfO5MMAR5M)

Demo video : https://www.youtube.com/watch?v=xUfO5MMAR5M

## Repository Structure

```
Repository
├── demo
│   ├── demo1.jpg
│   ├── demo2.jpg
│   └── demo3.jpg
├── extract_feats.py
├── libs
│   ├── datasets.py
│   ├── models.py
│   └── utils.py
├── LICENSE
├── main.py
├── query_images
│   ├── q_001.jpg
│   ├── q_002.jpg
│   ├── q_003.jpg
│   ├── q_004.jpeg
│   ├── q_005.jpg
│   └── q_006.jpg
├── README.md
├── cars196_checkpoint.pth.tar (should be added)
├── cub200_checkpoint.pth.tar (should be added)
└───Datasets (should be added)
|    ├── cub200
|    ├── cars196
```

## Dataset Structures
__CUB200-2011/CARS196__
```
cub200/cars196
└───images
|    └───001.Black_footed_Albatross
|           │   Black_Footed_Albatross_0001_796111
|           │   ...
|    ...
```

---

# Demo


## Clone this repository.

```
git clone https://github.com/Chien-Hung/ImageRetrievalGUI.git
cd ImageRetrievalGUI
```

## Train a model for extracting image feature

For this demo, you should train a resnet50 models for cub200 / cars196 dataset by [Deep-Metric-Learning-Baselines](https://github.com/Confusezius/Deep-Metric-Learning-Baselines). Or you can download the trained checkpoints [cub200_checkpoint.pth.tar](https://drive.google.com/file/d/1Gem3-9mzutHbNtBVQS8yIi_DPOG0YV2S/view?usp=sharing) / [cars196_checkpoint.pth.tar](https://drive.google.com/file/d/1wvP3Engemk9RTwiE6cZJjjjonXLXscEA/view?usp=sharing).

Link to the Deep-Metric-Learning-Baselines datasets or download the cub200 / cars196 dataset in this folder.

```
ln -s Deep-Metric-Learning-Baselines/Datasets ./Datasets
```

## Extract image collection features by the trained model.

This offer the trained checkpoints for demo.

```
python extract_feats.py --dataset cub200 --ckpt cub200_checkpoint.pth.tar
```

```
python extract_feats.py --dataset cars196 --ckpt cars196_checkpoint.pth.tar
```

## Display the results.

```
python main.py --dataset cub200 --ckpt cub200_checkpoint.pth.tar 
```

```
python main.py --dataset cars196 --ckpt cars196_checkpoint.pth.tar
```
---

# An example of customize model and customize dataset.

I use pretrained resnet18 model and sample some images from imagenet2012 training data for example.

## Customize dataset

Place your dataset images in `Datasets` with following structure:

### Dataset Structure
__imagenet2012__
```
imagenet2012
└───images
|    └───n01440764
|           ├── n01440764_18.JPEG
|           ├── ...
|    ...
```

Create your dataset in `libs/dataset.py`:

```python
class Imagenet2012Dataset(Dataset):
    def __init__(self, root_dir, transform):
        self.img_list = []
        for folder in os.listdir(root_dir):

            for img_name in os.listdir(osp.join(root_dir, folder)):
                self.img_list.append(osp.join(root_dir, folder, img_name))
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = self.transform(img)

        return self.img_list[idx], img
```

Add your dataset in `extract_feats.py`:

```python
...
    elif args.dataset == 'imagenet2012':
        root_dir = 'Datasets/imagenet2012/images'
        dataset = Imagenet2012Dataset(root_dir, transform)
...
```

## Customize model

Create your model in `libs/models.py`:

```python
import torchvision.models as models
...

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        
        self.model = models.resnet18(pretrained=True)
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return torch.nn.functional.normalize(x, dim=-1)
```

Change the default model (ResNet50) to your model (ResNet18) in `extract_feats.py` and `main.py`:

```python
...
# from libs.models import ResNet50 as model
from libs.models import ResNet18 as model
...
```

## Extract image collection features by the trained model.

```
python extract_feats.py --dataset imagenet2012
```

## Display the results.

```
python main.py --dataset imagenet2012
```

---

# Hotkeys

|     KEY    | ACTION                                    |
|:----------:|-------------------------------------------|
|   ↑ , ↓    | change image.                              |
|   ← , →    | change tab.                                | 
|     q     | colse this GUI.                            |

---

# Reference  

https://github.com/Confusezius/Deep-Metric-Learning-Baselines
