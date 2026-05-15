import torchvision
import pandas as pd
from PIL import Image
import os
import torch


def get_transform(mode: str):
    if mode == 'train':
        transform = torchvision.transforms.Compose([
            # resize
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ])

    return transform



class CustomDataset(torchvision.datasets.VisionDataset):
    def __init__(self, data_root, csv_file, mode, transform=None):
        self.data_root = data_root
        self.csv_file = csv_file
        self.mode = mode
        self.transform = transform
        self.data = self._load_data()
        print('{} dataset size: {}'.format(mode, len(self.data)))
    
    def _load_data(self):
        data = []
        df = pd.read_csv(self.csv_file)
        df = df[df['split'] == self.mode]
        # convert the label_name to label index
        label_names = sorted(df['label'].unique())
        label_to_index = {name: idx for idx, name in enumerate(label_names)}
        temp = {v:k for k,v in label_to_index.items()}
        print('Label mapping:')
        for i in range(len(temp)):
            print(f"{i}: {temp[i]}")
        df['label'] = df['label'].map(label_to_index)
        # Assuming the CSV has columns 'filepath' and 'label', convert to lists, such [[filepath1, label1], [filepath2, label2], ...]
        data = df[['filepath', 'label']]
        return data.reset_index(drop=True)
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.data.iloc[idx]['filepath'])
        label = self.data.iloc[idx]['label']
        image = Image.open(img_path).convert('RGB')
        # the image is concatenated horizontally, split it into two images he and ihc
        he = image.crop((0, 0, image.width // 2, image.height))
        ihc = image.crop((image.width // 2, 0, image.width, image.height))
        
        if self.transform:
            he = self.transform(he)
            ihc = self.transform(ihc)
        return he, ihc, label



class ExtractedDataset(torchvision.datasets.VisionDataset):
    def __init__(self, pth_file, mode):
        self.mode = mode
        self.data = self._load_data(pth_file)
        print('{} dataset size: {}'.format(mode, len(self.data['labels'])))
    
    def _load_data(self, pth_file):
        data = torch.load(pth_file)
        data = data[self.mode]
        return data
        
        
    def __len__(self):
        return len(self.data['labels'])
    
    def __getitem__(self, idx):

        label = self.data['labels'][idx]
        he = self.data['he_features'][idx]
        ihc = self.data['ihc_features'][idx]
        return he, ihc, label

if __name__ == "__main__":
    data_root = '/jhcnas4/lwq/virtual_staining/downstream_tasks/predictions/GCHTID'
    csv_file = '/jhcnas3/VirtualStaining/downstream_tasks/labels/GCHTID_labels.csv'
    

    train_transform = get_transform('train')
    transform = get_transform('val')
    train_dataset = CustomDataset(data_root, csv_file, mode='train', transform=train_transform)
    val_dataset = CustomDataset(data_root, csv_file, mode='val', transform=transform)
    test_dataset = CustomDataset(data_root, csv_file, mode='test', transform=transform)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    he, ihc, label = train_dataset[0]
    print(f"HE image shape: {he.shape}, IHC image shape: {ihc.shape}, Label: {label}")
    # save to verify
    torchvision.utils.save_image(he, 'he_sample.png')
    torchvision.utils.save_image(ihc, 'ihc_sample.png')