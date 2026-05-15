from model import BackBone
from dataset import CustomDataset, get_transform
import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Extract features from datasets')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to the folder containing input predictions/images')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save extracted features')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='CSV file with labels and split information')
    parser.add_argument('--device', type=str, required=False, default='cuda:2', help='Torch device to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for dataloader')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = args.data_root
    save_dir = args.save_dir
    csv_file = args.csv_file
    device = args.device

    os.makedirs(save_dir, exist_ok=True)
    val_transform = get_transform('val')

    train_dataset = CustomDataset(data_root, csv_file, mode='train', transform=val_transform)
    val_dataset = CustomDataset(data_root, csv_file, mode='val', transform=val_transform)
    test_dataset = CustomDataset(data_root, csv_file, mode='test', transform=val_transform)

    cuda_available = torch.cuda.is_available() and 'cpu' not in device.lower()
    pin_memory = True if cuda_available else False

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)

    model = BackBone()
    model = model.to(device)

    def extract_and_save_features(data_loader):
        results = {}
        model.eval()
        he_features = []
        ihc_features = []
        labels_list = []
        with torch.no_grad():
            for i, (he, ihc, labels) in enumerate(data_loader):
                print(f"Processing batch {i+1}/{len(data_loader)}")
                he, ihc = he.to(device), ihc.to(device)
                he_feature = model(he).cpu()
                ihc_feature = model(ihc).cpu()
                he_features.append(he_feature)
                ihc_features.append(ihc_feature)
                # labels might already be numpy or torch tensor
                try:
                    labels_list.extend(labels)
                except Exception:
                    labels_list.extend([int(x) for x in labels])
        
        he_features = torch.vstack(he_features)
        ihc_features = torch.vstack(ihc_features)
    
        results['he_features'] = he_features
        results['ihc_features'] = ihc_features
        results['labels'] = labels_list
        return results

    train_features = extract_and_save_features(train_loader)
    val_features = extract_and_save_features(val_loader)
    test_features = extract_and_save_features(test_loader)

    final_results = {
        'train': train_features,
        'val': val_features,
        'test': test_features
    }

    # save torch pth file
    out_path = os.path.join(save_dir, 'extracted_features.pth')
    torch.save(final_results, out_path)
    print(f"Saved extracted features to: {out_path}")


if __name__ == '__main__':
    main()