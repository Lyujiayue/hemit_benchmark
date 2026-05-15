from dataset import ExtractedDataset
from model import LinearProbeModel
import os
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np


def argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Train Downstream Model")
    parser.add_argument('--pth_file', type=str, required=True, help='Path to the .pth file with extracted features')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save models and logs')
    parser.add_argument('--mode', type=str, choices=['HE', 'HE&IHC'], required=True, help='Model mode: HE or HE&ICHC')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    return args

def bootstrap_confidence_interval(metric_fn, preds, labels, n_bootstraps=1000, ci=0.95):
    rng = np.random.RandomState(42)
    bootstrapped_metrics = []
    preds = np.array(preds)
    labels = np.array(labels)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue
        score = metric_fn(labels[indices], preds[indices])
        bootstrapped_metrics.append(score)
    sorted_metrics = np.sort(bootstrapped_metrics)
    lower_bound = ((1.0 - ci) / 2.0) * 100
    upper_bound = (ci + ((1.0 - ci) / 2.0)) * 100
    lower = sorted_metrics[int(lower_bound / 100.0 * len(sorted_metrics))]
    upper = sorted_metrics[int(upper_bound / 100.0 * len(sorted_metrics))]
    return lower, upper


def validate(model, test_loader, device, mode):
    # calulate accuracy, F1-score, precision, recall, and confusion matrix. save the results to json file
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for he, ihc, labels in tqdm(test_loader, desc="Validating", leave=False):
            he, ihc, labels = he.to(device), ihc.to(device), labels.to(device)
            if mode == 'HE':
                ihc = torch.zeros_like(ihc).to(device)
            elif mode == 'HE&IHC':
                pass
            else:
                raise ValueError("Invalid mode. Choose from ['HE', 'HE&ICHC']")
            
            outputs = model(he, ihc)
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    # bootstrap to calculate 95% confidence interval for accuracy
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    bootstrapped_accuracies = []
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(all_labels), len(all_labels))
        if len(np.unique(all_labels[indices])) < 2:
            continue
        score = accuracy_score(all_labels[indices], all_preds[indices])
        bootstrapped_accuracies.append(score)
    sorted_accuracies = np.sort(bootstrapped_accuracies)
    lower = sorted_accuracies[int(0.025 * len(sorted_accuracies))]
    upper = sorted_accuracies[int(0.975 * len(sorted_accuracies))]
    print(f"95% confidence interval for accuracy: [{lower:.4f} - {upper:.4f}]")
    # f1, precision, recall can also have confidence intervals calculated similarly if needed
    f1_lower, f1_upper = bootstrap_confidence_interval(f1_score, all_preds, all_labels)
    precision_lower, precision_upper = bootstrap_confidence_interval(precision_score, all_preds, all_labels)
    recall_lower, recall_upper = bootstrap_confidence_interval(recall_score, all_preds, all_labels)
    print(f"95% confidence interval for F1-score: [{f1_lower:.4f} - {f1_upper:.4f}]")
    print(f"95% confidence interval for Precision: [{precision_lower:.4f} - {precision_upper:.4f}]")
    print(f"95% confidence interval for Recall: [{recall_lower:.4f} - {recall_upper:.4f}]")
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'f1_ci_lower': f1_lower,
        'f1_ci_upper': f1_upper,
        'accuracy_ci_lower': lower,
        'accuracy_ci_upper': upper,
        'precision': precision,
        'precision_ci_lower': precision_lower,
        'precision_ci_upper': precision_upper,
        'recall': recall,
        'recall_ci_lower': recall_lower,
        'recall_ci_upper': recall_upper,
        'confusion_matrix': cm.tolist()
    }
    return results


def main():
    args = argparse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    test_dataset = ExtractedDataset(args.pth_file, 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    n_class = torch.stack(test_dataset.data['labels']).max().item() + 1
    print(n_class)
    model = LinearProbeModel(num_classes=n_class)
    ckpt_path = os.path.join(args.output_dir, 'best_downstream_model.pth')
    msg = model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("Loaded model weights from {} with msg: {}".format(ckpt_path, msg))
    model = model.to(device)
    test_results = validate(model, test_loader, device, args.mode)
    
    print("\nTest Results:")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"F1-Score: {test_results['f1_score']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    
    # save test results to json file
    with open(os.path.join(args.output_dir, 'test_results_ci.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"Test results saved to {os.path.join(args.output_dir, 'test_results_ci.json')}")

if __name__ == '__main__':
    main()