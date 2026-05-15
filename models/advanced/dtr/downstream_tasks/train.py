from dataset import CustomDataset, get_transform
from model import DownstreamModel
import os
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

def argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Train Downstream Model")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of images')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with labels')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save models and logs')
    parser.add_argument('--mode', type=str, choices=['HE', 'HE&ICHC'], required=True, help='Model mode: HE or HE&ICHC')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    
    args = parser.parse_args()
    return args

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
            elif mode == 'HE&ICHC':
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
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist()
    }
    return results


def main():
    args = argparse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_transform = get_transform('train')
    val_transform = get_transform('val')
    
    train_dataset = CustomDataset(args.data_root, args.csv_file, mode='train', transform=train_transform)
    val_dataset = CustomDataset(args.data_root, args.csv_file, mode='val', transform=val_transform)
    test_dataset = CustomDataset(args.data_root, args.csv_file, mode='test', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = DownstreamModel(num_classes=len(set(train_dataset.data['label'])))
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    # get the trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for he, ihc, labels in train_pbar:
            he, ihc, labels = he.to(device), ihc.to(device), labels.to(device)
            if args.mode == 'HE':
                ihc = torch.zeros_like(ihc).to(device)
            elif args.mode == 'HE&ICHC':
                pass
            else:
                raise ValueError("Invalid mode. Choose from ['HE', 'HE&ICHC']")
            
            outputs = model(he, ihc)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update training metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar description
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        
        # Validation step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]', leave=False)
        with torch.no_grad():
            for he, ihc, labels in val_pbar:
                he, ihc, labels = he.to(device), ihc.to(device), labels.to(device)
                if args.mode == 'HE':
                    ihc = torch.zeros_like(ihc).to(device)
                elif args.mode == 'HE&ICHC':
                    pass
                else:
                    raise ValueError("Invalid mode. Choose from ['HE', 'HE&ICHC']")
                
                outputs = model(he, ihc)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_downstream_model.pth'))
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
    
    # Test the best model
    print("Loading best model for testing...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_downstream_model.pth')))
    test_results = validate(model, test_loader, device, args.mode)
    
    print("\nTest Results:")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"F1-Score: {test_results['f1_score']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    
    # save test results to json file
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"Test results saved to {os.path.join(args.output_dir, 'test_results.json')}")

if __name__ == '__main__':
    main()