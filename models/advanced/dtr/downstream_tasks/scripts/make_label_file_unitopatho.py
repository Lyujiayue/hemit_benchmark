import os
import random

path = '/storage/jmabq/VirtualStaining/downstream_tasks/UniToPatho'

class_names = os.listdir(path)

train_ratio = 0.7
val_ratio = 0.1
random.seed(42)
train_files = []
val_files = []
test_files = []
for class_name in class_names:
    class_path = os.path.join(path, class_name)
    files = os.listdir(class_path)
    random.shuffle(files)
    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files += [(os.path.join(class_name, f), class_name) for f in files[:n_train]]
    val_files += [(os.path.join(class_name, f), class_name) for f in files[n_train:n_train + n_val]]
    test_files += [(os.path.join(class_name, f), class_name) for f in files[n_train + n_val:]]

with open('/jhcnas3/VirtualStaining/downstream_tasks/labels/UniToPatho_labels.csv', 'w') as f:
    f.write('filepath,label,split\n')
    for filepath, label in train_files:
        f.write(f'"{filepath}",{label},train\n')
    for filepath, label in val_files:
        f.write(f'"{filepath}",{label},val\n')
    for filepath, label in test_files:
        f.write(f'"{filepath}",{label},test\n')