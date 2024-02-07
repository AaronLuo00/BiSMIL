import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MIDataset(Dataset):
    """
    A custom dataset class for handling medical images.

    Attributes:
    pos_dict (dict): A dictionary of positive samples.
    neg_dict (dict): A dictionary of negative samples.
    transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
    """
    def __init__(self, pos_dict, neg_dict, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        self.bag_ids = []
        self.sequence_digits = []

        # Combine positive and negative dictionaries
        for bag_id, bag in {**pos_dict, **neg_dict}.items():
            self.data.append(bag['paths'])
            self.labels.append(1 if bag_id in pos_dict else 0)
            self.bag_ids.append(bag_id)
            self.sequence_digits.append(bag['sequence_digits'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bag_images = [Image.open(img_path).convert('RGB') for img_path in self.data[idx]]

        if self.transform:
            bag_images = [self.transform(img) for img in bag_images]

        bag_images = torch.stack(bag_images)
        return bag_images, self.labels[idx], self.bag_ids[idx], self.sequence_digits[idx]

def save_training_variables(file_name, variables):
    """
    Saves training variables to a file.
    """
    with open(file_name, 'wb') as f:
        pickle.dump(variables, f)

def save_training_plots(step_losses, train_acc, test_acc, val_acc, best_test_acc, exp_name, exp_path):
    """
    Saves training plot images with annotations for highest test and validation accuracies and corresponding epochs.
    """
    # Finding the epoch numbers for highest test and validation accuracies
    highest_test_acc_epoch = test_acc.index(max(test_acc))
    highest_val_acc_epoch = val_acc.index(max(val_acc))

    # Plotting
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13.5, 5))

    # Plotting step losses
    axs[0].plot(step_losses, label="Avg Train Loss", linewidth=2, color='royalblue')
    axs[0].set_title("Cumulative Average Training Loss")
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Cumulative Average Loss")
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()

    # Plotting accuracies
    axs[1].plot(train_acc, label='Train Accuracy', linewidth=2, color='darkgreen')
    axs[1].plot(test_acc, label='Test Accuracy', linewidth=2, color='crimson')
    axs[1].plot(val_acc, label='Val Accuracy', linewidth=2, color='darkblue')
    axs[1].set_title("Accuracy vs Epochs")
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()

    # Annotating the highest test accuracy and epoch
    axs[1].annotate(f'Acc: {max(test_acc):.4f}\nEpoch: {highest_test_acc_epoch}',
                    xy=(highest_test_acc_epoch, max(test_acc)), 
                    textcoords="offset points", xytext=(10,-10), fontsize=8)

    # Annotating the highest validation accuracy and epoch
    axs[1].annotate(f'Acc: {max(val_acc):.4f}\nEpoch: {highest_val_acc_epoch}',
                    xy=(highest_val_acc_epoch, max(val_acc)), 
                    textcoords="offset points", xytext=(10,-10), fontsize=8)

    # Annotating test accuracy and epoch at the epoch of highest validation accuracy
    axs[1].annotate(f'Acc: {test_acc[highest_val_acc_epoch]:.4f}\nEpoch: {highest_val_acc_epoch}',
                    xy=(highest_val_acc_epoch, test_acc[highest_val_acc_epoch]), 
                    textcoords="offset points", xytext=(10,-10), fontsize=8)

    # Finalizing plot
    fig.suptitle(f'Experiment Name: {exp_name}, Highest Test Accuracy: {best_test_acc:.4f}', fontsize=16)
    plt.savefig(exp_path + f'{exp_name}_plot.png')
    print("/n Successfully save fig")
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)



def load_images_from_dir_image(directory, label):
    images_dict = {}
    for fname in os.listdir(directory):
        if fname.endswith('.jpeg'):
            bag_id = fname.split('_')[0]  # first part before '_' as bag id
            sequence_digits = int(fname.split('_')[2].split(
                '-')[0])  # extract sequence digits

            if bag_id not in images_dict:
                images_dict[bag_id] = {'images': [],
                                       'labels': [], 'sequence_digits': []}

            img_path = os.path.join(directory, fname)
            img = Image.open(img_path).convert('RGB')  # Load and convert image here

            images_dict[bag_id]['images'].append(img)
            images_dict[bag_id]['labels'].append(label)
            images_dict[bag_id]['sequence_digits'].append(sequence_digits)

    # Sort images by sequence digits within each bag
    for bag_id in images_dict:
        seq_digits, images, labels = zip(*sorted(zip(images_dict[bag_id]['sequence_digits'],
                                                     images_dict[bag_id]['images'],
                                                     images_dict[bag_id]['labels'])))
        images_dict[bag_id]['sequence_digits'] = list(seq_digits)
        images_dict[bag_id]['images'] = list(images)
        images_dict[bag_id]['labels'] = list(labels)

    return images_dict


class MIDataset_image(Dataset):
    """
    A custom dataset class for handling medical images.

    Attributes:
    pos_dict (dict): A dictionary of positive samples.
    neg_dict (dict): A dictionary of negative samples.
    transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
    """
    def __init__(self, pos_dict, neg_dict, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        self.bag_ids = []
        self.sequence_digits = []

        # Combine positive and negative dictionaries
        for bag_id, bag in {**pos_dict, **neg_dict}.items():
            self.data.append(bag['images'])  # Store images directly
            self.labels.append(1 if bag_id in pos_dict else 0)
            self.bag_ids.append(bag_id)
            self.sequence_digits.append(bag['sequence_digits'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bag_images = self.data[idx]

        if self.transform:
            bag_images = [self.transform(img) for img in bag_images]

        bag_images = torch.stack(bag_images)
        return bag_images, self.labels[idx], self.bag_ids[idx], self.sequence_digits[idx]


def save_to_excel(experiment_results, file_path):
    if not os.path.isfile(file_path):
        df = pd.DataFrame(columns=["Experiment Name", "Seed Number", "Model Name", "Highest Test ACC", "Highest Val ACC", "Test ACC for Highest Val ACC"])
    else:
        df = pd.read_excel(file_path, index_col=0)
    
    new_df = pd.DataFrame([experiment_results])
    df = pd.concat([df, new_df], ignore_index=True)
    
    df.to_excel(file_path)

def count_parameters_and_proportion(model):
    total_params = sum(p.numel() for p in model.parameters())
    params_info = {name: {'params': sum(p.numel() for p in module.parameters()),
                          'proportion (Percent)': sum(p.numel() for p in module.parameters()) / total_params * 100 }
                   for name, module in model.named_children()}
    return params_info


def save_checkpoint(model, path, is_best=False):
    model.cpu()  
    torch.save(model.state_dict(), path)
    model.to(device)  
    if is_best:
        print(f"Saved best model at {path}")


def save_training_artifacts(path, variables):
    with open(path + 'training_variables.pkl', 'wb') as f:
        pickle.dump(variables, f)
        
def update_misclassified_samples(misclassified_samples_count, misclassified_samples):
    for epoch, bag_id, predicted, label in misclassified_samples:
        if bag_id in misclassified_samples_count:
            misclassified_samples_count[bag_id] += 1
        else:
            misclassified_samples_count[bag_id] = 1