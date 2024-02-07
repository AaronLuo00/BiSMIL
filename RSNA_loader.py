from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import DataLoader, random_split
from utils import *

def load_images_from_new_dir(directory, transform=None):
    images_dict = {}
    for patient_folder in os.listdir(directory):
        patient_path = os.path.join(directory, patient_folder)

        # Ensure it's a directory and contains an underscore (to avoid potential issues)
        if os.path.isdir(patient_path) and "_" in patient_folder:
            label = int(patient_folder.split("_")[0])  # Label is before the first "_"
            bag_id = patient_folder.split("_")[1]  # Patient ID after the first "_"

            for fname in os.listdir(patient_path):
                if fname.endswith('.png'):
                    sequence_digits = int(fname.split("_")[0])  # first part before '_' as sequence digits

                    if bag_id not in images_dict:
                        images_dict[bag_id] = {'images': [], 'labels': [], 'sequence_digits': []}

                    img_path = os.path.join(patient_path, fname)
                    img = Image.open(img_path).convert('RGB')

                    # Apply transformation at load time
                    if transform:
                        img = transform(img)

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

class MIDataset_RSNA(Dataset):
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

root_dir = './data_rsna/'
train_dir = os.path.join(root_dir, "train")
test_dir = os.path.join(root_dir, "test")

# Load images from new directories
train_images_dict = load_images_from_new_dir(train_dir)
test_images_dict = load_images_from_new_dir(test_dir)

train_pos_dict = {k: v for k, v in train_images_dict.items() if v['labels'][0] == 1}
train_neg_dict = {k: v for k, v in train_images_dict.items() if v['labels'][0] == 0}
test_pos_dict = {k: v for k, v in test_images_dict.items() if v['labels'][0] == 1}
test_neg_dict = {k: v for k, v in test_images_dict.items() if v['labels'][0] == 0}

transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # For pretrain on the Imagenet
    ])


train_dataset = MIDataset_RSNA(train_pos_dict, train_neg_dict, transform)
test_dataset = MIDataset_RSNA(test_pos_dict, test_neg_dict, transform)

seed = 0 
seed_everything(seed)
# Calculate the sizes of the new training and validation sets
train_size = int(0.875 * len(train_dataset))
validation_size = len(train_dataset) - train_size

# Split the original training dataset into new training and validation datasets
train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

# Create DataLoaders for the new training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

torch.save(train_dataset, './RSNA/RSNA_train_dataset.pt')
torch.save(validation_dataset, './RSNA/RSNA_validation_dataset.pt')
torch.save(test_dataset, './RSNA/RSNA_test_dataset.pt')

# Save the DataLoaders
torch.save(train_loader, './RSNA/RSNA_train_loader.pt')
torch.save(validation_loader, './RSNA/RSNA_validation_loader.pt')
torch.save(test_loader, './RSNA/RSNA_test_loader.pt')
