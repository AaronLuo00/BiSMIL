from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import DataLoader, random_split
from utils import *

def load_images_from_new_dir(directory, label, transform=None):
    images_dict = {}
    for patient in os.listdir(directory):
        patient_path = os.path.join(directory, patient)
        if os.path.isdir(patient_path):
            bag_id = directory.split('/')[-1] + "_" + patient  # category_patient as bag id
            for fname in os.listdir(patient_path):
                if fname.endswith('.png'):
                    sequence_digits = int(fname.split('.')[0])  # extract sequence digits
                    
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
            seq_digits, images, labels = zip(*sorted(zip(images_dict[bag_id]['sequence_digits'],
                                                        images_dict[bag_id]['images'],
                                                        images_dict[bag_id]['labels'])))
            images_dict[bag_id]['sequence_digits'] = list(seq_digits)
            images_dict[bag_id]['images'] = list(images)
            images_dict[bag_id]['labels'] = list(labels)

    return images_dict

class MIDataset_Covid(Dataset):
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


root_dir = '../New_Data_CoV2/'
train_covid_dir = os.path.join(root_dir, "train/Covid")
train_others_dir = os.path.join(root_dir, "train/Others")
train_healthy_dir = os.path.join(root_dir, "train/Healthy")
test_covid_dir = os.path.join(root_dir, "test/Covid")
test_others_dir = os.path.join(root_dir, "test/Others")
test_healthy_dir = os.path.join(root_dir, "test/Healthy")

# Load images from directories
train_covid_dict = load_images_from_new_dir(train_covid_dir, 1)
train_others_dict = load_images_from_new_dir(train_others_dir, 1)
train_healthy_dict = load_images_from_new_dir(train_healthy_dir, 0)

test_covid_dict = load_images_from_new_dir(test_covid_dir, 1)
test_others_dict = load_images_from_new_dir(test_others_dir, 1)
test_healthy_dict = load_images_from_new_dir(test_healthy_dir, 0)

train_pos_dict = {**train_covid_dict, **train_others_dict}  
test_pos_dict = {**test_covid_dict, **test_others_dict}  

transform = transforms.Compose([
        transforms.ToTensor(),
    ])

train_dataset = MIDataset_Covid(train_pos_dict, train_healthy_dict, transform)
test_dataset = MIDataset_Covid(test_pos_dict, test_healthy_dict, transform)

seed = 0 
seed_everything(seed)
# Calculate the sizes of the new training and validation sets
train_size = int(0.875 * len(train_dataset))
validation_size = len(train_dataset) - train_size

# Split the original training dataset into new training and validation datasets
train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

torch.save(train_dataset, './Covid/Covid_train_dataset.pt')
torch.save(validation_dataset, './Covid/Covid_validation_dataset.pt')
torch.save(test_dataset, './Covid/Covid_test_dataset.pt')

# Save the DataLoaders
torch.save(train_loader, './Covid/Covid_train_loader.pt')
torch.save(validation_loader, './Covid/Covid_validation_loader.pt')
torch.save(test_loader, './Covid/Covid_test_loader.pt')