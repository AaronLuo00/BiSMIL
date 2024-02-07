import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model import *
from utils import *
from train_and_test import initialize_training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


weight_s = 0
weight_o = 0
weight_x = 1.5

def incremental_predict(model, bag, incremental_ratio=0.5):
    """
    Incrementally predicts using a model on subsets of a bag.
    """
    outputs = []
    total_len = bag.shape[1]
    index = math.ceil(total_len * incremental_ratio)

    for i in range(index, total_len + 1):
        # Extract subset of instances from the bag
        subset_bag = bag[:, :i, ...]
        output = model(subset_bag, total_len=bag.shape[1])
        outputs.append(output)

    return outputs

def construct_sequence(n, temperature=2.0):
    """
    Constructs a softmax-weighted sequence.
    """
    sequence = np.arange(1, n + 1)
    e_x = np.exp((sequence - np.max(sequence)) / temperature)
    softmax_sequence = e_x / e_x.sum()
    return softmax_sequence

def compute_weighted_incremental_loss(outputs, labels, softmax_sequence):
    """
    Computes the weighted incremental loss for a list of outputs.
    """
    loss = nn.BCELoss()
    losses = []

    for i, output in enumerate(outputs):
        instance_loss = loss(output, labels)
        weighted_instance_loss = instance_loss * softmax_sequence[i]
        losses.append(weighted_instance_loss)

    return sum(losses)


def SMILU_metric(bag_outputs_dict, ratio=1, weight_s = weight_s, weight_o = weight_o, temperature=2):   
    custom_ranges = {}
    for bag_id, instance_outputs in bag_outputs_dict.items():
        if len(instance_outputs) == 1:  # if bag length is 1 or 2
            U_S = np.mean([min(abs(x - 0), abs(x - 1)) for x in instance_outputs])
            U_O = instance_outputs[0] * (1-instance_outputs[0])
            SMILU = U_S * weight_s + U_O * weight_o
        else:  # if bag length is greater than 2
            U_S = np.std(instance_outputs)
            abs_diff = np.abs(np.diff(instance_outputs))
            weights_diff = construct_sequence(len(abs_diff), temperature=temperature)
            U_O = np.average(abs_diff, weights=weights_diff)
            SMILU = U_S * weight_s + U_O * weight_o
    return SMILU



def calculate_entropy(probabilities):
    """
    Calculate the entropy of a list of probabilities.
    """
    # Filter out zero probabilities as log2(0) is not defined
    probabilities = [p for p in probabilities if p > 0]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy

def calculate_entropy_for_bags(bag_outputs_dict):
    """
    Calculate the entropy for each bag in the bag_outputs_dict.
    The function binarizes the outputs (greater than 0.5 to 1, otherwise 0)
    and then calculates the entropy.
    """
    entropy_dict = {}
    for bag_id, outputs in bag_outputs_dict.items():
        # Binarize the outputs
        binarized_outputs = [1 if output > 0.5 else 0 for output in outputs]
        # Calculate the frequency of each outcome (0 or 1)
        frequencies = np.bincount(binarized_outputs) / len(binarized_outputs)
        # Calculate the entropy
        entropy = calculate_entropy(frequencies)
        entropy_dict[bag_id] = entropy
    return entropy_dict




def incremental_predict_test(model, bag):
    # Initialize list to store outputs
    outputs = []

    # Compute total length of bag
    total_len = bag.shape[1]

    # Loop over instances in bag
    for i in range(1, total_len+1):
        subset_bag = bag[:, :i, ...]
        bag_feature = model(subset_bag)[-1]

        if len(bag_feature.shape) == 0:
            bag_feature = bag_feature.unsqueeze(0)
        output = bag_feature

        # Append output to list
        outputs.append(output.item())

    return outputs


def plot_uncertainty_histogram(exp_path, test_dataset, trans = True, uncertainty_metric=SMILU_metric):
    # Load the model
    model = BiSMIL()
    model.load_state_dict(torch.load(exp_path + "model_best.pth"))
    
    # Initialize test data loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    incremental_outputs = []  
    bag_id_list = []
    model.eval()  

    # Get incremental outputs for all instances in all bags
    with torch.no_grad():  
        for i, (bag, label, bag_id,  bag_seq_digits) in enumerate(tqdm(test_loader)):
            label = label.float()
            total_len = bag.shape[1]
            bag_incremental_outputs = incremental_predict_test(model, bag)

            
            incremental_outputs.append(bag_incremental_outputs)
            bag_id_list.append(bag_id)

    # Compute uncertainty scores
    bag_incremental_outputs_dict = {bag_id: inc_out for bag_id, inc_out in zip(bag_id_list, incremental_outputs)}
    uncertainty_scores = uncertainty_metric(bag_incremental_outputs_dict)

    # Get scores and calculate mean
    scores = list(uncertainty_scores.values())
    mean_scores = np.mean(scores)
    
    # Create histogram
    plt.hist(scores, bins=20, alpha=0.5, color='g')
    plt.axvline(mean_scores, color='r', linestyle='dashed', linewidth=2)
    plt.text(mean_scores, plt.ylim()[1] * 0.9, 'Mean: {:.2f}'.format(mean_scores), 
            horizontalalignment='center', fontsize=12, color='r')
    plt.title('Histogram of Uncertainty Metric')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()



def load_model_for_seed(experiment_name, seed, configuration, base_path="/outputs/"):
    config = configuration.copy()
    config['parameters'] = config['parameters'].copy()
    config['parameters']['seed'] = seed
    model_exp_path = f"{experiment_name}_{seed}"

    model, _, _ = initialize_training(config)
    model_path = os.path.join(base_path, model_exp_path, "model_best.pth")

    model.load_state_dict(torch.load(model_path))
    print("Successfully load model at ", model_path)

    return model

def load_all_models(configurations_list):
    models_by_dataset = {"model_name": [], }

    for i, experiment_name in enumerate(models_by_dataset.keys()):
        for seed in range(5):  
            model = load_model_for_seed(experiment_name, seed, configurations_list[list(configurations_list)[i]])
            models_by_dataset[experiment_name].append(model)

    return models_by_dataset



