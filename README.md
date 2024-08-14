# Towards Interpretable, Sequential Multiple Instance Learning: An Application to Clinical Imaging

## Introduction
This work introduces the Sequential Multiple Instance Learning (SMIL) framework, addressing the challenge of interpreting sequential, variable-length sequences of medical images with a single diagnostic label. Diverging from traditional MIL approaches that treat image sequences as unordered sets, SMIL systematically integrates the sequential nature of clinical imaging. We develop a bidirectional Transformer architecture, BiSMIL, that optimizes for both early and final prediction accuracies through a novel training procedure to balance diagnostic accuracy with operational efficiency. We evaluate BiSMIL on three medical image datasets to demonstrate that it simeultaneously achieves state-of-the-art final accuracy and superior performance in early prediction accuracy, requiring 30-50\% fewer images for a similar level of performance compared to existing models. Additionally, we introduce SMILU, an interpretable uncertainty metric that outperforms traditional metrics in identifying challenging instances.

## Dataset

### Download
- The **RSNA** dataset used in this paper can be download via [Kaggle Challenge Dataset](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data)
  -  Preprocessed Dataset: https://drive.google.com/drive/folders/1NKmR38EggLHxL8bmbgVlgGE0u9fqamrf?usp=sharing
- The **Covid** dataset used in this paper can be download via https://www.synapse.org/#%21Synapse:syn22174850
  -  Preprocessed Dataset: https://drive.google.com/drive/folders/1NKmR38EggLHxL8bmbgVlgGE0u9fqamrf?usp=sharing

## Install Requirements: 
```
conda create -n BiSMIL python=3.10
conda activate BiSMIL
conda install scikit-learn numpy scipy openpyxl seaborn matplotlib pillow tqdm networkx
```


## Usage

`Covid_loader.py`: Generates training, validation and test loader from original Covid dataset. A bag is given a positive label if the patient belong to the "Covid" or "Other" class.
If run as main, it computes the ratio of positive bags as well as the mean, max and min value for the number per instances in a bag.

`RSNA_loader.py`: Generates training, validation and test loader from original RSNA dataset. There are five types of brain hemorrhage denoted in the dataset, and we create the bag-level binary label where a positive label indicates if any of the five types of hemorrhage is present.

`train_and_test.py`: Train a model with the Adam optimization algorithm, and then perform validation and test.
The training takes 40 epochs. Last, the accuracy and loss of the model on the test set is computed, and the trained model is saved.

`model.py`: Implementation of SiSMIL, BiSMIL, Attention Pooling, Max Pooling, and SA_DMIL model. The Incremental Prediction is  implemented inside of the SiSMIL model.

`uncertainty.py`: Implementation of SMILU.


## Contributors
Some of the code in this repository is based on the following amazing works.

* https://github.com/yunanwu2168/sa-mil
* https://github.com/AMLab-Amsterdam/AttentionDeepMIL


# Citation
If you find this work helpful, please cite our paper.
```bibtex
TBA
