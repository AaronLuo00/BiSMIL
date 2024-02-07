# BiSMIL

# Towards Interpretable, Sequential Multiple Instance Learning: An Application to Clinical Imaging

## Introduction
This work introduces the Sequential Multiple Instance Learning (SMIL) framework, addressing the challenge of interpreting sequential, variable-length sequences of medical images with a single diagnostic label. Diverging from traditional MIL approaches that treat image sequences as unordered sets, SMIL systematically integrates the sequential nature of clinical imaging. We develop a bidirectional Transformer architecture, BiSMIL, that optimizes for both early and final prediction accuracies through a novel training procedure to balance diagnostic accuracy with operational efficiency. We evaluate BiSMIL on three medical image datasets to demonstrate that it simeultaneously achieves state-of-the-art final accuracy and superior performance in early prediction accuracy, requiring 30-50\% fewer images for a similar level of performance compared to existing models. Additionally, we introduce SMILU, an interpretable uncertainty metric that outperforms traditional metrics in identifying challenging instances.

## Dataset

### Download
- The **RSNA** dataset used in this paper can be download via [Kaggle Challenge Dataset](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data)
- The **Covid** dataset used in this paper can be download via https://www.synapse.org/#%21Synapse:syn22174850

## 1. Install Requirements: 
```
conda create -n BiSMIL python=3.10
conda activate BiSMIL
pip install -r requirements.txt
```


## Usage




## Contributors
Some of the code in this repository is based on the following amazing works.

* https://github.com/yunanwu2168/sa-mil
* https://github.com/AMLab-Amsterdam/AttentionDeepMIL


# Citation
If you find this work helpful, please cite our paper.
```bibtex
TBA
