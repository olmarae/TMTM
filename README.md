# TMTM

### Introduction

The detection of Twitter bots has become essential in combating misinformation and protecting user interests. The continuous evolution of bots and their strategies to evade detection has led to the development of models that combine feature-based, text-based, and graph-based techniques. We propose TMTM ("the more the merrier"), a multimodal model that leverages the heterogeneous Twitter network to create a graph from various user relationships, incorporating account-related features and semantic information from profile descriptions and tweets.


### Dataset download

TwiBot-22 is available at [Google Drive](https://drive.google.com/drive/folders/1YwiOUwtl8pCd2GD97Q_WEzwEUtSPoxFs?usp=sharing). You should apply for access. See more at [TwiBot-22 github repository](https://github.com/LuoUndergradXJTU/TwiBot-22).

Please download the dataset to a folder called "Dataset". 

### Requirements

- pip: `pip install -r requirements.txt`
- conda : `conda install --yes --file requirements.txt `

### Preprocess

preprocess the dataset by running

   `python preprocess.py`

### Train Model

train TMTM model by running:

   `python train_test.py`

