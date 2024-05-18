import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from Dataset import Dataset_TMTM
from model import TMTM
from utils import sample_mask, init_weights
import time
import itertools
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Connection configuration
relation_select = [0, 1] 
# Random seed configuration
random_seed = [0, 1, 2, 3, 4] 
# Default weight_decay
weight_decay = 5e-4


def main(seed):
    dataset = Dataset_TMTM('./Dataset')
    data = dataset[0]

    out_dim = 2
    data.y = data.y2


    sample_number = len(data.y)
    shuffled_idx = shuffle(np.array(range(sample_number)), random_state=seed)
    train_idx = shuffled_idx[:int(0.7 * sample_number)]
    val_idx = shuffled_idx[int(0.7 * sample_number):int(0.9 * sample_number)]
    test_idx = shuffled_idx[int(0.9 * sample_number):]
    data.train_mask = sample_mask(train_idx, sample_number)
    data.val_mask = sample_mask(val_idx, sample_number)
    data.test_mask = sample_mask(test_idx, sample_number)

    test_mask = data.test_mask
    train_mask = data.train_mask
    val_mask = data.val_mask

    data = data.to(device)
    embedding_size = data.x.shape[1]
    relation_num = len(relation_select)
    index_select_list = (data.edge_type == 100)

    relation_dict = {
        0:'following',
        1:'followers',
        2:'ownership'
    }

    for features_index in relation_select:
        if features_index in torch.unique(data.edge_type):
            index_select_list |= (data.edge_type == features_index)
            print('Relation used:', relation_dict[features_index])
    
    # Use only the selected relations
    edge_index = data.edge_index[:, index_select_list]
    edge_type = data.edge_type[index_select_list]

    # Model configuration
    try:
        model = TMTM(embedding_size, hidden_dimension, out_dim, relation_num, dropout)
    except Exception as e:
        print("Error during the model initialization:", e)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    
    def train(epoch):
        # Train model
        model.train()
        try:
            output = model(data.x, edge_index, edge_type)
        except Exception as e:
            print("Error in train during the model initialization:", e)
        loss_train = loss(output[data.train_mask], data.y[data.train_mask])
        out = output.max(1)[1].to('cpu').detach().numpy()
        label = data.y.to('cpu').detach().numpy()
        acc_train = accuracy_score(out[train_mask], label[train_mask])
        acc_val = accuracy_score(out[val_mask], label[val_mask])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'acc_val: {:.4f}'.format(acc_val.item()), )
        return acc_val


    def test():
        # Test model
        model.eval()
        output = model(data.x, edge_index, edge_type)
        loss_test = loss(output[data.test_mask], data.y[data.test_mask])
        out = output.max(1)[1].to('cpu').detach().numpy()
        label = data.y.to('cpu').detach().numpy()
        acc_test = accuracy_score(out[test_mask], label[test_mask])
        f1 = f1_score(out[test_mask], label[test_mask], average='macro')
        precision = precision_score(out[test_mask], label[test_mask], average='macro')
        recall = recall_score(out[test_mask], label[test_mask], average='macro')
        return acc_test, loss_test, f1, precision, recall

    model.apply(init_weights)


    max_val_acc = 0
    for epoch in range(epochs):
        acc_val = train(epoch)
        acc_test, loss_test, f1, precision, recall = test()
        if acc_val > max_val_acc:
            max_val_acc = acc_val
            max_acc = acc_test
            max_epoch = epoch + 1
            max_f1 = f1
            max_precision = precision
            max_recall = recall


    print("Test set results:",
          "epoch= {:}".format(max_epoch),
          "test_accuracy= {:.4f}".format(max_acc),
          "precision= {:.4f}".format(max_precision),
          "recall= {:.4f}".format(max_recall),
          "f1_score= {:.4f}".format(max_f1)
          )

    return max_acc, max_precision, max_recall, max_f1

if __name__ == "__main__":


    t = time.time()

    # Hyperparameters configuration
    hidden_dimension_options = [64, 128, 256]
    dropout_options = [0.2, 0.3, 0.4]
    epochs_options = [100, 150, 200]
    lr_options = [1e-2, 1e-3, 1e-4]

    # Best global configuration
    best_global_f1 = 0
    best_global_config = None

    # File to save the results
    with open('results.txt', 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f

        # Iter over all the possible configurations
        for hidden_dimension, dropout, epochs, lr in itertools.product(hidden_dimension_options, dropout_options, epochs_options, lr_options):
            acc_list = []
            precision_list = []
            recall_list = []
            f1_list = []

            # Actual configuration
            current_hyperparams = {
                'hidden_dimension': hidden_dimension, 
                'dropout': dropout, 
                'epochs': epochs, 
                'lr': lr, 
                'weight_decay': weight_decay
            }

            for i, seed in enumerate(random_seed):
                print('\nTraining {}th model'.format(i + 1))
                acc, precision, recall, f1, f1_w = main(seed)
                acc_list.append(acc * 100)
                precision_list.append(precision * 100)
                recall_list.append(recall * 100)
                f1_list.append(f1 * 100)

            print(f'Configuration: {current_hyperparams}')
            print('Accuracy: {:.2f} ± {:.2f}'.format(np.mean(acc_list), np.std(acc_list)))
            print('Precision: {:.2f} ± {:.2f}'.format(np.mean(precision_list), np.std(precision_list)))
            print('Recall: {:.2f} ± {:.2f}'.format(np.mean(recall_list), np.std(recall_list)))
            print('F1 Score: {:.2f} ± {:.2f}'.format(np.mean(f1_list), np.std(f1_list)))

            max_f1 = np.mean(f1_list)
            if max_f1 > best_global_f1:
                best_global_f1 = max_f1
                best_global_config = current_hyperparams
                print(f"New best configuration: {best_global_config} with a f1 value of {best_global_f1}")

        print(f"New best global configuration: {best_global_config} with a f1 value of {best_global_f1:.2f}")
        print('Total time:', time.time() - t)

        # Reset the standard output
        sys.stdout = original_stdout
