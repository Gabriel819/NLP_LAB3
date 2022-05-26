import torch
import pandas as pd
from nltk.tokenize import word_tokenize
import pickle
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from model_1 import MyModel_1
from dataset_1 import MyDataset_1
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


def load_file(filepath, is_csv=True, is_train=True):
    if is_csv and is_train:  # load train csv file(has 'sentence' and 'label' columns)
        df = pd.read_csv(filepath)

        data = list(df['sentence'])
        targets = list(df['label'])

        return data, targets

    elif is_csv:  # load test csv file(only has 'sentence' column)
        df = pd.read_csv(filepath)

        data = list(df['sentence'])

        return data

    else:  # load Glove word embedding dictionary json file
        df = pd.read_json(filepath)

        return df

##### load data ####
tr_sents, tr_labels = load_file(filepath='./data/train_set.csv', is_csv=True, is_train=True) # train: 5000
ts_sents = load_file(filepath='./data/test_set.csv', is_csv=True, is_train=False) # test: 452
glove_word = load_file(filepath='./data/glove_word.json', is_csv=False, is_train=False) # glove_word: 9225 x 300

glove_word_list = list(glove_word.columns)

# check the unique category value labels
category_list = list(set(tr_labels))
num_category = len(category_list)
# print(category_list)
# print(num_category)

######### Step 1. Tokenize the input sentence & Step 2. Pre-padding & Pre-sequence truncation [1 pt] #########
max_len = 20 # max length of each sentence

def tokenization(sents):
    tokens = []

    for sen in sents:
        tmp_token = word_tokenize(sen)

        for idx in range(len(tmp_token)): # if this token is not in glove_word_list, change the token to '[UNK]' token
            if tmp_token[idx] not in glove_word_list:
                tmp_token[idx] = '[UNK]'

        if len(tmp_token) < max_len: # pre-padding
            # If the sentence's length is smaller than 20, pad '[PAD]' at the first part of the sentence
            tmp_token = ['[PAD]'] * (max_len - len(tmp_token)) + tmp_token
            tokens.append(tmp_token)
        elif len(tmp_token) > max_len: # pre-sequence truncation
            # If the sentence's length exceed 20, cut the rest
            tokens.append(tmp_token[0:max_len])
        else: # len(sen) == max_len
            # If the sentence's length is just 20, just use the all words
            tokens.append(tmp_token)

    return tokens

# if tr_tokens & ts_tokens already exist
# if tr_tokens & ts_tokens doesn't exist, comment out this cell
# with open('./tr_tokens.pickle','rb') as f:
#     tr_tokens = pickle.load(f)
# with open('./ts_tokens.pickle','rb') as f:
#     ts_tokens = pickle.load(f)

# # tokenization
tr_tokens = tokenization(tr_sents)
ts_tokens = tokenization(ts_sents)

# # store tr_tokens and ts_tokens
# with open('./tr_tokens.pickle','wb') as fw:
#     pickle.dump(tr_tokens, fw)
# with open('./ts_tokens.pickle','wb') as fw:
#     pickle.dump(ts_tokens, fw)

######### Step 3. Convert token into vector with given Glove word embedding dictionary #########
# if tr_vec & ts_vec already exist
# if tr_vec & ts_vec doesn't exist, comment out this cell
# with open('./tr_vec.pickle','rb') as f:
#     tr_vec = pickle.load(f)
# with open('./ts_vec.pickle','rb') as f:
#     ts_vec = pickle.load(f)

tr_vec = torch.zeros((len(tr_sents), max_len, len(glove_word))) # (5000, 20, 300)
ts_vec = torch.zeros((len(ts_sents), max_len, len(glove_word))) # (452, 20, 300)

# make training vector with Glove word embedding dictionary
for token_idx, token in enumerate(tr_tokens):
    for idx in range(max_len):
        print(token[idx])
        tr_vec[token_idx][idx] = torch.tensor(glove_word[token[idx]])

# make test vector with Glove word embedding dictionary
for token_idx, token in enumerate(ts_tokens):
    for idx in range(max_len):
        print(token[idx])
        ts_vec[token_idx][idx] = torch.tensor(glove_word[token[idx]])

# # store tr_vec and ts_vec
# with open('./tr_vec.pickle','wb') as fw:
#     pickle.dump(tr_vec, fw)
# with open('./ts_vec.pickle','wb') as fw:
#     pickle.dump(ts_vec, fw)

# ######### Step 4. Training Classification Model [3pt] #########
D_H, D_T, D_E = 512, num_category, 300
batch_size = 256

model = MyModel_1(D_E, D_H, D_T).to(device)

exp_num = 13 # set the number of this experiment
lr = 1e-5
num_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

# Define DataLoader
my_dataset = MyDataset_1(tr_vec, tr_labels)

train_size = int(0.9 * len(my_dataset)) # train dataset's size is 0.9 * total_labeled_dataset
valid_size = len(my_dataset) - train_size # valid dataset's size is 0.1 * total_labeled_dataset

# randomly choose data from total dataset to put in train_dataset or valid_dataset
train_dataset, valid_dataset = torch.utils.data.random_split(my_dataset, [train_size, valid_size])

# shuffle train dataset but not shuffle valid dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle = False)

##### Train #####
def get_ap_score(y_true, y_scores):
    """
    Get average precision score between 2 1-d numpy arrays
    Args:
        y_true: batch of true labels
        y_scores: batch of confidence scores

    Returns:
        sum of batch average precision
    """
    scores = 0.0

    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true=y_true[i], y_score=y_scores[i])

    return scores


train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
best_val_acc = 0.0

for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0

    model = model.to(device)

    ############### Training Phase #############
    for idx, (tr_data, tr_label) in enumerate(train_loader):
        # tr_data: (256, 20, 300), tr_label: (256,)
        tr_data = tr_data.to(device)

        tr_label = F.one_hot(tr_label, num_classes=num_category).type(torch.float32).to(device)

        model.train()
        optimizer.zero_grad()

        tr_output = model(tr_data, device)

        tr_loss = criterion(tr_output, tr_label)
        tr_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping with max_norm 1.0
        optimizer.step()

        train_loss += tr_loss.item()
        train_acc += get_ap_score(torch.Tensor.cpu(tr_label).detach().numpy(),
                                  torch.Tensor.cpu(tr_output).detach().numpy())

    train_num_samples = float(len(train_loader.dataset))
    tr_loss_ = train_loss / train_num_samples
    tr_acc_ = train_acc / train_num_samples

    train_loss_list.append(tr_loss_)
    train_acc_list.append(tr_acc_)

    ############### Evaluation Phase #############
    for idx, (val_data, val_label) in enumerate(valid_loader):
        val_data = val_data.to(device)
        val_label = F.one_hot(val_label, num_classes=num_category).type(torch.float32).to(device)

        model.eval()

        vl_output = model(val_data, device)

        vl_loss = criterion(vl_output, val_label)

        val_loss += vl_loss.item()
        val_acc += get_ap_score(torch.Tensor.cpu(val_label).detach().numpy(),
                                torch.Tensor.cpu(vl_output).detach().numpy())

    valid_num_samples = float(len(valid_loader.dataset))
    val_loss_ = val_loss / valid_num_samples
    val_acc_ = val_acc / valid_num_samples

    val_loss_list.append(val_loss_)
    val_acc_list.append(val_acc_)

    print(
        '\nEpoch {}, train_loss: {:.6f}, train_acc:{:.3f}, valid_loss: {:.6f}, valid_acc:{:.3f}'.format(epoch, tr_loss_,
                                                                                                        tr_acc_,
                                                                                                        val_loss_,
                                                                                                        val_acc_))

    # if this epoch's model's validation accuracy is better than before, store the model parameter
    if val_acc_ > best_val_acc:
        best_val_acc = val_acc_
        torch.save(model.state_dict(), f'./LAB3-1_parameters/model_{exp_num}.pth')
        print(f'Epoch {epoch} model saved')

##### make figure
epoch_list = list(range(num_epochs))

plt.plot(epoch_list, train_loss_list, 'r')
plt.plot(epoch_list, val_loss_list, 'b')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.show()
# plt.savefig(f'./LAB3-1_submissions/exp{exp_num}_loss_graph.png') # store the loss figure in png file

plt.plot(epoch_list, train_acc_list, 'r')
plt.plot(epoch_list, val_acc_list, 'b')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
# plt.savefig(f'./LAB3-1_submissions/exp{exp_num}_acc_graph.png') # store the accuracy figure in png file

######### Step 5. Evaluate your trained model performance on the test set [x1.0 or x0.5] #########
test_y = model(ts_vec.to(device), device)

final_pred = test_y.argmax(dim=1)

submission = ['ID,label\n']
f2 = './data/classification_class.pred.csv'

with open(f2, 'rb') as f:
    file = f.read().decode('utf-8')
    content = file.split('\n')[:-1] # column name

    for idx, line in enumerate(content):
        if idx == 0: # first line is ID, label so just skip it
            continue
        tmp1 = line.split(',') # split the id and prediction result by ,
        res = final_pred[idx-1].item() # get the final prediction result of this id
        tmp2 = tmp1[0] + ',' + str(res) + '\n'
        submission.append(tmp2)

print(submission)

with open(f'./LAB3-1_submissions/20214047_lab3-1_submission{exp_num}.csv', 'w') as f:
    f.write(''.join(submission)) # store the submission file
