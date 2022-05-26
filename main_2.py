import torch
import pandas as pd
import pickle
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from model_2 import MyModel_2
from dataset_2 import MyDataset_2
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import time
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

############## Step 1. Post-padding & Pre-sequence truncation [1pt] #####################
tr_df = pd.read_json('./data/train_set.json')
ts_df = pd.read_json('./data/test_set.json')

# training & test tokens list
tr_tokens = tr_df.loc['tokens', :]
tr_tokens_list = list(tr_tokens)
tr_len = len(tr_tokens_list)

ts_tokens = ts_df.loc['tokens', :]
ts_tokens_list = list(ts_tokens)
ts_len = len(ts_tokens_list) # 500

# calculate length of each sentence of training & test set
orig_len_tensor = torch.zeros(len(tr_tokens), dtype=torch.int64)
ts_orig_len_tensor = torch.zeros(len(ts_tokens), dtype=torch.int64)
for i in range(tr_len):
    orig_len_tensor[i] = len(tr_tokens[i])
for i in range(ts_len):
    ts_orig_len_tensor[i] = len(ts_tokens[i])
ts_max_len = torch.max(ts_orig_len_tensor).item() # 65

with open('./orig_len_tensor.pickle', 'wb') as fw:
    pickle.dump(orig_len_tensor, fw)
with open('./ts_orig_len_tensor.pickle', 'wb') as fw:
    pickle.dump(ts_orig_len_tensor, fw)
print("Original length of tensor store complete!")


# ud_tags_list
tr_ud_tags = tr_df.loc['ud_tags', :]
tr_ud_tags_list = list(tr_ud_tags)

# fasttext word
fasttext_word = pd.read_json('./data/fasttext_word.json')
word_dic_columns = list(fasttext_word.columns)
word_dic = {word:i for i, word in enumerate(word_dic_columns)}

fasttext_word_list = list(fasttext_word.columns)

max_len = 20

tr_int = torch.zeros((tr_len, max_len))
ts_int = torch.zeros((ts_len, ts_max_len))

# training set
for idx in range(tr_len):
    cur_len = len(tr_tokens_list[idx])

    for j in range(cur_len):
        if tr_tokens_list[idx][j] not in fasttext_word_list:
            tr_tokens_list[idx][j] = '[UNK]'

    if cur_len < max_len:
        tr_tokens_list[idx] += ['[PAD]'] * (max_len - cur_len)
        tr_ud_tags_list[idx] += ['[PAD]'] * (max_len - cur_len)

    elif cur_len > max_len:
        tr_tokens_list[idx] = tr_tokens_list[idx][0:max_len]
        tr_ud_tags_list[idx] = tr_ud_tags_list[idx][0:max_len]

    for k in range(20):
        tr_int[idx][k] = word_dic[tr_tokens_list[idx][k]]

# test set
for idx in range(ts_len):
    cur_len = len(ts_tokens_list[idx])

    for j in range(cur_len):
        if ts_tokens_list[idx][j] not in fasttext_word_list:
            ts_tokens_list[idx][j] = '[UNK]'

    if cur_len < ts_max_len:
        ts_tokens_list[idx] += ['[PAD]'] * (ts_max_len - cur_len)

    for k in range(ts_max_len):
        ts_int[idx][k] = word_dic[ts_tokens_list[idx][k]]

with open('./tr_tokens_list.pickle','wb') as fw:
    pickle.dump(tr_tokens_list, fw)
with open('./ts_tokens_list.pickle','wb') as fw:
    pickle.dump(ts_tokens_list, fw)
with open('./tr_ud_tags_list.pickle','wb') as fw:
    pickle.dump(tr_ud_tags_list, fw)
with open('./tr_int.pickle','wb') as fw:
    pickle.dump(tr_int, fw)
with open('./ts_int.pickle','wb') as fw:
    pickle.dump(ts_int, fw)
print("tr_tokens_list, tr_ud_tags_list, tr_int store complete.")


# with open('./tr_ud_tags_list.pickle','rb') as f:
#     tr_ud_tags_list = pickle.load(f)

# load label dictionary
f = open("./data/tgt.txt", 'r')
label_dic = {}
label_dic_rev = {}
lines = f.readlines()
for i, line in enumerate(lines):
    line = line.strip()
    label_dic[line] = i
    label_dic_rev[i] = line
f.close()

with open('./label_dic.pickle','wb') as fw:
    pickle.dump(label_dic, fw)
with open('./label_dic_rev.pickle','wb') as fw:
    pickle.dump(label_dic_rev, fw)

num_category = len(label_dic)

tr_label = torch.zeros((tr_len, max_len))

for idx in range(tr_len):
    for j in range(20):
        tr_label[idx][j] = label_dic[tr_ud_tags_list[idx][j]]

with open('./tr_label.pickle','wb') as fw:
    pickle.dump(tr_label, fw)
print("tr label store complete.")

############## Step 3. Flipped input tokens for bi-directional RNN [1pt] #####################
# with open('./tr_int.pickle','rb') as f:
#     tr_int = pickle.load(f)
# with open('./ts_int.pickle','rb') as f:
#     ts_int = pickle.load(f)

tr_flip_int = torch.zeros_like(tr_int)
ts_flip_int = torch.zeros_like(ts_int)

for idx in range(len(tr_int)):
    res = np.zeros_like(tr_int[idx])
    b = torch.flip(tr_int[idx], dims = [0])
    b_np = b.numpy()
    start = np.where(b_np!=0)[0][0]
    res[0:20-start] = b_np[start:]
    tr_flip_int[idx] = torch.tensor(res)

for idx in range(len(ts_int)):
    res = np.zeros_like(ts_int[idx])
    b = torch.flip(ts_int[idx], dims = [0])
    b_np = b.numpy()
    start = np.where(b_np!=0)[0][0]
    res[0:65-start] = b_np[start:]
    ts_flip_int[idx] = torch.tensor(res)

with open('./ts_flip_int.pickle','wb') as fw:
    pickle.dump(ts_flip_int, fw)
print("ts_flip_int store complete.")


############## Step 2. Use given FastText word embedding dictionary and vectors #####################
# with open('./fasttext_word_tensor.pickle','rb') as f:
#     fasttext_word_tensor = pickle.load(f)

# with open('./tr_int.pickle','rb') as f:
#     tr_int = pickle.load(f)
# with open('./tr_flip_int.pickle','rb') as f:
#     tr_flip_int = pickle.load(f)

fasttext_word = pd.read_json('./data/fasttext_word.json')
fasttext_word_list = list(fasttext_word.columns)

fasttext_word_tensor = torch.zeros((19674, 300))

for i, word in enumerate(fasttext_word_list):
    fasttext_word_tensor[i] = torch.tensor(fasttext_word[word])
with open('./fasttext_word_tensor.pickle','wb') as fw:
    pickle.dump(fasttext_word_tensor, fw)
print("fasttext word tensor store complete.")

# word embedding
tr_word_embed = torch.zeros((len(tr_int), 20, 300)) # (12543, 20, 300)
tr_flip_embed = torch.zeros((len(tr_int), 20, 300)) # (12543, 20, 300)

for i in range(tr_len):
    for j in range(20):
        print(i, j)
        tr_word_embed[i][j] = fasttext_word_tensor[int(tr_int[i][j].item())]
        tr_flip_embed[i][j] = fasttext_word_tensor[int(tr_flip_int[i][j].item())]

with open('./tr_word_embed.pickle','wb') as fw:
    pickle.dump(tr_word_embed, fw)
with open('./tr_flip_embed.pickle','wb') as fw:
    pickle.dump(tr_flip_embed, fw)
print("tr_word_embed, tr_flip_embed store complete.")

# with open('./ts_int.pickle','rb') as f:
#     ts_int = pickle.load(f)
# with open('./ts_flip_int.pickle','rb') as f:
#     ts_flip_int = pickle.load(f)

# test data word embedding
ts_word_embed = torch.zeros((len(ts_int), 65, 300)) # (500, 65, 300)
ts_flip_embed = torch.zeros((len(ts_int), 65, 300)) # (500, 65, 300)

for i in range(len(ts_int)):
    for j in range(65):
        print(i, j)
        ts_word_embed[i][j] = fasttext_word_tensor[int(ts_int[i][j].item())]
        ts_flip_embed[i][j] = fasttext_word_tensor[int(ts_flip_int[i][j].item())]

with open('./ts_word_embed.pickle','wb') as fw:
    pickle.dump(ts_word_embed, fw)
with open('./ts_flip_embed.pickle','wb') as fw:
    pickle.dump(ts_flip_embed, fw)
print("ts_word_embed, ts_flip_embed store complete.")

############ Step 4. Masking Padding token to prevent gradient flow from 'PAD' token [1pt] ###############
# with open('./tr_int.pickle','rb') as f:
#     tr_int = pickle.load(f)
# with open('./ts_int.pickle','rb') as f:
#     ts_int = pickle.load(f)

tr_mask = tr_int > 0
ts_mask = ts_int > 0

with open('./tr_mask.pickle','wb') as fw:
    pickle.dump(tr_mask, fw)
with open('./ts_mask.pickle','wb') as fw:
    pickle.dump(ts_mask, fw)
print("tr_mask, ts_mask store complete.")
##################################### Step 5. Training Model [3pt] #######################################
num_category = 18
D_H, D_T, D_E = 512, num_category, 300
batch_size = 256

model = MyModel_2(D_E, D_H, D_T).to(device)

# with open('./tr_label.pickle','rb') as f:
#     tr_label = pickle.load(f)

## Exp Setting ##
exp_num = 6 # set the number of this experiment
lr = 1e-3
num_epochs = 150
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

# Define DataLoader
# with open('./orig_len_tensor.pickle','rb') as f:
#     orig_len_tensor = pickle.load(f)

# with open('./tr_word_embed.pickle','rb') as f:
#     tr_word_embed = pickle.load(f)
# with open('./tr_flip_embed.pickle', 'rb') as f:
#     tr_flip_embed = pickle.load(f)
# with open('./tr_mask.pickle', 'rb') as f:
#     tr_mask = pickle.load(f)

my_dataset = MyDataset_2(tr_word_embed, tr_flip_embed, tr_label, tr_mask, orig_len_tensor)

train_size = int(0.9 * len(my_dataset)) # train dataset's size is 0.9 * total_labeled_dataset
valid_size = len(my_dataset) - train_size # valid dataset's size is 0.1 * total_labeled_dataset

# randomly choose data from total dataset to put in train_dataset or valid_dataset
train_dataset, valid_dataset = torch.utils.data.random_split(my_dataset, [train_size, valid_size])

# shuffle train dataset but not shuffle valid dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle = False)

###### Train #####
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

print("Train Start!")
for epoch in range(num_epochs):
    start_time = time.time()

    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0

    model = model.to(device)

    ############### Training Phase #############
    for idx, (l2r_data, r2l_data, tr_label, tr_mask, orig_len_data) in enumerate(train_loader):
        print(idx)
        # l2r_data & r2l_data: (256, 20, 300), tr_label: (256, 20), mask: (256, 20)
        l2r_data = l2r_data.to(device)
        r2l_data = r2l_data.to(device)

        mask_np = tr_mask.flatten() # (5120, )
        mask_tmp_np = mask_np.clone().detach().cpu().numpy()
        mask_idx = np.where(mask_tmp_np == False)

        # tr_mask = tr_mask.unsqueeze(2).repeat(1, 1, num_category)
        # tr_mask = tr_mask.flatten(0, 1).to(device)
        tr_mask = tr_mask.flatten().unsqueeze(1).repeat(1, num_category).to(device)

        orig_len_data = orig_len_data.to(device)

        tr_label = F.one_hot(tr_label.flatten().type(torch.int64), num_classes=num_category).type(torch.float32).to(device)

        model.train()
        optimizer.zero_grad()

        tr_output = model(l2r_data, r2l_data, orig_len_data, device) # (256, 20, 18)
        tr_output = tr_output.flatten(0, 1) # (5120, 18)

        tr_output *= tr_mask

        tr_output[mask_idx, 0] = 1

        tr_loss = criterion(tr_output, tr_label)

        # torch.autograd.set_detect_anomaly(True)

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
    for idx, (val_l2r_data, val_r2l_data, val_label, val_mask, val_orig_len_data) in enumerate(valid_loader):
        val_l2r_data = val_l2r_data.to(device)
        val_r2l_data = val_r2l_data.to(device)
        
        val_mask_np = val_mask.flatten() # (5120, )
        val_mask_tmp_np = val_mask_np.clone().detach().cpu().numpy()
        val_mask_idx = np.where(val_mask_tmp_np == False)

        val_mask = val_mask.flatten().unsqueeze(1).repeat(1, num_category).to(device)

        val_label = F.one_hot(val_label.flatten().type(torch.int64), num_classes=num_category).type(torch.float32).to(device)

        val_orig_len_data = val_orig_len_data.to(device)

        model.eval()

        vl_output = model(val_l2r_data, val_r2l_data, val_orig_len_data, device)
        vl_output = vl_output.flatten(0, 1)

        vl_output *= val_mask
        vl_output[val_mask_idx, 0] = 1

        vl_loss = criterion(vl_output, val_label)

        val_loss += vl_loss.item()
        val_acc += get_ap_score(torch.Tensor.cpu(val_label).detach().numpy(),
                                torch.Tensor.cpu(vl_output).detach().numpy())

    valid_num_samples = float(len(valid_loader.dataset))
    val_loss_ = val_loss / valid_num_samples
    val_acc_ = val_acc / valid_num_samples

    val_loss_list.append(val_loss_)
    val_acc_list.append(val_acc_)

    end_time = time.time()

    writer.add_scalar("LAB3-2_Loss/train", tr_loss_, epoch)
    writer.add_scalar("LAB3-2_Acc/train", tr_acc_, epoch)
    writer.add_scalar("LAB3-2_Loss/Valid", val_loss_, epoch)
    writer.add_scalar("LAB3-2_Acc/Valid", val_acc_, epoch)

    print(
        '\nEpoch {}, train_loss: {:.6f}, train_acc:{:.3f}, valid_loss: {:.6f}, valid_acc:{:.3f}, time: {:.3f}'.format(epoch, tr_loss_,
                                                                                                        tr_acc_,
                                                                                                        val_loss_,
                                                                                                        val_acc_, end_time-start_time))

    # if this epoch's model's validation accuracy is better than before, store the model parameter
    if val_acc_ > best_val_acc:
        best_val_acc = val_acc_
        torch.save(model.state_dict(), f'./LAB3-2_parameters/model_{exp_num}.pth')
        print(f'Epoch {epoch} model saved')

writer.flush()
writer.close()

############ Step 6. Evaluate your trained model performance on the test set [x1.0 or x0.5] ###############
# with open('./ts_word_embed.pickle','rb') as f:
#     ts_word_embed = pickle.load(f)
# with open('./ts_flip_embed.pickle','rb') as f:
#     ts_flip_embed = pickle.load(f)
# with open('./ts_orig_len_tensor.pickle','rb') as f:
#     ts_orig_len_tensor = pickle.load(f)
# with open('./label_dic_rev.pickle','rb') as f:
#     label_dic_rev = pickle.load(f)
# with open('./ts_mask.pickle','rb') as f:
#     ts_mask = pickle.load(f)

print("Evaluate on Test set Start!")

test_y = model(ts_word_embed.to(device), ts_flip_embed.to(device), ts_orig_len_tensor.to(device), device) # (500, 65, 18)

final_pred = test_y.argmax(dim=2) # (500, 65)
final_pred = final_pred.flatten() # (10000, )

ts_mask_np = ts_mask.flatten() # (10000, )
ts_mask_tmp_np = ts_mask_np.clone().numpy()
ts_mask_idx = np.where(ts_mask_tmp_np == False)[0]
final_pred[ts_mask_idx] = -1
final_pred = final_pred.reshape(500, 65)

submission = ['ID,label\n']

line_idx = 0
for i in range(len(ts_orig_len_tensor)):
    for j in range(ts_orig_len_tensor[i]):
        tmp = "S"+str(line_idx)+','+str(final_pred[i][j].item())+'\n'
        submission.append(tmp)
        line_idx += 1

print(submission)

with open(f'./LAB3-2_submissions/20214047_lab3-2_submission{exp_num}.csv', 'w') as f:
    f.write(''.join(submission)) # store the submission file
