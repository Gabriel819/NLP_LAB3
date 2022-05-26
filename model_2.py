import torch.nn as nn
import torch
import numpy as np

# model structure
class MyModel_2(nn.Module):
    def __init__(self, D_E, D_H, D_T):
        super(MyModel_2, self).__init__()
        self.D_H = D_H
        self.D_T = D_T

        # Embedding layer
        self.l2r_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_E, D_H)))  # input shape: D_E, output shape: D_H
        self.r2l_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_E, D_H)))

        ############################## l2r RNN 3 layers ############################
        # RNN layer 1
        self.l2r_w_x_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.l2r_w_h_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.l2r_b_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, D_H)))

        # RNN layer 2
        self.l2r_w_x_1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.l2r_w_h_1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.l2r_b_1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, D_H)))

        # RNN layer 3
        self.l2r_w_x_2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.l2r_w_h_2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.l2r_b_2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, D_H)))
        #############################################################################

        ############################## r2l RNN 3 layers ############################
        # RNN layer 1
        self.r2l_w_x_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.r2l_w_h_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.r2l_b_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, D_H)))

        # RNN layer 2
        self.r2l_w_x_1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.r2l_w_h_1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.r2l_b_1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, D_H)))

        # RNN layer 3
        self.r2l_w_x_2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.r2l_w_h_2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.r2l_b_2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, D_H)))
        #############################################################################

        self.layer_norm = nn.LayerNorm(D_H)
        self.softmax = torch.nn.Softmax(dim=1)
        self.tanh = torch.nn.Tanh()

        # Final Classifier layer
        self.classifier = nn.Linear(2 * D_H, D_T)

    def forward(self, l2r_input, r2l_input, orig_len, device):
        b_s, max_len = l2r_input.shape[0], l2r_input.shape[1]

        l2r_h_prev = torch.zeros((3, b_s, self.D_H), requires_grad=False).to(device)
        r2l_h_prev = torch.zeros((3, b_s, self.D_H), requires_grad=False).to(device)

        l2r_out = torch.zeros((b_s, max_len, self.D_H), requires_grad=False)
        r2l_out = torch.zeros((b_s, max_len, self.D_H), requires_grad=False)

        for idx in range(max_len):
            ####################################################### l2r ######################################################
            # l2r embedding layer
            l2r_v = torch.matmul(l2r_input[:, idx, :], self.l2r_embedding) # embedding: , input[:, idx, :]:

            # first RNN
            l2r_h_t_0 = self.tanh(torch.matmul(l2r_v, self.l2r_w_x_0) + torch.matmul(l2r_h_prev[0], self.l2r_w_h_0) + self.l2r_b_0.repeat(b_s, 1))
            l2r_out_0 = self.layer_norm(l2r_h_t_0)

            # second RNN
            l2r_h_t_1 = self.tanh(torch.matmul(l2r_out_0, self.l2r_w_x_1) + torch.matmul(l2r_h_prev[1], self.l2r_w_h_1) + self.l2r_b_1.repeat(b_s, 1))
            l2r_out_1 = self.layer_norm(l2r_h_t_1)

            # third RNN
            l2r_h_t_2 = self.tanh(torch.matmul(l2r_out_1, self.l2r_w_x_2) + torch.matmul(l2r_h_prev[2], self.l2r_w_h_2) + self.l2r_b_2.repeat(b_s, 1))
            l2r_out_2 = self.layer_norm(l2r_h_t_2)

            l2r_out[:, idx] = l2r_out_2

            l2r_h_prev = torch.cat([l2r_h_t_0.unsqueeze(dim=0), l2r_h_t_1.unsqueeze(dim=0), l2r_h_t_2.unsqueeze(dim=0)], dim=0).to(device)
            ###################################################################################################################
            ####################################################### r2l ######################################################
            # r2l embedding layer
            r2l_v = torch.matmul(r2l_input[:, idx, :], self.r2l_embedding)

            # first RNN
            r2l_h_t_0 = self.tanh(torch.matmul(r2l_v, self.r2l_w_x_0) + torch.matmul(r2l_h_prev[0], self.r2l_w_h_0) + self.r2l_b_0.repeat(b_s, 1))
            r2l_out_0 = self.layer_norm(r2l_h_t_0)

            # second RNN
            r2l_h_t_1 = self.tanh(torch.matmul(r2l_out_0, self.r2l_w_x_1) + torch.matmul(r2l_h_prev[1],self.r2l_w_h_1) + self.r2l_b_1.repeat(b_s, 1))
            r2l_out_1 = self.layer_norm(r2l_h_t_1)

            # third RNN
            r2l_h_t_2 = self.tanh(torch.matmul(r2l_out_1, self.r2l_w_x_2) + torch.matmul(r2l_h_prev[2], self.r2l_w_h_2) + self.r2l_b_2.repeat(b_s, 1))
            r2l_out_2 = self.layer_norm(r2l_h_t_2)

            for j in range(b_s):
                k = max_len - min(orig_len[j].item(), 20) - idx
                r2l_out[j][k] = r2l_out_2[j]

            r2l_h_prev = torch.cat([r2l_h_t_0.unsqueeze(dim=0), r2l_h_t_1.unsqueeze(dim=0), r2l_h_t_2.unsqueeze(dim=0)], dim=0).to(device)
            ###################################################################################################################

        # classifier
        # l2r_out: (256, 20, 512), r2l_out: (256, 20, 512)
        out = torch.cat([l2r_out, r2l_out], dim=2).to(device) # (256, 20, 1024)
        res = self.softmax(self.classifier(out)).clone()

        return res
