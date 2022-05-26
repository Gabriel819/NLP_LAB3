import torch.nn as nn
import torch


# model structure
class MyModel_1(nn.Module):
    def __init__(self, D_E, D_H, D_T):
        super(MyModel_1, self).__init__()
        self.D_H = D_H

        # Embedding layer
        self.embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_E, D_H)))  # input shape: D_E, output shape: D_H

        ###### RNN 3 layers #####

        # RNN layer 1
        self.w_x_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.w_h_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.b_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, D_H)))

        # RNN layer 2
        self.w_x_1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.w_h_1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.b_1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, D_H)))

        # RNN layer 3
        self.w_x_2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.w_h_2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(D_H, D_H)))
        self.b_2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, D_H)))

        self.layer_norm = nn.LayerNorm(D_H)
        self.softmax = torch.nn.Softmax(dim=1)
        self.tanh = torch.nn.Tanh()

        # Final Classifier layer
        self.classifier = nn.Linear(D_H, D_T)  # Unidirectional RNN

    def forward(self, inputs, device):
        b_s, max_len = inputs.shape[0], inputs.shape[1]
        
        h_prev = torch.zeros(3, b_s, self.D_H).to(device)

        for idx in range(max_len):
            # embedding layer
            v = torch.matmul(inputs[:, idx, :], self.embedding)  # embedding: (300, 512), input[:, idx, :]: (256, 300)

            # first RNN
            # v: (256, 512), w_x_0: (512, 512), w_h_0: (512, 512), b_0: (512,)
            h_t_0 = self.tanh(torch.matmul(v, self.w_x_0) + torch.matmul(h_prev[0], self.w_h_0) + self.b_0.repeat(b_s, 1))
            out_0 = self.layer_norm(h_t_0)

            # second RNN
            h_t_1 = self.tanh(torch.matmul(out_0, self.w_x_1) + torch.matmul(h_prev[1], self.w_h_1) + self.b_1.repeat(b_s, 1))
            out_1 = self.layer_norm(h_t_1)

            # third RNN
            h_t_2 = self.tanh(torch.matmul(out_1, self.w_x_2) + torch.matmul(h_prev[2], self.w_h_2) + self.b_2.repeat(b_s, 1))
            out_2 = self.layer_norm(h_t_2)

            h_prev = torch.cat([h_t_0.unsqueeze(dim=0), h_t_1.unsqueeze(dim=0), h_t_2.unsqueeze(dim=0)], dim=0).to(device)

        # classifier
        out = self.softmax(self.classifier(out_2))

        return out  # many-to-one은 맨 마지막 input만 return
