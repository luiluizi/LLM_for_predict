import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x, spatial_enc):
        query = self.query(x)
        key = self.key(spatial_enc)
        value = self.value(spatial_enc)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = F.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention_scores, value)
        return out

class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super(LaplacianPE, self).__init__()
        self.embedding_pos_enc = nn.Linear(lape_dim, embed_dim)
    
    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc

class SpatialSelfAttention(nn.Module):
    def __init__(self, dim, feature_dim, input_window, attn_drop=0.0, num_head=4):
        super().__init__()
        self.embed_dim = dim
        self.feature_dim = feature_dim
        self.input_window = input_window
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.SConv = nn.Conv2d(dim // 2, dim, kernel_size=1, bias=True)
        self.EConv = nn.Conv2d(input_window, feature_dim, kernel_size=1, bias=True)
        self.num_head =  num_head
        self.head_dim = self.embed_dim // self.num_head
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        x = x.transpose(1, 3)
        x = self.SConv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        B, T, N, D = x.shape
        q = self.q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        k = self.k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        v = self.v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # reshape
        q = q.reshape(B, T, N, self.num_head, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, T, N, self.num_head, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, T, N, self.num_head, self.head_dim).permute(0, 1, 3, 2, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).reshape(B, T, N, D)
        x = self.EConv(x)
        return x
        
class MyEncoder(nn.Module):
    def __init__(self, config, data_feature):
        super(MyEncoder, self).__init__()
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 2)
        self.output_dim = data_feature.get('output_dim', 2)
        self.region_id_list = data_feature.get('region_id_list')

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)

        self.dropout = config.get('enc_drop', 0.1)
        self.dilation_exponential = config.get('enc_dilation_exponential', 1)

        self.conv_channels = config.get('enc_conv_channels', 32)
        self.residual_channels = config.get('enc_residual_channels', 32)
        self.skip_channels = config.get('enc_skip_channels', 64)
        self.end_channels = config.get('enc_end_channels', 128)
        self.layers = config.get('enc_layers', 3)
        self.attn_drop = config.get('attn_drop', 0.0)
        
        self.lape_dim = config.get('lape_dim', 8)
        self.st_hidden_size = config.get('st_hidden_size', 64)

        self.cross_attention = CrossAttention(dim=self.st_hidden_size)
        self.spatial_attn = SpatialSelfAttention(dim=self.st_hidden_size, 
                                                 attn_drop=self.attn_drop, 
                                                 feature_dim=self.feature_dim, 
                                                 input_window=self.input_window)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        self.spatial_embedding = LaplacianPE(self.lape_dim, self.residual_channels)
        self.project = nn.Linear(2 * self.st_hidden_size, self.st_hidden_size)
        self.project_drop = nn.Dropout(self.attn_drop)

        kernel_size = 7
        if self.dilation_exponential > 1:
            self.receptive_field = int(self.output_dim + (kernel_size-1) * (self.dilation_exponential**self.layers-1)
                                       / (self.dilation_exponential - 1))
        else:
            self.receptive_field = self.layers * (kernel_size-1) + self.output_dim

        for i in range(1):
            if self.dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size-1) * (self.dilation_exponential**self.layers-1)
                                / (self.dilation_exponential - 1))
            else:
                rf_size_i = i * self.layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, self.layers+1):
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1) * (self.dilation_exponential**j - 1)
                                    / (self.dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(DilatedInception(self.residual_channels,
                                                          self.conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(DilatedInception(self.residual_channels,
                                                        self.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.residual_channels, kernel_size=(1, 1)))
                if self.input_window > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
                                                     kernel_size=(1, self.input_window-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
                                                     kernel_size=(1, self.receptive_field-rf_size_j+1)))

                new_dilation *= self.dilation_exponential

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window, kernel_size=(1, 1), bias=True)
        if self.input_window > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
                                   out_channels=self.skip_channels,
                                   kernel_size=(1, self.input_window), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels,
                                   out_channels=self.skip_channels,
                                   kernel_size=(1, self.input_window-self.receptive_field+1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
                                   out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels,
                                   out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)

    def check_scale(self, embedding):
        print(f"Max: {embedding.max().item()}")
        print(f"Min: {embedding.min().item()}")
        print(f"Mean: {embedding.mean().item()}")
        print(f"Std: {embedding.std().item()}")

    def normalize(self, embedding):
        min_val = embedding.min()
        max_val = embedding.max()
        embedding = (embedding - min_val) / (max_val - min_val)
        return embedding

    def forward(self, source, lap_mx):
        inputs = source
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)
        if self.input_window < self.receptive_field:
            inputs = nn.functional.pad(inputs, (self.receptive_field-self.input_window, 0, 0, 0))
        x = self.start_conv(inputs)
        skip = self.skip0(F.dropout(inputs, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filters = self.filter_convs[i](x)
            filters = torch.tanh(filters)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filters * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        return x
    
    def forward_with_spatial(self, source, lap_mx):
        inputs = source
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)
        x = self.start_conv(inputs)
        spatial_enc = self.spatial_embedding(lap_mx).transpose(1, 3)
        spatial_enc = spatial_enc[:, :, self.region_id_list, :]
        x += spatial_enc
        raw_x = x
        if self.input_window < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field-self.input_window, 0, 0, 0))
            inputs = nn.functional.pad(inputs, (self.receptive_field-self.input_window, 0, 0, 0))
        # get time embedding
        skip = self.skip0(F.dropout(inputs, self.dropout, training=self.training))
        
        for i in range(self.layers):
            residual = x
            filters = self.filter_convs[i](x)
            filters = torch.tanh(filters)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filters * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

        skip = self.skipE(x) + skip
        time_embedding = F.relu(skip).transpose(1, 3)
        # get spatial embedding
        spatial_embedding = self.spatial_attn(raw_x)
        # fusion
        x = self.project(torch.cat([time_embedding, spatial_embedding], dim=-1)).transpose(1, 3)
        x = self.project_drop(x)  
        return x