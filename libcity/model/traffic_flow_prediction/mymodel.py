import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model.traffic_flow_prediction.myencoder import MyEncoder
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import AutoConfig, StoppingCriteria

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    mae_loss = torch.abs(true - pred)
    return torch.mean(mae_loss)

def scaler_mae_loss(scaler=None, mask_value=None):
    def loss(preds, labels, mask=None):
        if scaler is not None:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if mask is not None:
            preds = preds * mask
            labels = labels * mask
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

class MyModel(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        # torch.autograd.set_detect_anomaly(True)
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.config = config
        self.data_feature = data_feature
        self.active_layers = config.get('active_layers', 0)
        self.llama_config = None
        self.st_start_token = data_feature.get('st_start_token', -1)
        self.st_end_token = data_feature.get('st_end_token', -1)
        self.st_start_id0, self.st_start_id1, self.st_start_id2, self.st_end_id1, self.st_end_id2 = -100000, -100000, -100000, -100000, -100000
        self.llama_model = None
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.lin_hidden_size = config.get('llm_lin_hidden_size', 128)
        self.time_steps = config.get('llm_time_steps', 12)
        self.st_hidden_size = config.get('st_hidden_size', 64)
        self.tokenizer = data_feature.get('tokenizer')
        self.initialize_llm_model()
        self.hidden_size = self.llama_model.config.hidden_size
        self.vocab_size = self.llama_model.config.vocab_size
        self.st_tower = MyEncoder(config, data_feature)
        self.initialize_encoder()
        self.st_projector = nn.Linear(self.st_hidden_size, self.hidden_size)
        self.st_pred_linear_1 = nn.Linear(self.hidden_size, self.lin_hidden_size)
        self.st_pred_linear_2 = nn.Linear(self.lin_hidden_size*2, self.time_steps)
        self.st_pred_linear_3 = nn.Linear(self.hidden_size, self.lin_hidden_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.st_pre_res = []
        self.convert_type()
        
    def convert_type(self):
        self.st_projector.to(torch.bfloat16)
        self.st_pred_linear_1.to(torch.bfloat16)
        self.st_pred_linear_2.to(torch.bfloat16)
        self.st_pred_linear_3.to(torch.bfloat16)
        self.llama_model.lm_head.to(torch.bfloat16)
    
    def get_data_feature(self):
        return self.data_feature
        
    def initialize_encoder(self):
        path = '/home/panda/private/jjw/hck/br/LLM_for_predict/checkpoints/myencoder_pretrain.pth'
        load_state_dict = torch.load(path)
        cur_state_dict = self.st_tower.state_dict()
        for name, weight in load_state_dict.items():
            if 'predictor' in name:
                cur_name = name.replace('predictor.', '')
                cur_state_dict[cur_name].copy_(weight)
        self.st_tower.requires_grad_(False)
        # self.st_tower.spatial_embedding.requires_grad_(True)
        # self.st_tower.start_conv.requires_grad_(True)
    
    def initialize_llm_model(self):
        path = '/home/panda/private/jjw/hck/br/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
        self.llama_model = LlamaForCausalLM.from_pretrained(path).to(torch.bfloat16)
        self.llama_model.requires_grad_(False)
        num_new_tokens = 2  # start and end
        # 使新加入的特殊token的嵌入获取不是随机初始化的
        self.llama_model.resize_token_embeddings(len(self.tokenizer))
        input_embeddings = self.llama_model.get_input_embeddings().weight.data
        output_embeddings = self.llama_model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        # tune_st_mlp_adapter freeze input and tune output
        for p in self.llama_model.get_input_embeddings().parameters():
            p.requires_grad = False
        for p in self.llama_model.get_output_embeddings().parameters():
            p.requires_grad = True
        # self.freeze_llama_model()
    
    def freeze_llama_model(self):
        # 解冻后active_layers层的自注意层
        total_layers = len(self.llama_model.model.layers)
        for i, layer in enumerate(self.llama_model.model.layers):
            for param in layer.mlp.parameters():
                param.requires_grad = False
            if i < total_layers - self.active_layers:
                for param in layer.self_attn.parameters():
                    param.requires_grad = False

    def forward(self, batch, lap_mx=None):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        st_data_x = batch['st_data_x']
        st_data_y = batch['st_data_y']
        region_id = batch['region_id']
        labels = batch['labels'] 
        inputs_embeds = self.llama_model.model.embed_tokens(input_ids)
        if len(st_data_x) > 1:
            st_data_x = torch.cat(st_data_x, dim=0)
        if type(st_data_x) is list:
            STE_out = self.st_tower(st_data_x[0][..., :2], lap_mx)
            if STE_out.shape[2] >= 1:
                region_select_out = STE_out[:, :, region_id[0]:region_id[0] + 1, :].to(torch.bfloat16)
        else:
            STE_out = self.st_tower(st_data_x[..., :2], lap_mx)
            region_select_out = STE_out[:, :, region_id[0]:region_id[0] + 1, :].to(torch.bfloat16)
        st_projector_out = self.st_projector(region_select_out.transpose(1, -1))
        new_input_embeds = []
        cur_st_idx = 0
        for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
            cur_st_feature = st_projector_out[cur_st_idx]
            cur_st_feature = cur_st_feature.reshape(cur_st_feature.shape[0], -1)
            num_patches = cur_st_feature.shape[0]
            st_start_tokens = torch.where(cur_input_ids == self.st_start_token)[0]
            # insert st embedding into the prompt
            st_start_token_pos1, st_start_token_pos2, st_start_token_pos3 = st_start_tokens[0], st_start_tokens[1], st_start_tokens[2]
            self.st_start_id0 = st_start_token_pos1
            self.st_start_id1 = st_start_token_pos3
            
            if cur_input_ids[st_start_token_pos1 + num_patches + 1] != self.st_end_token:
                raise ValueError('end token should follow the start token')
            
            cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos1].detach(),
                                            cur_input_embeds[st_start_token_pos1:st_start_token_pos1 + 1],
                                            cur_st_feature,
                                            cur_input_embeds[st_start_token_pos1 + num_patches + 1:st_start_token_pos1 + num_patches + 2],
                                            cur_input_embeds[st_start_token_pos1 + num_patches + 2:st_start_token_pos2].detach(),
                                            cur_input_embeds[st_start_token_pos2:st_start_token_pos2 + num_patches + 2],
                                            cur_input_embeds[st_start_token_pos2 + num_patches + 2:st_start_token_pos3].detach(),
                                            cur_input_embeds[st_start_token_pos3:st_start_token_pos3 + num_patches + 2],
                                            cur_input_embeds[st_start_token_pos3 + num_patches + 2:].detach()), dim=0)
            cur_st_idx += 1
            new_input_embeds.append(cur_new_input_embeds)
        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        # processed by llama
        outputs = self.llama_model.model.forward(input_ids=None, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        hidden_states = outputs[0]
        batch_size = hidden_states.shape[0]
        if labels is not None:
            st_pre_embs1 = hidden_states[:,
                           self.st_start_id0 + 1:self.st_start_id0 + self.feature_dim + 1,
                           :].detach().reshape(batch_size, -1, self.feature_dim, self.hidden_size)
            st_pre_out1 = self.relu(self.st_pred_linear_1(st_pre_embs1))
            st_pre_embs2 = hidden_states[:,
                           self.st_start_id1 + 1:self.st_start_id1 + self.feature_dim + 1,
                           :].reshape(batch_size, -1, self.feature_dim, self.hidden_size)
            st_pre_out2 = self.relu(self.st_pred_linear_3(st_pre_embs2))
            # 256 --> 12
            st_pre_final = self.st_pred_linear_2(torch.cat([st_pre_out1, st_pre_out2], dim=-1))
            # 12 --> 2
            st_pre_final = st_pre_final.transpose(-1, -2)
        else:
            self.st_pre_res.append(hidden_states.clone())
            
        logits = self.llama_model.lm_head(hidden_states)
        if labels is not None:
             # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)

            if len(st_data_y) > 1:
                st_data_y = torch.cat(st_data_y, dim=0)
                labels_stpre = st_data_y[:, :, region_id[0]:region_id[0] + 1, :self.feature_dim].transpose(1, 2).to(torch.bfloat16)
            else:
                labels_stpre = st_data_y[0][0:1, :, region_id[0]:region_id[0] + 1, :self.feature_dim].transpose(1, 2).to(torch.bfloat16)

            regress_idx_list = []
            regress_result_list = []
            for i in range(batch_size):
                regress_idx_list.append(i)
                regress_result_list.append(st_pre_final[i:i + 1, ...])
            regress_result = torch.cat(regress_result_list, dim=0) # (4, 1, 12, 2)
            return labels_stpre, regress_result, shift_logits, shift_labels
        assert(False, "No labels provided!")

    def calculate_loss(self, batch, lap_mx=None):
        labels_stpre, regress_result, shift_logits, shift_labels = self.forward(batch, lap_mx)
        loss_fct = CrossEntropyLoss()
        rec_loss = scaler_mae_loss(scaler=None, mask_value=None)
        loss_regress = rec_loss(regress_result, labels_stpre)

        loss = loss_fct(shift_logits, shift_labels) + loss_regress
        return loss
    
    def predict1(self, batch, lap_mx=None):
        labels_stpre, regress_result, shift_logits, shift_labels = self.forward(batch, lap_mx)
        return labels_stpre, regress_result

    def predict(self, batch, lap_mx=None):
        input_ids = batch['input_ids']
        st_data_x = batch['st_data_x']
        st_data_y = batch['st_data_y']
        region_id = batch['region_id']
        inputs_embeds = self.llama_model.model.embed_tokens(input_ids)
        if len(st_data_x) > 1:
            st_data_x = torch.cat(st_data_x, dim=0)
        if type(st_data_x) is list:
            STE_out = self.st_tower(st_data_x[0][..., :2], lap_mx)
            if STE_out.shape[2] >= 1:
                region_select_out = STE_out[:, :, region_id[0]:region_id[0] + 1, :].to(torch.bfloat16)
        else:
            STE_out = self.st_tower(st_data_x[..., :2], lap_mx)
            region_select_out = STE_out[:, :, region_id[0]:region_id[0] + 1, :].to(torch.bfloat16)
        st_projector_out = self.st_projector(region_select_out.transpose(1, -1))
        new_input_embeds = []
        cur_st_idx = 0
        for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
            cur_st_feature = st_projector_out[cur_st_idx]
            cur_st_feature = cur_st_feature.reshape(cur_st_feature.shape[0], -1)
            num_patches = cur_st_feature.shape[0]
            st_start_tokens = torch.where(cur_input_ids == self.st_start_token)[0]
            # insert st embedding into the prompt
            if st_start_tokens.shape[0] >= 3:
                st_start_token_pos1, st_start_token_pos2, st_start_token_pos3 = st_start_tokens[0], st_start_tokens[1], st_start_tokens[2]
                self.st_start_id0 = st_start_token_pos1
                self.st_start_id1 = st_start_token_pos3
                
                cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos1].detach(),
                                                cur_input_embeds[st_start_token_pos1:st_start_token_pos1 + 1],
                                                cur_st_feature,
                                                cur_input_embeds[st_start_token_pos1 + num_patches + 1:st_start_token_pos1 + num_patches + 2],
                                                cur_input_embeds[st_start_token_pos1 + num_patches + 2:st_start_token_pos2].detach(),
                                                cur_input_embeds[st_start_token_pos2:st_start_token_pos2 + num_patches + 2],
                                                cur_input_embeds[st_start_token_pos2 + num_patches + 2:st_start_token_pos3].detach(),
                                                cur_input_embeds[st_start_token_pos3:st_start_token_pos3 + num_patches + 2],
                                                cur_input_embeds[st_start_token_pos3 + num_patches + 2:].detach()), dim=0)
            else:
                st_start_token_pos = st_start_tokens[0]
                self.st_start_id0 = st_start_token_pos
                
                cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos].detach(),
                                                cur_input_embeds[st_start_token_pos:st_start_token_pos + 1],
                                                cur_st_feature,
                                                cur_input_embeds[st_start_token_pos + num_patches + 1:st_start_token_pos + num_patches + 2],
                                                cur_input_embeds[st_start_token_pos + num_patches + 2:].detach()), dim=0)
            cur_st_idx += 1
            new_input_embeds.append(cur_new_input_embeds)
        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        # processed by llama
        outputs = self.llama_model.model.forward(input_ids=None, inputs_embeds=inputs_embeds)
        hidden_states = outputs[0]
        batch_size = hidden_states.shape[0]
        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        # generate
        output_ids = self.llama_model.generate(
            inputs = input_ids,
            do_sample = True,
            temperature = 0.01,
            max_new_tokens=256,
            stopping_criteria=[stopping_criteria]
        )
        start_inx = torch.where(output_ids[0, :] == 32001)[0]
        st_pre_embs1 = hidden_states[:,
                        self.st_start_id0 + 1:self.st_start_id0 + self.feature_dim + 1,
                        :].detach().reshape(batch_size, -1, self.feature_dim, self.hidden_size)
        # 4096 --> 128
        error_i = 0
        st_pre_out1 = self.relu(self.st_pred_linear_1(st_pre_embs1))
        if start_inx.shape[0] == 3:
            print('hidden_state', hidden_states.shape[1])
            if hidden_states.shape[1] > start_inx[2] + 1 + self.feature_dim:
                    st_pre_embs2 = hidden_states[:, start_inx[2] + 1:start_inx[2] + 1 + self.feature_dim, :]
            else:
                self._logger.info('========error========')
                error_i = error_i + 2
                self._logger.info(error_i)
                # self._logger.info(hidden_states.shape, start_inx[2] + 1)
                st_pre_embs2 = hidden_states[:, -(2+self.feature_dim):-2, :]
        else:
            self._logger.info('========error========')
            error_i = error_i + 1
            self._logger.info(error_i)
            st_pre_embs2 = hidden_states[:, -(2+self.feature_dim):-2, :]
        st_pre_embs2 = st_pre_embs2.reshape(batch_size, -1, self.feature_dim, self.hidden_size)
        # 4096 --> 128
        st_pre_out2 = self.relu(self.st_pred_linear_3(st_pre_embs2))
        # 256 --> 12
        st_pre_final = self.st_pred_linear_2(torch.cat([st_pre_out1, st_pre_out2], dim=-1))
        # 12 --> 2
        st_pre_final = st_pre_final.transpose(-1, -2)
        if len(st_data_y) > 1:
            st_data_y = torch.cat(st_data_y, dim=0)
            labels_stpre = st_data_y[:, :, region_id[0]:region_id[0] + 1, :self.feature_dim].transpose(1, 2).to(torch.bfloat16)
        else:
            labels_stpre = st_data_y[0][0:1, :, region_id[0]:region_id[0] + 1, :self.feature_dim].transpose(1, 2).to(torch.bfloat16)

        regress_idx_list = []
        regress_result_list = []
        for i in range(batch_size):
            regress_idx_list.append(i)
            regress_result_list.append(st_pre_final[i:i + 1, ...])
        regress_result = torch.cat(regress_result_list, dim=0) # (4, 1, 12, 2)
        
        
        outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=False)[0]
        outputs = outputs.strip()
        if outputs.endswith('###'):
            outputs = outputs[:-len('###')]
        outputs = outputs.strip()
        self._logger.info(outputs)
        return labels_stpre, regress_result


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False