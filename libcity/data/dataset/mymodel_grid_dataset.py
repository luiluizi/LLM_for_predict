import os
import json
import copy
import torch
import numpy as np
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from libcity.data.dataset import TrafficStateGridDataset
from libcity.data.list_dataset import ListDataset
from transformers import AutoTokenizer
from libcity.utils import ensure_dir
import pickle

IGNORE_INDEX = -100
DEFAULT_ST_PATCH_TOKEN = "<ST_patch>"
DEFAULT_ST_START_TOKEN = "<ST_start>"
DEFAULT_ST_END_TOKEN = "<ST_end>"

class MyModelGridDataset(TrafficStateGridDataset):

    def __init__(self, config):
        super().__init__(config)
        self.traffic_type = self.config.get('traffic_type', 'traffic')
        self.loc = self.config.get('loc', 'unknown place')
        self.random_regions = 40
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'mymodel_grid_based_{}'.format(self.parameters_str))
        self.train_cache_file_name = self.cache_file_name + "_train.pkl"
        self.test_cache_file_name = self.cache_file_name + "_test.pkl"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/home/panda/private/jjw/hck/br/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            model_max_length = 2048,
            padding_side = "right",
            use_fast = False
        )
        self._initialize_tokenizer()
    
    def _load_rel(self):
        super()._load_grid_rel()
    
    def _initialize_tokenizer(self):
        self.tokenizer.add_tokens([DEFAULT_ST_PATCH_TOKEN, DEFAULT_ST_START_TOKEN, DEFAULT_ST_END_TOKEN], special_tokens=True)
        self.tokenizer.pad_token = self.tokenizer.unk_token
    
    def gen_timeslot_str(self, timeslot_id):
        time_strs = []
        seq = [timeslot_id, 
               timeslot_id + self.input_window - 1,
               timeslot_id + self.input_window, 
               timeslot_id + self.input_window + self.output_window - 1]
        for id in seq:  
            d = self.timesolts[id].astype("datetime64[s]")
            d = datetime.datetime.strptime(str(d), '%Y-%m-%dT%H:%M:%S')
            formatted_date = str(d.strftime("%B %d, %Y, %H:%M, %A"))
            time_strs.append(formatted_date)
        return time_strs
        
    def gen_in_out_flow_prompt(self, flows, time_strs):
        time_interval = self.time_intervals//60
        human_str = f"human: Given the historical data for {self.traffic_type} flow over {self.input_window} time steps in a specific region of {self.loc} City, \
the recorded {self.traffic_type} inflows are {str(flows[0])}, and the recorded {self.traffic_type} outflows are {str(flows[1])}. \
The recording time of the historical data is '{time_strs[0]} to {time_strs[1]}, \
with data points recorded at {time_interval}-minute intervals'. Now we want to predict the {self.traffic_type} inflow and outflow for \
the next {self.output_window} time steps during the time period of '{time_strs[2]} to {time_strs[3]}, \
with data points recorded at {time_interval}-minute intervals'. To improve prediction accuracy, a spatio-temporal model is utilized \
to encode the historical {self.traffic_type} data as tokens <ST_start><ST_patch><ST_patch><ST_end>, where the first and the second tokens correspond to the representations \
of {self.traffic_type} inflow and outflow. Please conduct an analysis of the traffic patterns in this region, taking into account \
the provided time and regional information, and then generate the predictive tokens for regression, in the form \"<ST_start><ST_patch><ST_patch><ST_end>\"."
        gpt_str = f"gpt: Based on the given information, the predicted tokens of {self.traffic_type} inflow and outflow in this region are <ST_start><ST_patch><ST_patch><ST_end>."
        return [human_str, gpt_str]
        
    def gen_single_flow_prompt(self, flows, time_strs):
        #TODO
        return None
        
    def _tokenize_fn(self, text):
        tokenzied_text = self.tokenizer(
            text,
            return_tensors='pt',
            padding='longest',
            max_length= self.tokenizer.model_max_length,
            truncation=True,
        )
        input_ids = labels = tokenzied_text.input_ids[0]
        input_ids_lens = labels_len = tokenzied_text.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_len=labels_len,
        )
    
    def gen_final_data(self, timeslot_id, region_id, flows, st_data_x, st_data_y):
        # F * T
        BEGIN_SIGNAL = '###'
        END_SIGNAL = "\n"
        header = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        time_strs = self.gen_timeslot_str(timeslot_id)
        if len(flows) == 2:
            pre_prompts = self.gen_in_out_flow_prompt(flows, time_strs)
        elif len(flows) == 1:
            pre_prompts = self.gen_single_flow_prompt(flows, time_strs)
        else:
            raise ValueError('unknown input')
        # connect
        prompts = []
        final_prompt = header + END_SIGNAL
        prompts.append(header + END_SIGNAL)
        for prompt in pre_prompts:
            final_prompt += BEGIN_SIGNAL + prompt + END_SIGNAL
            prompts.append(BEGIN_SIGNAL + prompt + END_SIGNAL)
        final_prompt += BEGIN_SIGNAL
        text_tokenized = self._tokenize_fn(final_prompt)
        input_ids = text_tokenized['input_ids']
        targets = copy.deepcopy(input_ids)
        tokenized_lens = [
            self._tokenize_fn(
                prompt
            )["input_ids_lens"] for prompt in prompts
        ]
        # adding mask
        cur_idx = tokenized_lens[0]
        targets[:cur_idx] = IGNORE_INDEX
        targets[cur_idx + 2:cur_idx + tokenized_lens[1]] = IGNORE_INDEX
        data_dict = dict(input_ids=input_ids, labels=targets)
        data_dict['st_data_x'] = torch.Tensor(st_data_x).unsqueeze(0)
        data_dict['st_data_y'] = torch.Tensor(st_data_y).unsqueeze(0)
        data_dict['region_id'] = region_id
        return data_dict
            
    def process(self, raw_data_x, raw_data_y):
        final_data = []
        data_x = raw_data_x.transpose(0, 2, 3, 1) # B N F T
        kk = 0
        print(len(data_x)*len(data_x[0]))
        for i in range(len(data_x)):
            cur_time_state = data_x[i]
            for j in range(len(cur_time_state)):
                flows = cur_time_state[j]
                final_data.append(self.gen_final_data(i, j, flows, raw_data_x[i], raw_data_y[i]))
        return final_data  
    
    def _generate_train_val_test(self):
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'chosen_data_id_40.json')
        with open(file_path, 'r') as f:
            id_list = np.array(json.load(f)[self.dataset][0])
        x, y = self._generate_data()
        x = x[:, :, id_list, :]
        y = y[:, :, id_list, :]
        self.feature_dim = x.shape[-1]
        return self._split_train_val_test(x, y)
    
    def _split_train_val_test(self, x, y):
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        x_train, y_train = x[int(num_train*(1 - self.part_train_rate)):num_train], y[int(num_train*(1 - self.part_train_rate)):num_train]
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        x_test, y_test = x[-num_test:], y[-num_test:]
        
        train_data = self.process(x_train, y_train)
        test_data = self.process(x_test, y_test)
        train_data_dict = {
            'input_ids': [d['input_ids'] for d in train_data],
            'labels': [d['labels'] for d in train_data],
            'st_data_x': [d['st_data_x'] for d in train_data],
            'st_data_y': [d['st_data_y'] for d in train_data],
            'region_id': [d['region_id'] for d in train_data],
        }
        test_data_dict = {
            'input_ids': [d['input_ids'] for d in test_data],
            'labels': [d['labels'] for d in test_data],
            'st_data_x': [d['st_data_x'] for d in test_data],
            'st_data_y': [d['st_data_y'] for d in test_data],
            'region_id': [d['region_id'] for d in test_data],
        }
        if self.rank == 0 and self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            # np.savez_compressed(self.train_cache_file_name, **train_data_dict)
            # np.savez_compressed(self.test_cache_file_name, **test_data_dict)
            with open(self.train_cache_file_name, 'wb') as f:
                pickle.dump(train_data_dict, f)

            with open(self.test_cache_file_name, 'wb') as f:
                pickle.dump(test_data_dict, f)
            self._logger.info('Saved at ' + self.train_cache_file_name + ", and"+ self.test_cache_file_name)
        return train_data, test_data
    
    def _load_cache_train_val_test(self):
        self._logger.info('Loading ' + self.cache_file_name)
        # train_cat_data = np.load(self.train_cache_file_name)
        # test_cat_data = np.load(self.test_cache_file_name)
        with open(self.train_cache_file_name, 'rb') as f:
            train_cat_data = pickle.load(f)
        with open(self.test_cache_file_name, 'rb') as f:
            test_cat_data = pickle.load(f)
        train_data = [{'input_ids': input_id, 'labels': label, 'st_data_x': st_data_x, 'st_data_y': st_data_y, 'region_id': region_id} 
              for input_id, label, st_data_x, st_data_y, region_id in zip(train_cat_data['input_ids'], train_cat_data['labels'], train_cat_data['st_data_x'], train_cat_data['st_data_y'], train_cat_data['region_id'])]
        test_data = [{'input_ids': input_id, 'labels': label, 'st_data_x': st_data_x, 'st_data_y': st_data_y, 'region_id': region_id} 
              for input_id, label, st_data_x, st_data_y, region_id in zip(test_cat_data['input_ids'], test_cat_data['labels'], test_cat_data['st_data_x'], test_cat_data['st_data_y'], test_cat_data['region_id'])]
        self.feature_dim = train_data[0]['st_data_x'].shape[-1]
        return train_data, test_data
                

    def get_data(self):
        train_data, test_data = [], []
        self._load_grid_3d(self.data_files[0]) #get time information
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.train_cache_file_name) and os.path.exists(self.test_cache_file_name):
                train_data, test_data = self._load_cache_train_val_test()
            else:
                train_data, test_data = self._generate_train_val_test()
        # train_data = self.process(x_train, y_train)
        # test_data = self.process(x_test, y_test)
        
        # train_data = None
        # eval_data = train_data
        # test_data = train_data
        self.train_dataloader, self.test_dataloader = \
            self.generate_dataloader(train_data, test_data, self.batch_size, self.num_workers)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, None, self.test_dataloader
    
    def generate_dataloader(self, train_data, test_data,
                        batch_size, num_workers, shuffle=True):
        train_dataset = ListDataset(train_data)
        test_dataset = ListDataset(test_data)
        train_sampler = None

        def collator(indices):
            input_ids, labels = tuple([indice[key] for indice in indices] for key in ("input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
            batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )

            st_data_x_batch = [indice['st_data_x'] for indice in indices]
            st_data_y_batch = [indice['st_data_y'] for indice in indices]
            region_id_batch = [indice['region_id'] for indice in indices]
            batch['st_data_x'] = st_data_x_batch
            batch['st_data_y'] = st_data_y_batch
            batch['region_id'] = region_id_batch
            return batch
        
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                    num_workers=num_workers, collate_fn=collator,
                                    shuffle=shuffle and train_sampler is None, sampler=train_sampler)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                    num_workers=num_workers, collate_fn=collator,
                                    shuffle=False)
        return train_dataloader, test_dataloader
    
    def get_data_feature(self):
        return {"scaler": self.scaler, "tokenizer": self.tokenizer,
                "st_start_token":self.tokenizer.convert_tokens_to_ids(DEFAULT_ST_START_TOKEN),
                "ext_dim": self.ext_dim, "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches,}
