import os
import json
import numpy as np
import datetime
from tqdm import tqdm
from libcity.data.dataset import TrafficStateGridDataset
from libcity.data.utils import generate_dataloader


class MyModelGridDataset(TrafficStateGridDataset):

    def __init__(self, config):
        super().__init__(config)
        self.traffic_type = self.config.get('traffic_type', 'traffic')
        self.loc = self.config.get('loc', 'unknown place')
        self.random_regions = 40
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'mymodel_grid_based_{}.npz'.format(self.parameters_str))
        # self.tokenizer
        self.feature_name = {
            'input_ids': 'int',
            'labels': 'int',
            'X': 'float',
            'y': 'float',
            'region_id': 'int'
        }
    
    def _load_rel(self):
        super()._load_grid_rel()
        
    def _generate_train_val_test(self):
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'chosen_data_id_40.json')
        with open(file_path, 'r') as f:
            id_list = np.array(json.load(f)[self.dataset][0])
        x, y = self._generate_data()
        x = x[:, :, id_list, :]
        y = y[:, :, id_list, :]
        return self._split_train_val_test(x, y)
    
    def gen_timeslot_str(self, timeslot_id):
        time_strs = []
        seq = [timeslot_id, 
               timeslot_id + self.input_window - 1,
               timeslot_id + self.input_window, 
               timeslot_id + self.input_window + self.output_window - 1]
        for id in seq:  
            d = self.timesolts[timeslot_id].astype("datetime64[s]")
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
        
    def gen_final_data(self, timeslot_id, region_id, flows):
        # F * T
        BEGIN_SIGNAL = '###'
        END_SIGNAL = "\n"
        header = "A chat between a curious human and an artificial intelligence assistant. \
                 The assistant gives helpful, detailed, and polite answers to the human's questions."
        time_strs = self.gen_timeslot_str(timeslot_id)
        if len(flows) == 2:
            prompts = self.gen_in_out_flow_prompt(flows, time_strs)
        elif len(flows) == 1:
            prompts = self.gen_single_flow_prompt(flows, time_strs)
        else:
            raise ValueError('unknown input')
        final_prompt = header + END_SIGNAL
        for i in prompts:
            final_prompt += BEGIN_SIGNAL + i + END_SIGNAL
        final_prompt += BEGIN_SIGNAL
        print(final_prompt)
        assert(False)
            
    def process(self, data_x):
        final_data = []
        data_x = data_x.transpose(0, 2, 3, 1) # B N F T
        for i in range(len(data_x)):
            cur_time_state = data_x[i]
            for j in range(len(cur_time_state)):
                flows = cur_time_state[j]
                final_data.append(self.gen_final_data(i, j, flows))
        return final_data            
                

    def get_data(self):
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
        self.feature_dim = x_train.shape[-1]
        self.process(x_train)

        
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample,
                                distributed=self.distributed)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "sd_mx": self.sd_mx, "sh_mx": self.sh_mx,
                "ext_dim": self.ext_dim, "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches,
                "dtw_matrix": self.dtw_matrix, "pattern_keys": self.pattern_keys}
