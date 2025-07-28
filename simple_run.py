import argparse
import os
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Create args object with fixed parameters for Weather dataset experiment
    class Args:
        def __init__(self):
            # Basic config
            self.is_training = 1
            self.model_id = 'weather_96_96'
            self.model = 'iTransformer'
            
            # Data loader
            self.data = 'custom'
            self.root_path = './dataset/weather/'
            self.data_path = 'weather.csv'
            self.features = 'M'  # multivariate predict multivariate
            self.target = 'OT'
            self.freq = 'h'
            self.checkpoints = './checkpoints/'
            
            # Forecasting task
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
            
            # Model define (Weather dataset has 21 features)
            self.enc_in = 21
            self.dec_in = 21
            self.c_out = 21
            self.d_model = 512
            self.n_heads = 8
            self.e_layers = 3
            self.d_layers = 1
            self.d_ff = 512
            self.moving_avg = 25
            self.factor = 1
            self.distil = True
            self.dropout = 0.1
            self.embed = 'timeF'
            self.activation = 'gelu'
            self.output_attention = False
            self.do_predict = False
            
            # Optimization
            self.num_workers = 10
            self.itr = 1
            self.train_epochs = 10
            self.batch_size = 32  # Paper standard batch size
            self.patience = 7
            self.learning_rate = 0.001
            self.des = 'Exp'
            self.loss = 'MSE'
            self.lradj = 'type1'
            self.use_amp = False
            
            # GPU settings
            self.use_multi_gpu = False
            self.devices = '0'
            self.gpu = 0
            
            # iTransformer specific
            self.exp_name = 'MTSF'
            self.channel_independence = False
            self.inverse = False
            self.class_strategy = 'projection'
            self.target_root_path = './dataset/weather/'
            self.target_data_path = 'weather.csv'
            self.efficient_training = False
            self.use_norm = True
            self.partial_start_index = 0

    args = Args()
    
    # Device detection: prioritize CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        args.use_gpu = True
        args.device = 'cuda'
        print("🚀 Using CUDA GPU acceleration")
    elif torch.backends.mps.is_available():
        args.use_gpu = False  # Set to False to avoid CUDA path in exp_basic.py
        args.device = 'mps'
        print("🍎 Using MPS (Apple Silicon) GPU acceleration")
    else:
        args.use_gpu = False
        args.device = 'cpu'
        print("💻 Using CPU (no GPU acceleration available)")

    print(f"Device: {args.device}")
    print(f"Experiment: {args.model_id}")
    print(f"Dataset: {args.data_path}")
    print(f"Sequence length: {args.seq_len} -> Prediction length: {args.pred_len}")
    print("-" * 50)

    # Custom experiment class for MPS support
    class Exp_MPS(Exp_Long_Term_Forecast):
        def _acquire_device(self):
            if args.device == 'mps':
                device = torch.device('mps')
                print('Use GPU: mps')
            elif self.args.use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
            else:
                device = torch.device('cpu')
                print('Use CPU')
            return device

    # Run the experiment
    Exp = Exp_MPS if args.device == 'mps' else Exp_Long_Term_Forecast

    for ii in range(args.itr):
        # Setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        # Clear cache based on device type
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        elif args.device == 'mps':
            torch.mps.empty_cache()

    print("🎉 Experiment completed!")