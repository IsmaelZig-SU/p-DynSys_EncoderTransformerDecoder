import torch, pickle, os
import pandas as pd
from src.Experiment import Experiment


torch.manual_seed(99)
import argparse

if __name__ == "__main__":
    #Parsing arguments
    parser = argparse.ArgumentParser(description='p-Variational-EncoderTransformerDecoder')

    #Training Params
    parser.add_argument('--load_epoch',             type = int, default = 0 ,        help = "loads model at a particular epoch for training")
    parser.add_argument('--dynsys',                 type = str, default = "2DCyl", help = "Choose Dynamical System to train: 1)KS 2)2DCyl 3)ExpData 4)Duffing")
    parser.add_argument('--pred_horizon',           type = int, default = 10,        help = "Number of steps to predict over while calculating loss")
    parser.add_argument('--nbr_ext_var',           type = int, default = 2,        help = "Number of external variables (Beta, Re, ...)")

    #Models
    parser.add_argument('--seq_model',  type = str, default = "TransformerModel",  help = "Sequence model to be used for the training")
    parser.add_argument('--AE_Model',   type = str, default = "Autoencoder", help = "Autoencoder model to be used for the training")

    #training Params ARGS
    parser.add_argument('--lr',      type = float, default=5e-5)
    parser.add_argument('--nepochs', type = int,   default=5000, help = "Number of epochs for training")
    parser.add_argument('--lambda_ResL',        type = float, default=1.0, help = "Controlling Parameter for Sequence Model prediction")
    parser.add_argument('--lambda_stateloss', type = float, default = 1.0, help = "Direct supervision weight")
    parser.add_argument('--beta_VAE', type = float, default = 5e-5, help = "regularization beta term for KLD in VAE")


    #AUTOENCODER Params ARGS
    parser.add_argument('--num_obs',            type = int,   default=4,   help = "Latent Size of the Autoencoder")
    parser.add_argument('--linear_autoencoder',    action = 'store_true',     help = "use linear autoencoder")

    #Transfo Params ARGS
    parser.add_argument('--seq_len',          type = int,   default=10,     help = "length of the sequence for memory term")
    parser.add_argument('--seq_model_weight', type = float, default = 1.0, help = "sequence model weight")
    parser.add_argument('--nattblocks',       type = int,   default=1,     help = "Number of attention blocks in the transformer")
    parser.add_argument('--nheads',       type = int,   default=8,     help = "Number of attention heads")
    parser.add_argument('--hidden_dim',       type = int,   default=64,     help = "Number of attention heads")

    #Data Params ARGS
    parser.add_argument('--ntransients', type = int,   default = 0, help = "number of trainsients to discard in the intial part of the dataset")
    parser.add_argument('--nenddata',    type = int,   default = None,  help = "if we want to skip last parts of the dataset")
    parser.add_argument('--bs',          type = int,   default = 64 ,   help = "BatchSize")
    parser.add_argument('--train_size',  type = float, default = 0.9,   help = "Train Data Proportion")
    parser.add_argument('--norm_input',  action = 'store_true', default = False, help = "normalises input")
    parser.add_argument('--noise_p',     type = float, default = 0.00,  help = "percentage noise to add to the data")


    #Directory Params ARGS
    parser.add_argument('--exp_dir',         type = str, default = "Trained_Models/2DCyl_new/",   help = "Directory for the Experiment")
    parser.add_argument('--load_exp_name',   type = str, default = "",   help = "Name of the experiment to be loaded")
    parser.add_argument('--data_dir',        type = str, default = "Data/2DCylinder/processed_data/npyfiles/data.npy", help = "Directory for the Data")
    parser.add_argument('--nsave',           type = int,   default = 500, help = "save every nsave number of epochs")
    parser.add_argument('--no_save_model',   action = 'store_false',     help = "doesn't save model")
    parser.add_argument('--time_sample', type = int,   default = 10,    help = "time sampling size")
    parser.add_argument('--info',            type = str, default = "_",  help = "extra infomration to be added to the experiment name")

    args = parser.parse_args()
    #############################################################################
    
    ##Running from scratch  
    if args.load_epoch == 0:
        exp = Experiment(args)
        exp.main_train(load_model = False)
    if args.load_epoch != 0:
        exp = Experiment(args)
        exp.main_train(load_model = True)

