import numpy as np
import pandas as pd
import os
from utils import *
from sklearn.model_selection import StratifiedKFold, KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop
from transformers import BertTokenizer
from model import Eyettention
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.nn.functional import cross_entropy, softmax
from collections import deque
import pickle
import json
import matplotlib.pyplot as plt
import argparse
from config import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run Eyettention on MECO dataset')
    parser.add_argument(
        '--test_mode',
        help='New Sentence Split: text, New Reader Split: subject',
        type=str,
        default='text'
    )
    parser.add_argument(
        '--atten_type',
        help='attention type: global, local, local-g',
        type=str,
        default='local-g'
    )
    parser.add_argument(
        '--save_data_folder',
        help='folder path for saving results',
        type=str,
        default='./results/MECO/'
    )
    parser.add_argument(
        '--scanpath_gen_flag',
        help='whether to generate scanpath',
        type=int,
        default=1
    )
    parser.add_argument(
        '--max_pred_len',
        help='if scanpath_gen_flag is True, you can determine the longest scanpath that you want to generate, which should depend on the sentence length',
        type=int,
        default=60
    )
    parser.add_argument(
        '--gpu',
        help='gpu index',
        type=int,
        default=0
    )
    args = parser.parse_args()
    gpu = args.gpu

    torch.set_default_tensor_type(torch.FloatTensor)
    availbl = torch.cuda.is_available()
    print(torch.cuda.is_available())
    if availbl:
        device = f'cuda:{gpu}'
    else:
        device = 'cpu'
    torch.cuda.set_device(gpu)

    cf = {
        "model_pretrained": "bert-base-multilingual-cased",
        "lr": 1e-3,
        "max_grad_norm": 10,
        "n_epochs": 1000,
        "n_folds": 5,
        "dataset": 'MECO',
        "atten_type": args.atten_type,
        "batch_size": 256,
        "max_sn_len": 27,  # include start token and end token
        "max_sp_len": 40,  # include start token and end token
        "norm_type": "z-score",
        "earlystop_patience": 20,
        "max_pred_len": args.max_pred_len
    }

    # Encode the label into integer categories
    le = LabelEncoder()
    valid_labels = np.arange(-cf["max_sn_len"]+3, cf["max_sn_len"])
    le.fit(np.unique(valid_labels))

    # Load corpus with proper column names for MECO
    word_info_df, pos_info_df, eyemovement_df = load_corpus(cf["dataset"])
    
    eyemovement_df['subid'] = pd.to_numeric(
    eyemovement_df['subid'].str.replace(r'\D+', '', regex=True), 
    errors='coerce')
    # Rename columns to match expected names in the model
    eyemovement_df = eyemovement_df.rename(columns={
        'word_position': 'wn',
        'landing_pos_norm': 'fl',
        'fixation_dur': 'dur',
        'subid': 'id',
        'sentnum': 'sn'
    })
    
    # Make list with sentence index
    sn_list = np.unique(eyemovement_df['sn'].values).tolist()
    # Make list with reader index
    reader_list = np.unique(eyemovement_df['id'].values).tolist()

    # Split training&test sets
    split_list = sn_list if args.test_mode == 'text' else reader_list

    n_folds = cf["n_folds"]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    fold_indx = 0
    
    sp_dnn_list = []
    sp_human_list = []
    
    for train_idx, test_idx in kf.split(split_list):
        loss_dict = {'val_loss': [], 'train_loss': [], 'test_ll': []}
        list_train = [split_list[i] for i in train_idx]
        list_test = [split_list[i] for i in test_idx]

        # Create train validation split
        kf_val = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        for train_index, val_index in kf_val.split(list_train):
            break  # we only evaluate a single fold
        
        list_train_net = [list_train[i] for i in train_index]
        list_val_net = [list_train[i] for i in val_index]

        if args.test_mode == 'text':
            sn_list_train = list_train_net
            sn_list_val = list_val_net
            sn_list_test = list_test
            reader_list_train, reader_list_val, reader_list_test = reader_list, reader_list, reader_list
        elif args.test_mode == 'subject':
            reader_list_train = list_train_net
            reader_list_val = list_val_net
            reader_list_test = list_test
            sn_list_train, sn_list_val, sn_list_test = sn_list, sn_list, sn_list

        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained(cf['model_pretrained'])
        
        # Preparing batch data with MECO dataset class
        dataset_train = MECOdataset(word_info_df, eyemovement_df, cf, reader_list_train, sn_list_train, tokenizer)
        train_dataloader = DataLoader(dataset_train, batch_size=cf["batch_size"], shuffle=True, drop_last=True)

        dataset_val = MECOdataset(word_info_df, eyemovement_df, cf, reader_list_val, sn_list_val, tokenizer)
        val_dataloader = DataLoader(dataset_val, batch_size=cf["batch_size"], shuffle=False, drop_last=True)

        dataset_test = MECOdataset(word_info_df, eyemovement_df, cf, reader_list_test, sn_list_test, tokenizer)
        test_dataloader = DataLoader(dataset_test, batch_size=cf["batch_size"], shuffle=False, drop_last=False)

        # z-score normalization for gaze features
        fix_dur_mean, fix_dur_std = calculate_mean_std(dataloader=train_dataloader, feat_key="sp_fix_dur", padding_value=0, scale=1000)
        landing_pos_mean, landing_pos_std = calculate_mean_std(dataloader=train_dataloader, feat_key="sp_landing_pos", padding_value=0)
        sn_word_len_mean, sn_word_len_std = calculate_mean_std(dataloader=train_dataloader, feat_key="sn_word_len")

        # Update config with normalization parameters
        cf.update({
            'fix_dur_mean': fix_dur_mean,
            'fix_dur_std': fix_dur_std,
            'landing_pos_mean': landing_pos_mean,
            'landing_pos_std': landing_pos_std,
            'sn_word_len_mean': sn_word_len_mean,
            'sn_word_len_std': sn_word_len_std
        })

        # Load model
        dnn = Eyettention(cf)
        dnn.train()
        dnn.to(device)
        
        optimizer = Adam(dnn.parameters(), lr=cf["lr"])
        av_score = deque(maxlen=100)
        old_score = 1e10
        save_ep_couter = 0
        
        print('Start training')
        for episode_i in range(cf["n_epochs"]+1):
            dnn.train()
            print('episode:', episode_i)
            counter = 0
            
            for batch in train_dataloader:
                counter += 1
                
                # Move all batch tensors to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                dnn_out, atten_weights = dnn(
                    sn_emd=batch["sn_input_ids"],
                    sn_mask=batch["sn_attention_mask"],
                    sp_emd=batch["sp_input_ids"],
                    sp_pos=batch["sp_pos"],
                    word_ids_sn=None,
                    word_ids_sp=None,
                    sp_fix_dur=batch["sp_fix_dur"],
                    sp_landing_pos=batch["sp_landing_pos"],
                    sn_word_len=batch["sn_word_len"]
                )
                
                dnn_out = dnn_out.permute(0, 2, 1)  # [batch, dec_o_dim, step]
                
                # Prepare label and mask
                pad_mask, label = load_label(batch["sp_pos"], cf, le, device)
                loss = nn.CrossEntropyLoss(reduction="none")
                valid_loss = torch.masked_select(loss(dnn_out, label), ~pad_mask)
                if valid_loss.numel() == 0:
                    print("Warning: No valid loss values!")
                    batch_error = torch.tensor(0.0, device=device)
                else:
                    batch_error = torch.mean(valid_loss)
                
                # Backward pass
                batch_error.backward()
                gradient_clipping(dnn, cf["max_grad_norm"])
                optimizer.step()
                
                av_score.append(batch_error.to('cpu').detach().numpy())
                print('counter:', counter)
                print('Sample {}\tAverage Error: {:.10f} '.format(counter, np.mean(av_score)), end=" ")
            
            loss_dict['train_loss'].append(np.mean(av_score))

            # Validation
            val_loss = []
            dnn.eval()
            for batch in val_dataloader:
                with torch.no_grad():
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    dnn_out_val, _ = dnn(
                        sn_emd=batch["sn_input_ids"],
                        sn_mask=batch["sn_attention_mask"],
                        sp_emd=batch["sp_input_ids"],
                        sp_pos=batch["sp_pos"],
                        word_ids_sn=None,
                        word_ids_sp=None,
                        sp_fix_dur=batch["sp_fix_dur"],
                        sp_landing_pos=batch["sp_landing_pos"],
                        sn_word_len=batch["sn_word_len"]
                    )
                    
                    dnn_out_val = dnn_out_val.permute(0, 2, 1)
                    pad_mask_val, label_val = load_label(batch["sp_pos"], cf, le, device)
                    batch_error_val = torch.mean(torch.masked_select(
                        nn.CrossEntropyLoss(reduction="none")(dnn_out_val, label_val), 
                        ~pad_mask_val
                    ))
                    val_loss.append(batch_error_val.detach().to('cpu').numpy())
            
            avg_val_loss = np.mean(val_loss)
            print('\nvalidation loss is {} \n'.format(avg_val_loss))
            loss_dict['val_loss'].append(avg_val_loss)

            # Save best model
            if avg_val_loss < old_score:
                torch.save(
                    dnn.state_dict(), 
                    '{}/CELoss_MECO_{}_eyettention_{}_newloss_fold{}.pth'.format(
                        args.save_data_folder, args.test_mode, args.atten_type, fold_indx
                    )
                )
                old_score = avg_val_loss
                print('\nsaved model state dict\n')
                save_ep_couter = episode_i
            else:
                if episode_i - save_ep_couter >= cf["earlystop_patience"]:
                    break

        # Evaluation
        dnn.eval()
        res_llh = []
        
        dnn.load_state_dict(torch.load(
            os.path.join(
                args.save_data_folder, 
                f'CELoss_MECO_{args.test_mode}_eyettention_{args.atten_type}_newloss_fold{fold_indx}.pth'
            ), 
            map_location='cpu'
        ))
        dnn.to(device)
        
        for batch in test_dataloader:
            if torch.isnan(batch["sp_fix_dur"]).any():
                print("NaN detected in sp_fix_dur!")
                continue
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                
                dnn_out_test, _ = dnn(
                    sn_emd=batch["sn_input_ids"],
                    sn_mask=batch["sn_attention_mask"],
                    sp_emd=batch["sp_input_ids"],
                    sp_pos=batch["sp_pos"],
                    word_ids_sn=None,
                    word_ids_sp=None,
                    sp_fix_dur=batch["sp_fix_dur"],
                    sp_landing_pos=batch["sp_landing_pos"],
                    sn_word_len=batch["sn_word_len"]
                )
                
                m = nn.Softmax(dim=2)
                dnn_out_test = m(dnn_out_test).detach().to('cpu').numpy()
                
                pad_mask_test, label_test = load_label(batch["sp_pos"].to('cpu'), cf, le, 'cpu')
                res_batch = eval_log_llh(dnn_out_test, label_test, pad_mask_test)
                res_llh.append(np.array(res_batch))

                if bool(args.scanpath_gen_flag):
                    sn_len = (torch.sum(batch["sn_attention_mask"], axis=1) - 2).detach().to('cpu').numpy()
                    sp_dnn, _ = dnn.scanpath_generation(
                        sn_emd=batch["sn_input_ids"],
                        sn_mask=batch["sn_attention_mask"],
                        word_ids_sn=None,
                        sn_word_len=batch["sn_word_len"],
                        le=le,
                        max_pred_len=cf['max_pred_len']
                    )
                    sp_dnn, sp_human = prepare_scanpath(sp_dnn.detach().to('cpu').numpy(), sn_len, batch["sp_pos"].to('cpu'), cf)
                    sp_dnn_list.extend(sp_dnn)
                    sp_human_list.extend(sp_human)

        res_llh = np.concatenate(res_llh).ravel()
        loss_dict.update({
            'test_ll': [res_llh],
            'fix_dur_mean': fix_dur_mean,
            'fix_dur_std': fix_dur_std,
            'landing_pos_mean': landing_pos_mean,
            'landing_pos_std': landing_pos_std,
            'sn_word_len_mean': sn_word_len_mean,
            'sn_word_len_std': sn_word_len_std
        })

        print('\nTest likelihood is {} \n'.format(np.mean(res_llh)))
        
        # Save results
        with open('{}/res_MECO_{}_eyettention_{}_Fold{}.pickle'.format(
            args.save_data_folder, args.test_mode, args.atten_type, fold_indx
        ), 'wb') as handle:
            pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        fold_indx += 1

    if bool(args.scanpath_gen_flag):
        dic = {"sp_dnn": sp_dnn_list, "sp_human": sp_human_list}
        with open(os.path.join(
            args.save_data_folder, 
            f'MECO_scanpath_generation_eyettention_{args.test_mode}_{args.atten_type}.pickle'
        ), 'wb') as handle:
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)