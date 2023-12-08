import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
import argparse
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--mask_flag', type=int, default=0, help='attention_mask_flag')
parser.add_argument('--pe_3d', type=int, default=0, help='0:learnable 1:3d learnable 2:Roep')
parser.add_argument('--gau', type=int, default=0, help='GAU control')
parser.add_argument('--LEB', type=int, default=0, help='language embeding')
parser.add_argument('--reluSquared', type=int, default=0, help='language embeding')

parser.add_argument('--res_attention', type=int, default=1, help='res_attention')

parser.add_argument('--rnn_matrix', type=int, default=0, help='rnn_matrix')

parser.add_argument('--ffn', type=int, default=1, help='ffn')

parser.add_argument('--resi_dual', type=int, default=0, help='dual_norm')
parser.add_argument('--norm', type=int, default=1, help='batch_norm :1   layer_norm:0 rms_norm 2')

parser.add_argument('--qkv_bias', type=int, default=1, help='true :1  false:0')
parser.add_argument('--gb', type=int, default=0, help='1:GAU all segment  0: GAU one dim')

parser.add_argument('--optim', type=str, default='Adam', help='Adam Adamw SGD ')
parser.add_argument('--attention', type=int, default=0, help='0, 1, 2, 3')


# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()
from data_provider.data_factory import data_provider
if __name__ == '__main__':
    # 步骤 1: 准备数据

    train_data_set, train_data_loader = data_provider(args, 'train')
    test_data_set, test_data_loader = data_provider(args, 'test')

    # scaler = StandardScaler()
    # # 步骤 2: 分割数据为训练集和测试集
    # train_size = int(len(data) * 0.7)
    # test_size = int(len(data) * 0.2)
    # train_data = data.iloc[:train_size]
    # test_data = data.iloc[-test_size:]
    # scaler.fit(train_data)

    train_data = train_data_set.data_x
    dim = train_data.shape[1]
    model_arr=[]

    for i in range(dim//25+1):
        if(dim>(i+1)*25):
            train_sub_data = train_data[:,i*25:(i+1)*25]
        else:
            train_sub_data = train_data[:,i*25:dim]

        diff_train_data = np.diff(train_sub_data, axis=0)

        model = VAR(diff_train_data)
        results = model.fit(maxlags=args.seq_len-1, ic='aic')
        lag_order = results.k_ar
        print('lag_order:'+str(lag_order))
        model_arr.append([lag_order,results])


    # 步骤 3: 差分训练数据以实现平稳性
    # diff_train_data = np.diff(train_data,axis=0)
    #
    # # 步骤 4: 创建和训练 VAR 模型
    # model = VAR(diff_train_data)
    # results = model.fit(maxlags=10, ic='aic')
    #
    # # 步骤 5: 预测
    #
    # lag_order = results.k_ar
    # print('lag_order:'+str(lag_order))




    for seq in (96,192,336,720):
        preds = []
        trues = []
        args.pred_len = seq
        test_data_set, test_data_loader = data_provider(args, 'test')
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_data_loader):
            for x, y in zip(batch_x,batch_y):
                x_diff= x.diff(axis=0)
                forecast_array=[]
                for i in range(dim//25+1):
                    lag_order,results = model_arr[i]
                    if (dim > (i + 1) * 25):
                        x_diff_sub = x_diff[:, i * 25:(i + 1) * 25]
                    else:
                        x_diff_sub = x_diff[:, i * 25:dim]

                    forecast = results.forecast(x_diff_sub[-lag_order:], steps=seq)
                    # 步骤 6: 将差分预测转换回原始尺度
                    # 累计求和（逆差分）
                    forecast_array.append(forecast)
                forecast_array=np.concatenate(forecast_array, 1)

                forecast_final = np.cumsum(np.vstack((x[-1], forecast_array)), axis=0)[1:]
                preds.append(forecast_final)
                trues.append(y[-seq:].numpy())

        preds = np.array(preds)
        trues = np.array(trues)

        # 步骤 7: 计算测试集上的 MSE 和 MAE
        mse = np.mean(np.abs(preds - trues))
        mae = np.mean((preds - trues) ** 2)
        print(f'MSE: {mse}')
        print(f'MAE: {mae}')
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write("VAR_" + str(args.data_path) + "_" + str(args.seq_len) + "_" + str(seq) + "\n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
