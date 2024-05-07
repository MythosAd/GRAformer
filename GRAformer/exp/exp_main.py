import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from GRAformer.data_provider.data_factory import data_provider
from GRAformer.exp.exp_basic import Exp_Basic
from GRAformer.models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, PatchNorm, Mlinear, \
    llama,Prompt_TST,MTGNN,LSTNET,DeepVAR,DeepAR,StemGNN,ADLGNN,GRAformer
from GRAformer.utils.metrics import metric
from GRAformer.utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop

# from lion_pytorch import Lion

warnings.filterwarnings('ignore')
os.environ["WANDB_API_KEY"] = "5332e67da7daf0a41f52e3a3ed1377882ed276a2"
os.environ["WANDB_MODE"] = "offline"
# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 128}

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'PatchNorm': PatchNorm,

            'Mlinear': Mlinear,
            'llama': llama,
            'Prompt_TST':Prompt_TST,
            'MTGNN':MTGNN,
            'LSTNET':LSTNET,
            'DeepVAR':DeepVAR,
            'DeepAR' :DeepAR,
            'StemGNN':StemGNN,
            'ADLGNN':ADLGNN,
            'GRAformer': GRAformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        self.optim = self.args.optim
        if self.optim == 'AdamW':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        elif self.optim == 'Adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.optim == 'SGD':
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
       # elif self.optim == 'Lion':
         #   model_optim = Lion(self.model.model.parameters(), lr=1e-4, weight_decay=1e-2)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)


        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'DeepAR' in self.args.model:
                    outputs = self.model(batch_x, batch_y)
                else:
                    if 'Linear' in self.args.model or 'TST' or 'Norm' in self.args.model:
                        outputs = self.model(batch_x)


                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if 'Rnn' in self.args.model or 'Mlinear' in self.args.model:
                    weight_u = outputs[1]
                    outputs = outputs[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        # scheduler = lr_scheduler.CosineLRScheduler(t_initial=300,lr_min=1e-5,warmup_lr_init=1e-6,warmup_t=10,cycle_limit=1)
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(t_initial=300,lr_min=1e-5,warmup_lr_init=1e-6,warmup_t=10,cycle_limit=1)

        batch_x_GBoost=[]
        batch_y_GBoost=[]
        # import xgboost as xgb
        num_params = sum(param.numel() for param in self.model.parameters())
        print("model_params_num:" + str(num_params / 1024 / 1024) + "M")
        print(
            "1 param per 4 byte ,expected_max_memory:" + str(num_params / 1024 / 1024 * 4) + "M")  # 12800000   8969942
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()


            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if 'DeepAR' in self.args.model:
                    outputs = self.model(batch_x,batch_y)
                else:
                    if 'Linear' in self.args.model or 'TST' or 'Norm' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    if 'Rnn' in self.args.model or 'Mlinear' in self.args.model:
                        weight_u = outputs[1]
                        outputs = outputs[0]
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    if 'Rnn' in self.args.model or 'Mlinear' in self.args.model:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} ,weight_u:{3}".format(i + 1, epoch + 1,
                                                                                              loss.item(),
                                                                                              weight_u.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} ".format(i + 1, epoch + 1,
                                                                                 loss.item(),
                                                                                 ))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()


            if 'Rnn' in self.args.model or 'Mlinear' in self.args.model:
                print("Epoch: {} cost time: {}  weight_u : {} ".format(epoch + 1, time.time() - epoch_time,
                                                                       weight_u.item()))
            else:
                print("Epoch: {} cost time: {}   ".format(epoch + 1, time.time() - epoch_time,
                                                          ))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST' and self.args.lradj != 'Lion':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def train_xgboost(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        batch_x_train_GBoost = []
        batch_y_train_GBoost = []
        import xgboost as xgb
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            batch_x_train_GBoost.append(batch_x)
            batch_y_train_GBoost.append(batch_y[:, -self.args.pred_len:, :])

        xgb_model = xgb.XGBRegressor(objective="reg:linear", tree_method = 'gpu_hist',random_state=42)
        # torch.cat(batch_x_GBoost,0).transpose(1,2).flatten(0,1)
        batch_x_train_GBoost = torch.cat(batch_x_train_GBoost, 0).transpose(1, 2).flatten(0, 1)
        batch_y_train_GBoost = torch.cat(batch_y_train_GBoost, 0).transpose(1, 2).flatten(0, 1)

        xgb_model.fit(batch_x_train_GBoost,batch_y_train_GBoost)

        batch_x_test_GBoost = []
        batch_y_test_GBoost = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x_test_GBoost.append(batch_x)
            batch_y_test_GBoost.append(batch_y[:, -self.args.pred_len:, :])

        batch_x_test_GBoost = torch.cat(batch_x_test_GBoost, 0).transpose(1, 2).flatten(0, 1)
        batch_y_test_GBoost = torch.cat(batch_y_test_GBoost, 0).transpose(1, 2).flatten(0, 1)

        y_pred = xgb_model.predict(batch_x_test_GBoost)

        from sklearn.metrics import  mean_squared_error
        mse = mean_squared_error(batch_y_test_GBoost, y_pred)
        print('mse:'+np.sqrt(mse))

    def create_xy(self,series, window_size, prediction_horizon, shuffle=False):
        x = []
        y = []
        series =series.data_x
        for i in range(0, len(series)):
            if len(series[(i + window_size):(i + window_size + prediction_horizon),:]) < prediction_horizon:
                break
            x.append(series[i:(i + window_size),:])
            y.append(series[(i + window_size):(i + window_size + prediction_horizon),:])
        x = np.array(x)
        y = np.array(y)
        return x, y

    def train_lightGBM(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        train_x = []
        train_y = []
        validate_x = []
        validate_y = []
        test_x = []
        test_y = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            train_x.append(batch_x)
            train_y.append(batch_y[:, -self.args.pred_len:, :])

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            validate_x.append(batch_x)
            validate_y.append(batch_y[:, -self.args.pred_len:, :])

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            test_x.append(batch_x)
            test_y.append(batch_y[:, -self.args.pred_len:, :])

        train_x = torch.cat(train_x, 0).transpose(1, 2).flatten(0, 1)
        train_y = torch.cat(train_y, 0).transpose(1, 2).flatten(0, 1)
        validate_x = torch.cat(validate_x, 0).transpose(1, 2).flatten(0, 1)
        validate_y = torch.cat(validate_y, 0).transpose(1, 2).flatten(0, 1)
        test_x = torch.cat(test_x, 0).transpose(1, 2).flatten(0, 1)
        test_y = torch.cat(test_y, 0).transpose(1, 2).flatten(0, 1)

        import lightgbm as lgb
        #  one-step forecasting
        params = {
            'n_estimators': 2000,
            'max_depth': 4,
            'num_leaves': 2 ** 4,
            'learning_rate': 0.1,
            'boosting_type': 'dart',
            'verbose':0,
            # early_stopping_rounds = 10,
        }

        model = lgb.LGBMRegressor(first_metric_only=True, **params)
        window_size = 365
        prediction_horizon = 1

        from joblib import dump, load
        # lr是一个LogisticRegression模型
        if os.path.exists('./checkpoints/1_step.model'):
            model = load('./checkpoints/1_step.model')
        else:
            model.fit(train_x, train_y[:, 0:prediction_horizon].numpy(),
                      eval_metric='l1',
                      eval_set=[(validate_x, validate_y[:, 0:prediction_horizon].numpy())],
                      )
            dump(model, './checkpoints/1_step.model')

        forecast_1_step = model.predict(test_x)

        print('  LightGBM MAE: %.4f' % (np.mean(np.abs(forecast_1_step - test_y[:,0:prediction_horizon].squeeze(1).numpy())))) #   LightGBM MAE: 0.2100

        import matplotlib.pyplot as plt
        plot_x_size = 15
        plot_y_size = 2
        plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]
        plt.plot(forecast_1_step[3000:3200], label='Forecast')
        plt.plot(test_y[:,0:prediction_horizon].squeeze(1).numpy()[3000:3200],label='groud true')
        plt.legend()
        plt.show()
        plt.savefig('1step_predict.png')
        plt.cla()# clean

        # Tuning Window Size
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import RepeatedKFold
        # https://phdinds-aim.github.io/time_series_handbook/08_WinningestMethods/lightgbm_m5_forecasting.html
        params = {
            'n_estimators': 2000,
            'max_depth': 4,
            'num_leaves': 2 ** 4,
            'learning_rate': 0.1,
            'boosting_type': 'dart'
        }
        windows = [7, 30, 96,180, 365, 545, 730]
        results = []
        names = []
        from einops import rearrange
        # for w in windows:
        #     window_size = w
        #
        #     train_x, train_y = self.create_xy(train_data, 336, 96)
        #
        #     train_x = rearrange(train_x,' batch step dim -> (batch dim) step')
        #     train_y = train_y.flatten()
        #
        #     cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=123)
            # scores = cross_val_score(lgb.LGBMRegressor(**params), train_x, train_y, scoring='neg_mean_absolute_error',
            #                          cv=cv, n_jobs=-1)
            # results.append(scores)
            # names.append(w)
            # print('%3d --- MAE: %.3f (%.3f)' % (w, np.mean(scores), np.std(scores)))


        # 4 Multi-Step Prediction
        ### HYPERPARAMETERS ###
        window_size = 336
        prediction_horizon = 1

        ### TRAIN VAL SPLIT ###

        # params = {
        #     'n_estimators': 2000,
        #     'max_depth': 4,
        #     'num_leaves': 2 ** 4,
        #     'learning_rate': 0.1,
        #     'boosting_type': 'dart',
        #     # early_stopping_rounds = 10,
        #     'verbose' : 0,
        # }
        #
        # model = lgb.LGBMRegressor(first_metric_only=True, **params)
        #
        # model.fit(train_x, train_y,
        #           eval_metric='l1',
        #           eval_set=[(test_x, test_y)])

        recursive_x = test_x[:, :] #  18816
        forecast_ms = []
        # test_x, test_y = self.create_xy(test_data, 336, 96)
        # test_x = rearrange(test_x, ' batch step dim -> (batch dim) step')
        # test_y = rearrange(test_y, ' batch step dim -> (batch dim) step')
        for i in range(test_y.shape[1]):
            pred = model.predict(recursive_x)
            recursive_x = np.concatenate((recursive_x[:,1:], np.expand_dims(pred,1)),1)
            forecast_ms.append(pred)

        forecast_ms_rec = np.asarray(forecast_ms).transpose(1,0)

        print('Multi-Step MAE: %.4f' % (np.mean(np.abs(forecast_ms_rec - test_y.numpy())))) # 0.4208233903975961

        plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]

        # series[-test_size:].plot(label='True')
        # plt.plot(forecast_ms_rec, label='Forecast Multi-Step')
        plt.plot(forecast_ms_rec[0], label='Forecast Multi-Step')
        plt.plot(test_y[0],label='groud true')
        # plt.plot(forecast_1_step, label='Forecast One-Step')
        # plt.legend()
        plt.show()
        plt.savefig('2step_predict.png')
        plt.cla()

        from sklearn.multioutput import MultiOutputRegressor
        # train_artifact = wandb.Artifact('train-data', type='train-split')
        # valid_artifact = wandb.Artifact('valid-data', type='valid-split')
        model = MultiOutputRegressor(lgb.LGBMRegressor(), n_jobs=-1)
        model.fit(train_x, train_y)
        # wandb.log()
        forecast_ms_dir = model.predict(test_x)
        print('Multi-Step MAE: %.4f' % (np.mean(np.abs(forecast_ms_dir - test_y.numpy()))))

        plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]

        # series[-test_size:].plot(label='True')
        # plt.plot(forecast_ms_rec, label='Forecast Multi-Step')
        plt.plot(forecast_ms_dir[0], label='Forecast Multi-Step')
        plt.plot(test_y[0], label='groud true')
        # plt.plot(forecast_1_step, label='Forecast One-Step')
        # plt.legend()
        plt.show()
        plt.savefig('3step_predict.png')
        plt.cla()


    def objective(self,trial):
        learning_rate =trial.suggest_loguniform('learning_rate',1e-3,1e-1)
        num_leaves = trial.suggest_int('num_leaves',2,256)
        min_data_in_leaf= trial.suggest_int('min_data_in_leaf',1,100)
        bagging_fraction =trial.suggest_uniform('bagging_fraction',0.1,1.0)
        colsample_bytree =trial.suggest_uniform('colsample_bytree',0.1,1.0)
        lags = trial.suggest_int('lags',14,56,step=7)
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(random_state=0,n_estimators=500,bagging_freq=1,
                                learning_rate= learning_rate,num_leaves=num_leaves,
                                min_data_in_leaf=min_data_in_leaf,bagging_fraction=bagging_fraction,
                  colsample_bytree=colsample_bytree)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        train_x = []
        train_y = []
        validate_x = []
        validate_y = []
        test_x = []
        test_y = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            train_x.append(batch_x)
            train_y.append(batch_y[:, -self.args.pred_len:, :])

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            validate_x.append(batch_x)
            validate_y.append(batch_y[:, -self.args.pred_len:, :])

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            test_x.append(batch_x)
            test_y.append(batch_y[:, -self.args.pred_len:, :])

        model.fit(train_x, train_y,
                  eval_metric='l1',
                  eval_set=[(validate_x, validate_y)])
        forecast = model.predict(test_x)
        prediction_horizon = 1
        error= np.mean(np.abs(forecast - test_y[:,0:prediction_horizon].squeeze(1).numpy()))
        return error


    def findParameter(self):
        # https://optuna.org/#installation
        import optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective,n_trials=30)
        study.best_params  # E.g. {'x': 2.002108042}
        print(study.best_params)

    def timeseries_fft(self,input):
        input_f = torch.fft.rfft(input, dim=1)  #  返回傅里叶变换 正值
        input_f_abs = np.abs(input_f) # amplify 幅度
        # sorted_indices = torch.argsort(input_f_abs, descending=True)[:,:30] # 使用sort后的代码
        input_clean = []
        i = 0
        for a in input_f:
            input_clean.append(torch.cat(( a.real, a.imag),0))
            # train_x_clean.append(a[b])

            #train_x_clean.append(a[b])
        input_clean_fft = torch.stack(input_clean,0)
        return input_clean_fft

    def timeseries_reverse_fft(self,input,signal_sizes):
        # 将奇数部分作为实数，偶数部分作为虚数，生成复数张量
        real_part = input * (input % 2)
        imaginary_part = input * ((input + 1) % 2)
        complex_tensor = torch.complex(real_part.float(), imaginary_part.float())


        return complex_tensor


    def FFT_lightGBM(self,args):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        predict_steps= args.pred_len
        train_x = []
        train_y = []
        validate_x = []
        validate_y = []
        test_x = []
        test_y = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            train_x.append(batch_x)
            train_y.append(batch_y[:, -self.args.pred_len:, :])

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            train_x.append(batch_x)
            train_y.append(batch_y[:, -self.args.pred_len:, :])

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            test_x.append(batch_x)
            test_y.append(batch_y[:, -self.args.pred_len:, :])

        train_x = torch.cat(train_x, 0).transpose(1, 2).flatten(0, 1)
        train_y = torch.cat(train_y, 0).transpose(1, 2).flatten(0, 1)
        # validate_x = torch.cat(validate_x, 0).transpose(1, 2).flatten(0, 1)
        # validate_y = torch.cat(validate_y, 0).transpose(1, 2).flatten(0, 1)
        test_x = torch.cat(test_x, 0).transpose(1, 2).flatten(0, 1)
        test_y = torch.cat(test_y, 0).transpose(1, 2).flatten(0, 1)

        # series[-test_size:].plot(label='True')
        # plt.plot(forecast_ms_rec, label='Forecast Multi-Step')
        import matplotlib.pyplot as plt
        plot_x_size = 15
        plot_y_size = 8
        plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]
        #plt.plot(train_x[0], label='1')
        sampling_step = train_x.shape[1] # 采样率
        t = np.arange( sampling_step)
        f_real=train_x[0]
        plt.plot(t,f_real,color='k',label='real')
        plt.legend()

        plt.show()
        plt.savefig('111.png')
        plt.cla()
        fft_size = train_x.shape[1] #FFT长度
        # [B, T, C]
        from scipy.fft import rfft, rfftfreq
        n = len(t)
        yf =rfft(f_real.numpy())
        xf =rfftfreq(n,1)
        yf_abs = np.abs(yf) # amplify 幅度
        plt.plot(xf,yf_abs)
        plt.savefig('222.png')
        plt.cla()
#################---------------------------------------
        train_x_clean =self.timeseries_fft(train_x)


        #_, top_list = torch.topk(frequency_list,q k)
########################################################
        indices = yf_abs>5 # fill out those values under 300
        yf_denoise = indices * yf
        plt.plot(xf,np.abs(yf_denoise))
        plt.savefig('333.png')
        plt.cla()
        from scipy.fft import irfft
        new_clean = irfft(yf_denoise)
        plt.plot(t, f_real, color='k', label='real')
        plt.plot(t,new_clean,label='clean')
        plt.savefig('444.png')
        plt.cla()

        torch.mean(np.abs(self.timeseries_reverse_fft(self.timeseries_fft(train_y),predict_steps) - train_y))
        x_aix = torch.linspace(0 , 1 , 96)
        y_aix= train_y[0]
        plt.plot(x_aix, y_aix,label='origin')
        y_fft = torch.fft.rfft(y_aix)
        amplitudes = torch.abs(y_fft)
        num_components = 3  # 取前三个
        sorted_indices = torch.argsort(amplitudes, descending=True)[0:30]

        y_recon_axi = torch.fft.irfft(y_fft[30:],predict_steps)
        y_recon_axi_2 = torch.fft.irfft(y_fft[0:30], predict_steps)
        # plt.plot(x_aix, y_recon_axi,label='after30')
        # plt.plot(x_aix, y_recon_axi_2,label='before30')

        # 重新构建信号
        y_reconstructed = torch.zeros_like(y_fft)
        y_reconstructed[sorted_indices] = y_fft[sorted_indices]
        reconstructed_signal = torch.fft.irfft(y_reconstructed, n=96)
        print(torch.mean(np.abs(y_recon_axi_2 - reconstructed_signal)).numpy())
        print(torch.mean(np.abs(y_aix - reconstructed_signal)).numpy())
        plt.plot(x_aix, reconstructed_signal, label='recountMax')

        plt.legend()
        plt.show()
        plt.savefig('55.png')



        plt.cla()


        import lightgbm as lgb
        from sklearn.multioutput import MultiOutputRegressor
        # train_artifact = wandb.Artifact('train-data', type='train-split')
        # valid_artifact = wandb.Artifact('valid-data', type='valid-split')
        #lgb.LGBMRegressor().fit(train_x_clean,train_y[:,0].unsqueeze(1))
        train_y_clean_fft = self.timeseries_fft(train_y)
        model = MultiOutputRegressor(lgb.LGBMRegressor(), n_jobs=-1)

        model.fit(train_x_clean, train_y_clean_fft)
        # torch.mean(np.abs(self.timeseries_reverse_fft(self.timeseries_fft(train_y),predict_steps) - train_y))
        forecast_ms_dir = torch.tensor(model.predict(self.timeseries_fft(test_x)))
        print('Multi-Step MAE: %.4f' % (torch.mean(np.abs(self.timeseries_reverse_fft(forecast_ms_dir,predict_steps) - test_y))))





    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            torch.load(os.path.join('./checkpoints/' + setting + '/1.pth'))[
                'model.backbone.encoder.layers.1.self_attn.G_V.0.weight']
            # seaborn.distplot(self.G_V.state_dict()['0.weight'].data.view(-1).cpu().numpy())
            # seaborn.pairplot(pd.DataFrame(self.G_V.state_dict()['0.weight'].data.view(-1).cpu().numpy()))
            # import matplotlib.pyplot as plt
            # plt.show()
            # import pandas as pd
            # f = open("gate_weight_1.csv", 'a')
            # f.write(pd.DataFrame(torch.load(os.path.join('./checkpoints/' + setting + '/1.pth'))[
            #    'model.backbone.encoder.layers.1.self_attn.G_V.0.weight'].data.view(-1).cpu().numpy()).to_csv())
            # f.write(pd.DataFrame(g_s.view(-1).cpu().numpy()).to_csv())
            # f.close()

            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting + '/1.pth')))
            num_params = sum(param.numel() for param in self.model.parameters())
            print(num_params)  # 13496453
        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        k = 0
        epoch_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                k = k + 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' or 'Norm' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' or 'Norm' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if 'Rnn' in self.args.model or 'Mlinear' in self.args.model:
                    weight_u = outputs[1]
                    outputs = outputs[0]
                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        costtime = time.time() - epoch_time
        costperbatch = costtime * 1000 / k
        print("test-- cost time: {} , k {} ，per batch {}".format(costtime, k, costperbatch))
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' or 'Norm' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' or 'Norm' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
