# GRAformer:A gated residual attention transformer for multivariate time series forecasting

### This is an offical implementation of GRAformer: [A gated residual attention transformer for multivariate time series forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0925231224002376).


## Key Designs

:star1: **Gated Residual Attention**: segmentation of time series into subseries-level patches which are served as input tokens to Transformer.

:star2: **Channel Adaptation Infusion**: A training approach where a subset of the model parameters are trained in a channel-dependent manner (e.g., based on specific tokens or relevance information), while the remaining parameters are trained independently in a channel-agnostic way.



## Results

### Supervised Learning

GRAformer outshines the best results achieved by Transformer-based models, delivering an impressive overall 1.0% reduction in Mean Squared Error (MSE) and a 1% decrease in Mean Absolute Error (MAE). Remarkably, it also boasts twice the training speed compared to the fastest model PatchTST. Furthermore, GRAformer surpasses the performance of non-Transformer-based models like DLinear, MTGNN, AGLGNN, and DeepAR, solidifying its position as a standout in the field.

[//]: # (![alt text]&#40;https://github.com/yuqinie98/PatchTST/blob/main/pic/table3.png&#41;)




## Getting Started


1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/```.  For example, if you want to get the multivariate forecasting results for weather dataset, just run the following command, and you can open ```./result.txt``` to see the results once the training is done:


You can adjust the hyperparameters based on your needs (e.g. different patch length, different look-back windows and prediction lengths.). I also provide codes for the baseline models.


## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/ts-kim/RevIN

https://github.com/timeseriesAI/tsai

https://github.com/timeseriesAI/tsai

https://github.com/yuqinie98/PatchTST/
## Contact

If you have any questions or concerns, please contact me:  yangchengcao@whu.edu.cn or submit an issue

## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```
@article{yang2024graformer,
  title={GRAformer: A gated residual attention transformer for multivariate time series forecasting},
  author={Yang, Chengcao and Wang, Yutian and Yang, Bing and Chen, Jun},
  journal={Neurocomputing},
  volume={581},
  pages={127466},
  year={2024},
  publisher={Elsevier}
}
```

