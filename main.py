import seaborn
import pandas as pd

import scipy.stats as stats # 统计学库
from scipy.stats import laplace  # 用于拟合正态分布曲线
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data =pd.read_csv('gate_weight.csv',header=None,names=['index','value'])
    #seaborn.displot(data['0'],kind="kde") # seaborn.displot(data['0'])
    #seaborn.displot(data['0'],kind="kde") # seaborn.displot(data['0'])
    seaborn.displot(data['value'])
    value=data['value']

    # bins=[-0.01,0.01]
    # segments=pd.cut(value,bins,right=False)
    # counts=pd.value_counts(segments,sort=False) # 28213248    12295(-0.0001,0.0001) 0.0436%  26077824(-0.5,0.5) 92.43%  11125418(-0.1,0.1) 39.43%
    #b =plt.bar(counts.index.astype(str),counts) #  1242435 (-0.01,0.01) 4.4%   125741 (-0.001,0.001) 0.446%
    #plt.bar_label(b,counts)
    plt.savefig('gate_weight_22.png')
    plt.show()
    #parameters = laplace.fit(value)

    #print(parameters) # (-0.0015659871, 0.19544081979911498)
    print('end')
