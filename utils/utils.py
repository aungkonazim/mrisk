import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def return_day(t):
    t = pd.to_datetime(t)
    timestring = t.strftime('%Y.%m.%d')
    return timestring

def f1Bias_scorer_CV(probs, y, ret_bias=True):
    precision, recall, thresholds = metrics.precision_recall_curve(y, probs)

    f1,bias = 0.0,.5
    min_recall = .5
    for i in range(0, len(thresholds)):
        if not (precision[i] == 0 and recall[i] == 0):
            f = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            if f > f1:
                f1 = f
                bias = thresholds[i]

    if ret_bias:
        return f1, bias
    else:
        return f1


def maintain_order(x,increase=True):
    for i,a in enumerate(x):
        if i==0:
            continue
        if increase:
            if x[i]<x[i-1]:
                x[i] = x[i-1]
        else:
            if x[i]>x[i-1]:
                x[i] = x[i-1]
    return x

def smooth_result(result1,window=5,polynomial=4):
    result = result1.copy()
    min_ = result[:,1].min()
    max_ = result[:,1].max()
    result[:,1] = savgol_filter(result[:,1],window , polynomial)
    # result[:,1] = maintain_order(result[:,1],True)
    result[result[:,1]>max_,1] = max_
    result[result[:,1]<min_,1] = min_    
    result[:,2] = savgol_filter(result[:,2],window , polynomial)
    result[:,2] = maintain_order(result[:,2],False)
    return result

def get_smoothed_result(final_intpday,final_recall,final_gap):
    result = np.array(list(zip(final_intpday,final_recall,final_gap)))
    result = result[result[:,0].argsort()]
    result1 = result[result[:,0]<=11]
    result_final = smooth_result(result1)
    return result_final

def get_interpolated_data(result_final,x,name='Overall',iteration=1):
    result_final = result_final[result_final[:,0].argsort()]
    # print(result_final.shape)
    # print(result_final[:,1].max()>1)
    f = interp1d(result_final[:,0],result_final[:,1],fill_value=(result_final[0,1],result_final[-1,1]),bounds_error=False)
    y_recall = f(x)
    f = interp1d(result_final[:,0],result_final[:,2],fill_value=(result_final[0,2],result_final[-1,2]),bounds_error=False)
    y_gap = f(x)
    return [list(a)+[name,iteration] for a in np.array(list(zip(x,y_recall,y_gap)))]

def get_df(df):
    if df.shape[0]==0:
        return df
    index = df.index.values
    index = np.random.choice(index,1)[0]
    df['pred'].at[index] = 1
    return df