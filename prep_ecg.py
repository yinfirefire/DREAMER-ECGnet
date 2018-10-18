
# coding: utf-8

# In[1]:


import numpy as np
import mat4py as m4p
from scipy import signal 
from biosppy.signals import ecg
from matplotlib import pyplot as plt
import matplotlib
import pywt
import warnings
import os
warnings.filterwarnings("ignore")

def score_dict(data):
    scores={}
    for i in range(len(data)):
        scores[i]=np.stack((np.array(data[i]['ScoreValence']).flatten(),np.array(data[i]['ScoreArousal']).flatten(),np.array(data[i]['ScoreDominance']).flatten()))
    return scores

def baseline_ds(data):
    ecg_base={}
    for i in range(len(data)):
        ecg_base[i]=np.array(data[i]['ECG']['baseline']).reshape((18,-1,2))
    return ecg_base

def stimuli_ds(data):
    ecg_stimuli={}
    for i in range(len(data)):
        individual_stimuli={}
        for j in range(18):
            individual_stimuli[j]=np.array(data[i]['ECG']['stimuli'][j]).reshape((-1,2))
        ecg_stimuli[i]=individual_stimuli
    return ecg_stimuli


# In[2]:


def va_model(score,person):
    va_sc=[]
    sc=score[person][0:2].T
    for i in sc:
        if 3 in i:
            va_sc.append('Neutral')
        elif i[0]>3:
            if i[1]>3:
                va_sc.append('Happiness')
            else:
                va_sc.append('Peacefulness')
        elif i[0]<3:
            if i[1]>3:
                va_sc.append('Scary')
            else:
                va_sc.append('Sad')
    return va_sc


# In[33]:


def select_n_slice(person,clip,dataset,seconds,transformation,filtered,score):
    #person(0-22) clip(0-17) dataset(baseline/stimuli) seconds() transformation(cwt/stft)
    affect=va_model(score,person)[clip]
    dirname='Images/'+affect
    try:
        os.mkdir(dirname)
    except FileExistsError:
        print(dirname,'already exist')
        
    if filtered:
        select = ecg.ecg(signal=dataset[person][clip][:,0],sampling_rate=256,show=False)['filtered']
    else:
        select=dataset[person][clip][:,0]
           
    length = select.shape[0]
    pts=seconds*256
    slices={}
    freq={}
    time={}
    i=0
    widths=np.arange(1,30)
    n=length//pts
    cwtmatr={}
    if transformation=='stft':
        while i<n:
            freq[i],time[i],slices[i]=signal.spectrogram(select[i*pts:(i+1)*pts],fs=256,nperseg=32,noverlap=16)
            i+=1
        for i in freq.keys():
            f_name=(str(dirname)+'/{0}.png').format(str(i)+'_p'+str(person)+'_c'+str(clip))
            plt.figure(figsize=(5.4,2.5),dpi=227)
            ax1=plt.pcolormesh(time[i],freq[i],np.log(slices[i]),cmap='jet')
            ax1.axes.set_xticks([])
            ax1.axes.set_ylim(0,125)
            ax1.axes.set_yticks([])
            plt.savefig(f_name,pad_inches=-0.1,bbox_inches='tight',transparent=True)
            plt.close()
            
    elif transformation=='cwt':
        while i<n:
            cwtmatr[i],_ = pywt.cwt(select[i*pts:(i+1)*pts], widths, "mexh")
            i+=1
        for i in cwtmatr.keys():
            f_name=(str(dirname)+'/{0}.png').format(str(i)+'_p'+str(person)+'_c'+str(clip))
            plt.figure(figsize=(5.4,2.5),dpi=227)
            ax1=plt.imshow(cwtmatr[i],aspect='auto',vmax=abs(cwtmatr[i]).max(), vmin=-abs(cwtmatr[i]).max())
            ax1.axes.set_xticks([])
            ax1.axes.set_yticks([])
            plt.savefig(f_name,pad_inches=-0.1,bbox_inches='tight',transparent=True)
            plt.close()
            


