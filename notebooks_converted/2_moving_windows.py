#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass

import helpers.example_helper as eh
import helpers.analysis_helper as ah
import msgpack
import zlib
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns; sns.set()

#%% [markdown]
# # Get Data

#%%
# define the location of the input file
filename_augmento_topics = "../data/example_data/augmento_topics.msgpack.zlib"
filename_augmento_data = "../data/example_data/augmento_data.msgpack.zlib"
filename_bitmex_data = "../data/example_data/bitmex_data.msgpack.zlib"

# load the example data
all_data = eh.load_example_data(filename_augmento_topics,
                             filename_augmento_data,
                             filename_bitmex_data)
aug_topics, aug_topics_inv, t_aug_data, aug_data, t_price_data, price_data = all_data
all_topics = aug_data.T.astype(float)

#%% [markdown]
# # Example for Topics "Bullish" and "Bearish"
# Selects count data for particular topics

#%%
aug_signal_bullish = aug_data[:, aug_topics_inv["Bullish"]].astype(np.float64)
aug_signal_bearish = aug_data[:, aug_topics_inv["Bearish"]].astype(np.float64)


#%%
# define the window size for the sentiment score calculation
n_days = 7
window_size = 24 * n_days # in hours

# generate the sentiment score
sent_score = ah.nb_calc_sentiment_score_a(aug_signal_bullish, aug_signal_bearish, window_size, window_size)

# define some parameters for the backtest
start_pnl = 1.0
buy_sell_fee = 0.0075

# run the backtest
pnl = ah.nb_backtest_a(price_data, sent_score, start_pnl, buy_sell_fee)

#%% [markdown]
# # Compare various windows sizes

#%%
sent_score = ah.nb_calc_sentiment_score_a(aug_signal_bullish,aug_signal_bearish,1,2)
pnl = ah.nb_backtest_a(price_data, sent_score, 1.0, 0.0075)


#%%
sent_score = ah.nb_calc_sentiment_score_a(aug_signal_bullish,aug_signal_bearish,7*24,7*24)
pnl = ah.nb_backtest_a(price_data, sent_score, 1.0, 0.0075)


#%%
# different windows sizes for sentiment score b
#h = 24
s_days = 20 # short
l_days = 20 # long

win_all_a = np.zeros(shape=(s_days,l_days))
win_all_b = np.zeros(shape=(s_days,l_days))

# matrix of size (s_days,l_days)

for i in range(0, s_days):
    for j in range(0, l_days):
        sent_score_a = ah.nb_calc_sentiment_score_a(aug_signal_bullish,aug_signal_bearish,(i+1)*24,(j+1)*24)
        sent_score_b = ah.nb_calc_sentiment_score_b(aug_signal_bullish, aug_signal_bearish, (i+1)*24,(j+1)*24)
        #pnl_a = ah.nb_backtest_a(price_data, sent_score_a, 1.0, 0.0075)
        #pnl_b = ah.nb_backtest_a(price_data, sent_score_b, 1.0, 0.0075)
        win_all_a[i,j] = ah.nb_backtest_a(price_data, sent_score_a, 1.0, 0.0075)[-1]
        win_all_b[i,j] = ah.nb_backtest_a(price_data, sent_score_b, 1.0, 0.0075)[-1]


#%%
##plot
#cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.0, dark=1.2, as_cmap=True)
#figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
#ax = sns.heatmap(win_all_a, linewidth=0.01, cmap=cmap)
#plt.show()


#%%
##plot
#cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.0, dark=1.2, as_cmap=True)
#figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
#ax = sns.heatmap(win_all_b, linewidth=0.01, cmap=cmap)
#plt.show()


#%%
# different windows sizes for sentiment score b
#h = 24
s_days = 20 # short
l_days = 20 # long

win_all_a = np.zeros(shape=(s_days,l_days))

# matrix of size (s_days,l_days)

for i in range(0, s_days):
    for j in range(0, l_days):
        sent_score_a = ah.nb_calc_sentiment_score_a(aug_signal_bullish,aug_signal_bearish,(i+1)*24,(j+1)*24)
        #pnl_a = ah.nb_backtest_a(price_data, sent_score_a, 1.0, 0.0075)
        #pnl_b = ah.nb_backtest_a(price_data, sent_score_b, 1.0, 0.0075)
        win_all_a[i,j] = ah.nb_backtest_a(price_data, sent_score_a, 1.0, 0.0075)[-1]


#%%
# different windows sizes for sentiment score b
#h = 24
s_days = 20 # short
l_days = 20 # long

f, axes = plt.subplots(1, 5, figsize=(40,10))
win_all_a = np.zeros(shape=(s_days,l_days))
# matrix of size (s_days,l_days)
for std in range(0,5):
    for i in range(0, s_days):
        for j in range(0, l_days):
            sent_score_a = ah.nb_calc_sentiment_score_a(aug_signal_bullish,aug_signal_bearish,(i+1)*24+np.random.normal(0,std),(j+1)*24+np.random.normal(0,std))
            win_all_a[i,j] = ah.nb_backtest_a(price_data, sent_score_a, 1.0, 0.0075)[-1]
    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.0, dark=1.2, as_cmap=True)
    figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
    ax = sns.heatmap(win_all_a, linewidth=0.01, cmap=cmap,ax=axes[std])
plt.show()


#%%
# different windows sizes for sentiment score b
#h = 24
s_days = 20 # short
l_days = 20 # long

f, axes = plt.subplots(1, 5, figsize=(40,10))
win_all_a = np.zeros(shape=(s_days,l_days))
# matrix of size (s_days,l_days)
for std in range(0,5):
    for i in range(0, s_days):
        for j in range(0, l_days):
            sent_score_a = ah.nb_calc_sentiment_score_a(aug_signal_bullish,aug_signal_bearish,(i+1)*24+np.random.uniform(0,std),(j+1)*24+np.random.uniform(0,std))
            win_all_a[i,j] = ah.nb_backtest_a(price_data, sent_score_a, 1.0, 0.0075)[-1]
    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.0, dark=1.2, as_cmap=True)
    figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
    ax = sns.heatmap(win_all_a, linewidth=0.01, cmap=cmap,ax=axes[std])
plt.show()


#%%
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.0, dark=1.2, as_cmap=True)
figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
ax = sns.heatmap(win_all_a, linewidth=0.01, cmap=cmap)
plt.show()


#%%
# different windows sizes for sentiment score b
#h = 24
s_days = 20 # short
l_days = 20 # long

win_all_b = np.zeros(shape=(s_days,l_days))

# matrix of size (s_days,l_days)

for i in range(0, s_days):
    for j in range(0, l_days):
        sent_score_a = ah.nb_calc_sentiment_score_a(aug_signal_bullish,aug_signal_bearish,(i+1)*24+np.random.normal(0,1),(j+1)*24+np.random.normal(0,1))
        win_all_b[i,j] = ah.nb_backtest_a(price_data, sent_score_a, 1.0, 0.0075)[-1]


#%%
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.0, dark=1.2, as_cmap=True)
figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
ax = sns.heatmap(win_all_b, linewidth=0.01, cmap=cmap)
plt.show()


#%%



