#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Backtesting Sentiment Pairs
#%% [markdown]
# <b>Summary: </b>
# <br>
# For rolling average and rolling standard deviation of the lenght 7 days, NLP was calcualted for all possible combinations. A trading fee was assumed 0.0075 (taken from Bitmex). Calculations show that the best pairs are Bots/Whitepaper, Announcement/Bearish, Shilling/Team, and FOMO/Whales. Not surprisingly, increasing fees and window sizes changes to outcome of top performing pairs of topics.
# <br>
# Furthermore, NLP of the top performers was plotted on with different windows values (up to 30). It's interesting to see that for some pairs, 7 days for rolling mean and standard deviation is not an optimal window. For example, for whales/FOMO, a window of 24 yields a much higher PNL.
# <br>
# Finally a heat map for various moving standard deviation and average windows sizes shows that an optimal value is concentrated within a specific area.
# <br>
# In conclusion, due to a very high amount of combinations of windows sizes and sentiment tags, finding the "best" pair is hard. Every pair will have its highest NLP concentrated in different areas. Changing a size of one of the windows might change results drastically.

#%%
import sys
sys.path.insert(0, "../src")
import example_helper as eh
import analysis_helper as ah
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

# calculate PNL for a given strategy
# if sentiment positive go long, else go short
# fees are assumed to be 0.75% (taker fee from BITMEX)

def strategy(price_data, signal_a, signal_b, window_1 = 24 * 7, window_2 = 24*7,buy_sell_fee = 0.0075, pnl_0 = 1.0):    
    sent_score = ah.nb_calc_sentiment_score_b(signal_a,signal_b,window_1,window_2)
    pnl = ah.nb_backtest_a(price_data, sent_score, 1.0, buy_sell_fee)
    return pnl

# PNL of various moving window size for a given combination of topics

#%% [markdown]
# ### Given window size 7 in rolling average and standard deviation, calculate PNL for every possible pair of strategies.
# It will give 8649 NLP values calculated from 2017 until the beginning of 2019

#%%



#%%
# for each combination of signals, generate PNL for the last period in data
total = np.zeros(shape=(93,93))
print("calculating... might take a minute or two...")
for i in range(0,len(all_topics)):
    for j in range(0,len(all_topics)):
        sent_score = ah.nb_calc_sentiment_score_b(all_topics[i],all_topics[j],ra_win_size_short=24*7,ra_win_size_long=24*7)
        pnl = ah.nb_backtest_a(price_data, sent_score, 1.0, buy_sell_fee=0.0075)
        total[i][j] = pnl[-1]
    #print("Row " + str(i+1) + " out of 93...")
print("done")


#%%


#%% [markdown]
# ### Impossible to see all 8649 values
# Chose top 30

#%%
# get all PNL in a dataframe
data = pd.DataFrame(total).rename(columns=aug_topics,index=aug_topics)
# given all combinations of signals, show the combinations that yield the highest PNL
c = data.abs()
s = c.unstack()
so_st = s.sort_values(kind="quicksort")
# specify n, a number of top combinations to be shown
t = so_st.tail(n=30).index

# labels for graphs and tables
columns_t = dict((y, x) for x, y in t).keys()
rows_t = dict((x, y) for x, y in t).keys()

# pick from the dataframes only the pairs of strategies that are within the top list
top = data[rows_t].loc[columns_t]


#%%
so_st.tail(n=20)

#%% [markdown]
# # Heat Map for top 30 pairs

#%%
# a sorted dataframe to get highest PNLs in the first rows
idx = pd.unique([i[1] for i in np.flip(t.values)])
col = pd.unique([i[0] for i in np.flip(t.values)])
sorted_df = data[col].loc[idx]

#%% [markdown]
# # Testing for different windows sizes
#%% [markdown]
# ### Before the backtesting was for only one window size. It's also interesting to see how the strategy would work with different windows sizes
#%% [markdown]
# # Different rolling mean and std window sizes
#%% [markdown]
# From a chosen pair of topics, compute NLP for various rolling average and rolling std. It's interesting to see, whether the "optimal" values are concentrated withing a specific range.
#%% [markdown]
# ### Example for 'Bots' and 'Whitepaper'

#%%
def window_combination(price_data,top_a,top_b,end_day_x,end_day_y,start_day_x=0,start_day_y=0,buy_sell_fee=0.0075):
    total_comb = np.zeros(shape=(end_day_x,end_day_y))
    print("Calculating...")
    for i in range(start_day_x,end_day_x):
        for j in range(start_day_y,end_day_y):
            if i<j:
                total_comb[i][j] = strategy(price_data,top_a,top_b,window_1=24*(i+1),window_2=24*(j+1),buy_sell_fee = 0.0075)[-1]
            else:
                pass
    print("Done.")
    return total_comb[start_day_x:end_day_x,start_day_y:end_day_y]


#%%
#specify tags
ix = 0 # specify startpoint number of rolling mean
iy = 10 # specify startpoint of rolling std
end_x = 20 # specify endpoint number of rolling mean
end_y = 30 # specify endpoint of rolling std
topic_a = 'Bots'
topic_b = 'Whitepaper'
top_b = all_topics[aug_topics_inv[topic_b]]
top_a = all_topics[aug_topics_inv[topic_a]]
total_s = window_combination(price_data,top_a,top_b,end_x,end_y,start_day_x=ix,start_day_y=iy)


#%%
# plot
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.0, dark=1.2, as_cmap=True)
figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
ax = sns.heatmap(total_s, linewidth=0.00, cmap="RdYlGn",yticklabels=np.arange(ix+1,end_x+1),xticklabels=np.arange(iy+1,end_y+1))
plt.show()

#%% [markdown]
# #### example for 'Positive' and 'Bearish'

#%%
#specify tags
ix = 0
iy = 0
end_x = 60
end_y = 60
topic_a = 'Positive'
topic_b = 'Bearish'
top_b = all_topics[aug_topics_inv[topic_b]]
top_a = all_topics[aug_topics_inv[topic_a]]
total_s = window_combination(price_data,top_a,top_b,end_x,end_y,start_day_x=ix,start_day_y=iy)


#%%
# plot

figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='black')
ax = sns.heatmap(total_s, linewidth=0.00, cmap="RdYlGn",yticklabels=np.arange(ix+1,end_x+1),xticklabels=np.arange(iy+1,end_y+1))
ax.set_title('Bearish/Positive')
ax.set_ylabel('First Moving Average')
ax.set_xlabel('Second Moving Average')
plt.show()

#%% [markdown]
# ### Plotted 4 points

#%%
# Pick Topics
aug_signal_a = aug_data[:, aug_topics_inv["Positive"]].astype(np.float64)
aug_signal_b = aug_data[:, aug_topics_inv["Bearish"]].astype(np.float64)

# generate the sentiment score
sent_score = ah.nb_calc_sentiment_score_b(aug_signal_a, aug_signal_b, 7*24, 39*24)
sent_score1 = ah.nb_calc_sentiment_score_b(aug_signal_a, aug_signal_b, 18*24, 38*24)
sent_score2 = ah.nb_calc_sentiment_score_b(aug_signal_a, aug_signal_b, 18*24, 39*24)
sent_score3 = ah.nb_calc_sentiment_score_b(aug_signal_a, aug_signal_b, 7*24, 40*24)

# define some parameters for the backtest
start_pnl = 1.0
buy_sell_fee = 0.0075

# run the backtest
pnl = ah.nb_backtest_a(price_data, sent_score, start_pnl, buy_sell_fee)
pnl1 = ah.nb_backtest_a(price_data, sent_score1, start_pnl, buy_sell_fee)
pnl2 = ah.nb_backtest_a(price_data, sent_score2, start_pnl, buy_sell_fee)
pnl3 = ah.nb_backtest_a(price_data, sent_score3, start_pnl, buy_sell_fee)

# set up the figure
fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(20,10))

# initialise some labels for the plot
datenum_aug_data = [md.date2num(datetime.datetime.fromtimestamp(el)) for el in t_aug_data]
datenum_price_data = [md.date2num(datetime.datetime.fromtimestamp(el)) for el in t_price_data]

# plot stuff
ax[0].grid(linewidth=0.4)
ax[1].grid(linewidth=0.4)

ax[0].plot(datenum_price_data, price_data, linewidth=0.5)

ax[1].plot(datenum_aug_data, pnl, linewidth=0.5)
ax[1].plot(datenum_price_data, pnl1, linewidth=0.5)
ax[1].plot(datenum_price_data, pnl2, linewidth=0.5)
ax[1].plot(datenum_price_data, pnl3, linewidth=0.5)
ax[1].legend(("A","B","C","D"))


# label axes
ax[0].set_ylabel("Price")
ax[1].set_ylabel("PnL")
ax[0].set_title("Profit and Loss.py")

# generate the time axes
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax[0]=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d')
ax[0].xaxis.set_major_formatter(xfmt)



# show the plot
plt.show()


