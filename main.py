import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import distance
import mplcursors
from mpldatacursor import datacursor

# ---------------------preparing the dataset--------------------------
links = ['recovered', 'deaths', 'confirmed']
l = list()
for item in links:
    # reding dataset from online csv files
    df = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_" + item + "_global.csv")
    # droping province and longitude and latitude columns because they won't be needed
    df.drop(df.columns[[0, 2, 3]], axis=1, inplace=True)
    # merging cities data by country
    df = df.groupby('Country/Region').sum()
    df.index.rename('Country', inplace=True)
    # exporting refined dataset to csv file
    df.to_csv(item+'.csv')
    l.append(df)
# creating a summary dataset file for the most recent data from different datasets
data = {'Country':  l[0].index.tolist(), 'Total Cases': l[2].iloc[:, -1].tolist(
), 'Total Deaths': l[1].iloc[:, -1].tolist(), 'Total Recovered': l[0].iloc[:, -1].tolist(), 'Active Cases': (l[2].iloc[:, -1]-l[1].iloc[:, -1]-l[0].iloc[:, -1]).tolist()}
df = pd.DataFrame(data)
df.set_index('Country', inplace=True)
l.append(df)
df.to_csv('All.csv')
# creating a transposed dataset to be used in rapidminer
l[2].T.to_csv('Transposed.csv')
# ---------------Prediction and Plots---------------------------------

# defining exponential function


def func(x, a, b, c):
    return a * np.exp(b * x) + c


# collecting data to be used fro prediction
xdata = range(l[2].loc['Egypt', '3/4/20':'4/18/20'].count())
ydata = l[2].loc['Egypt', '3/4/20':'4/18/20'].tolist()
# making prediction equation by fitting data to the exponential function
popt, pcov = curve_fit(func, xdata, ydata)
xdata = range(l[2].loc['Egypt', '3/4/20':].count()+10)
# ---------------------------------------------
plt.figure('Egypt Progress and Prediction')
plt.plot(xdata, func(xdata, *popt), label='Prediction')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.plot(l[2].loc['Egypt', '3/4/20':], label='Real Data')
plt.xticks(rotation='vertical')
plt.legend(loc="upper left")
plt.get_current_fig_manager().resize(1600, 900)
plt.tight_layout()
# ---------------------------------------
plt.figure('China, Iran,  Italy, Spain & USA Progress')
plt.plot(l[2].loc['China'], label='China')
plt.plot(l[2].loc['Italy'], label='Italy')
plt.plot(l[2].loc['Iran'], label='Iran')
plt.plot(l[2].loc['Spain'], label='Spain')
plt.plot(l[2].loc['US'], label='USA')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.xticks(rotation='vertical')
plt.legend(loc="upper left")
plt.get_current_fig_manager().resize(1600, 900)
plt.tight_layout()
# ----------------------------------------
plt.figure('Boxplots')
plt.boxplot(l[3].iloc[:, 0], showfliers=False,
            positions=[0], labels=['Total Cases'])
plt.boxplot(l[3].iloc[:, 1], showfliers=False,
            positions=[1], labels=['Total Deaths'])
plt.boxplot(l[3].iloc[:, 2], showfliers=False,
            positions=[2], labels=['Total Recovered'])
plt.boxplot(l[3].iloc[:, 3], showfliers=False,
            positions=[3], labels=['Active Cases'])
plt.legend(loc="upper left")
plt.get_current_fig_manager().resize(1600, 900)
plt.tight_layout()
# ----------------------------------------
plt.figure('Skewness')
sk = l[3].groupby(by='Total Deaths').count().iloc[:, 0]
sk.index = sk.index.map(str)
plt.xticks(rotation='vertical')
plt.plot(sk, label='Total Death')
plt.get_current_fig_manager().resize(1600, 900)
plt.tight_layout()
# ------------------------------------------
datacursor()
plt.show()
# ------------------------Correlation---------------------
print("-----------------pearson : standard correlation coefficient---------------")
print(l[3].corr(method='pearson'))
print("-----------------kendall : Kendall Tau correlation coefficient------------")
print(l[3].corr(method='kendall'))
print("-----------------spearman : Spearman rank correlation---------------------")
print(l[3].corr(method='spearman'))
# --------------------Skewness------------------------
print("---------------- Skewness of Data ------------------")
print(l[3].skew(axis=0))
# ---------------------Similarities---------------
dis = ['China', 'Egypt', 'Iran', 'Italy']
euc = distance.pdist(l[3].loc[dis], 'minkowski', 2)
man = distance.pdist(l[3].loc[dis], 'minkowski', 1)
i = 0
k = 0
eu = []
ma = []
su = []
# calculating supermum similarities manually
while i < len(dis):
    j = i + 1
    while j < len(dis):
        eu.append([dis[i]+' & ' + dis[j], euc[k]])
        ma.append([dis[i]+' & ' + dis[j], man[k]])
        sup = distance.minkowski(
            l[3].loc[dis[i]], l[3].loc[dis[j]], p=float('inf'))
        su.append([dis[i]+' & ' + dis[j], sup])
        k = k + 1
        j = j + 1
    i = i + 1
print("-------------------- Eucledian Distance -------")
print(pd.DataFrame(eu))
print("-------------------- Manhatan Distance --------")
print(pd.DataFrame(ma))
print("-------------------- Supermum Distance --------")
print(pd.DataFrame(su))
print('-----------------------------------------')
# more options can be specified also
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(l[3].iloc[:, [0, 2]])
