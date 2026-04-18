#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np  # 新增：用来做对数处理
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 1. 数据准备
df = pd.read_csv("WorldEnergy.csv")

# 选取几个典型国家（咱们这次选得更有代表性一点）
asia = ['China', 'India', 'Japan', 'South Korea'] 
europe = ['Germany', 'France', 'United Kingdom', 'Italy']

df_lab3 = df[df['country'].isin(asia + europe)].copy()
df_lab3 = df_lab3[df_lab3['year'].isin([2010, 2020])]
df_lab3['region'] = df_lab3['country'].apply(lambda x: 'Asia' if x in asia else 'Europe')
df_lab3['year_cat'] = df_lab3['year'].astype(str)

# 【关键改进】：对发电量取对数！
# 这样能把“噪音”压下去，让“地区差异”显现出来
df_lab3['log_generation'] = np.log10(df_lab3['electricity_generation'] + 1)

# 2. 画两张图（记得在 Plots 窗口右侧切换看！）
plt.figure(figsize=(10, 5))
sns.pointplot(data=df_lab3, x='year_cat', y='log_generation', hue='region', capsize=.1)
plt.title('Log-Scaled Interaction Plot (Lab 3)')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=df_lab3, x='region', y='log_generation', hue='year_cat')
plt.title('Log-Scaled Distribution (Lab 3)')
plt.show()

# 3. 运行 ANOVA（这次用 log_generation 跑）
# 老师例题里如果数据波动大，通常也会建议做这种 Transformation
model = ols('log_generation ~ C(region) * C(year_cat)', data=df_lab3).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("\n" + "="*40)
print("   LAB 3 FINAL ANOVA (LOG-TRANSFORMED)")
print("="*40)
print(anova_table)


# In[ ]:




