# -*- coding: utf-8 -*-
# 第四步：用随机森林预测，并"回测"验证它比不比"星期几基线"强

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ---------- 1) 读取（只读需要的3列，省内存）+ 按天加总 ----------
path = r'D:\分析数据\项目\资金流入流出预测-挑战Baseline\Purchase Redemption Data\user_balance_table.csv'
df = pd.read_csv(path, usecols=['report_date', 'total_purchase_amt', 'total_redeem_amt'])
daily = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
daily.columns = ['report_date', 'purchase', 'redeem']
daily['report_date'] = pd.to_datetime(daily['report_date'], format='%Y%m%d')
daily = daily.sort_values('report_date').reset_index(drop=True)

# ---------- 2) 构造特征（特征工程） ----------
daily['weekday'] = daily['report_date'].dt.weekday
daily['day'] = daily['report_date'].dt.day
daily['month'] = daily['report_date'].dt.month
daily['dayofyear'] = daily['report_date'].dt.dayofyear
daily['is_weekend'] = (daily['weekday'] >= 5).astype(int)
daily['day_index'] = np.arange(len(daily))

FEATURES = ['weekday', 'day', 'month', 'dayofyear', 'is_weekend', 'day_index']

# ---------- 3) 回测：用8月之前训练，预测8月整月，对比"星期几基线" ----------
cut = pd.Timestamp('2014-08-01')
tr = daily['report_date'] < cut
te = (daily['report_date'] >= cut) & (daily['report_date'] <= pd.Timestamp('2014-08-31'))

def mape(real, pred):
    return np.mean(np.abs((real - pred) / real)) * 100

last4w = daily[daily['report_date'] < cut].tail(28)
base = last4w.groupby('weekday')[['purchase', 'redeem']].mean()

for target in ['purchase', 'redeem']:
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(daily.loc[tr, FEATURES], daily.loc[tr, target])
    rf_pred = model.predict(daily.loc[te, FEATURES])
    base_pred = daily.loc[te, 'weekday'].map(base[target]).values
    print(target, '| 随机森林 MAPE =', round(mape(daily.loc[te, target].values, rf_pred), 2),
          '% | 星期几基线 MAPE =', round(mape(daily.loc[te, target].values, base_pred), 2), '%')

# ---------- 4) 用全部数据训练，预测9月，生成提交文件 ----------
sep = pd.DataFrame({'report_date': pd.date_range('2014-09-01', '2014-09-30')})
sep['weekday'] = sep['report_date'].dt.weekday
sep['day'] = sep['report_date'].dt.day
sep['month'] = sep['report_date'].dt.month
sep['dayofyear'] = sep['report_date'].dt.dayofyear
sep['is_weekend'] = (sep['weekday'] >= 5).astype(int)
sep['day_index'] = np.arange(len(daily), len(daily) + 30)

sub = sep[['report_date']].copy()
for target in ['purchase', 'redeem']:
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(daily[FEATURES], daily[target])
    sub[target] = model.predict(sep[FEATURES]).round().astype(int)

sub['report_date'] = sub['report_date'].dt.strftime('%Y%m%d')
out = r'D:\分析数据\项目\资金流入流出预测-挑战Baseline\tc_comp_predict_table.csv'
sub[['report_date', 'purchase', 'redeem']].to_csv(out, index=False, header=False)
print('随机森林提交文件已保存到:', out)
print(sub.head())