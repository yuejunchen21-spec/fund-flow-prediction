# -*- coding: utf-8 -*-
# 第六步：最终提交 = 申购用"星期几基线" + 赎回用"带滞后的随机森林"

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ---------- 1) 读取 + 按天加总 ----------
path = r'D:\分析数据\项目\资金流入流出预测-挑战Baseline\Purchase Redemption Data\user_balance_table.csv'
df = pd.read_csv(path, usecols=['report_date', 'total_purchase_amt', 'total_redeem_amt'])
daily = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
daily.columns = ['report_date', 'purchase', 'redeem']
daily['report_date'] = pd.to_datetime(daily['report_date'], format='%Y%m%d')
daily = daily.sort_values('report_date').reset_index(drop=True)

# ---------- 2) 构造特征 ----------
daily['weekday'] = daily['report_date'].dt.weekday
daily['day'] = daily['report_date'].dt.day
daily['month'] = daily['report_date'].dt.month
daily['dayofyear'] = daily['report_date'].dt.dayofyear
daily['is_weekend'] = (daily['weekday'] >= 5).astype(int)

for col in ['purchase', 'redeem']:
    daily[col + '_lag1'] = daily[col].shift(1)
    daily[col + '_lag7'] = daily[col].shift(7)
    daily[col + '_mean7'] = daily[col].shift(1).rolling(7, min_periods=1).mean()

FEATURES = ['weekday', 'day', 'month', 'dayofyear', 'is_weekend',
            'purchase_lag1', 'purchase_lag7', 'purchase_mean7',
            'redeem_lag1', 'redeem_lag7', 'redeem_mean7']

sep_dates = pd.date_range('2014-09-01', '2014-09-30')

# ---------- 3) 申购：星期几基线（最近4周同星期几均值） ----------
purchase_base = daily.tail(28).groupby('weekday')['purchase'].mean()
purchase_pred = [int(round(purchase_base.loc[d.weekday()])) for d in sep_dates]

# ---------- 4) 赎回：带滞后的随机森林 ----------
full = daily.dropna(subset=FEATURES)
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(full[FEATURES], full['redeem'])

hist_p = daily['purchase'].tolist()
hist_r = daily['redeem'].tolist()
redeem_pred = []

for i, d in enumerate(sep_dates):
    feat = {
        'weekday': d.weekday(),
        'day': d.day,
        'month': d.month,
        'dayofyear': d.dayofyear,
        'is_weekend': 1 if d.weekday() >= 5 else 0,
        'purchase_lag1': hist_p[-1],
        'purchase_lag7': hist_p[-7],
        'purchase_mean7': float(np.mean(hist_p[-7:])),
        'redeem_lag1': hist_r[-1],
        'redeem_lag7': hist_r[-7],
        'redeem_mean7': float(np.mean(hist_r[-7:])),
    }
    r = int(round(model.predict(pd.DataFrame([feat])[FEATURES])[0]))
    redeem_pred.append(r)
    hist_p.append(purchase_pred[i])  # 把基线的申购预测追加，供下一天当"昨天"
    hist_r.append(r)                  # 把赎回预测追加

# ---------- 5) 拼成提交文件 ----------
sub = pd.DataFrame({'report_date': sep_dates.strftime('%Y%m%d'), 'purchase': purchase_pred, 'redeem': redeem_pred})
out = r'D:\分析数据\项目\资金流入流出预测-挑战Baseline\tc_comp_predict_table.csv'
sub.to_csv(out, index=False, header=False)
print('组合提交文件已保存:', out)
print(sub.head(10))