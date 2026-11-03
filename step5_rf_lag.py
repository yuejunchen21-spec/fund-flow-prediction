# -*- coding: utf-8 -*-
# 第五步：随机森林 + 滞后特征（昨天值、上周同期、最近7天均值）

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

# ---------- 2) 构造特征：日历 + 滞后 ----------
daily['weekday'] = daily['report_date'].dt.weekday
daily['day'] = daily['report_date'].dt.day
daily['month'] = daily['report_date'].dt.month
daily['dayofyear'] = daily['report_date'].dt.dayofyear
daily['is_weekend'] = (daily['weekday'] >= 5).astype(int)

for col in ['purchase', 'redeem']:
    daily[col + '_lag1'] = daily[col].shift(1)                 # 昨天值
    daily[col + '_lag7'] = daily[col].shift(7)                 # 上周同期值
    daily[col + '_mean7'] = daily[col].shift(1).rolling(7, min_periods=1).mean()  # 最近7天均值

FEATURES = ['weekday', 'day', 'month', 'dayofyear', 'is_weekend',
            'purchase_lag1', 'purchase_lag7', 'purchase_mean7',
            'redeem_lag1', 'redeem_lag7', 'redeem_mean7']

# ---------- 3) 回测：8月之前训练，预测8月，对比星期几基线 ----------
cut = pd.Timestamp('2014-08-01')
tr = daily['report_date'] < cut
te = (daily['report_date'] >= cut) & (daily['report_date'] <= pd.Timestamp('2014-08-31'))
train = daily[tr].dropna(subset=FEATURES)
test = daily[te]

def mape(real, pred):
    return np.mean(np.abs((real - pred) / real)) * 100

last4w = daily[daily['report_date'] < cut].tail(28)
base = last4w.groupby('weekday')[['purchase', 'redeem']].mean()

for target in ['purchase', 'redeem']:
    m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    m.fit(train[FEATURES], train[target])
    rf_pred = m.predict(test[FEATURES])
    base_pred = test['weekday'].map(base[target]).values
    print(target, '| 随机森林(带滞后) MAPE =', round(mape(test[target].values, rf_pred), 2),
          '% | 星期几基线 MAPE =', round(mape(test[target].values, base_pred), 2), '%')

# ---------- 4) 全量训练，递归预测9月（一天天来，把预测值喂给下一天） ----------
full = daily.dropna(subset=FEATURES)
models = {}
for target in ['purchase', 'redeem']:
    m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    m.fit(full[FEATURES], full[target])
    models[target] = m

hist_p = daily['purchase'].tolist()
hist_r = daily['redeem'].tolist()

rows = []
for d in pd.date_range('2014-09-01', '2014-09-30'):
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
    x = pd.DataFrame([feat])[FEATURES]
    p = int(round(models['purchase'].predict(x)[0]))
    r = int(round(models['redeem'].predict(x)[0]))
    rows.append((d.strftime('%Y%m%d'), p, r))
    hist_p.append(p)   # 把预测值追加回去，供下一天当"昨天"用
    hist_r.append(r)

sub = pd.DataFrame(rows, columns=['report_date', 'purchase', 'redeem'])
out = r'D:\分析数据\项目\资金流入流出预测-挑战Baseline\tc_comp_predict_table.csv'
sub.to_csv(out, index=False, header=False)
print('带滞后特征的随机森林提交文件已保存:', out)
print(sub.head())