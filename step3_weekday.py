# -*- coding: utf-8 -*-
# 资金流入流出预测：第三步，用"星期几"的周期性改进基线

import pandas as pd

# 1) 读取 + 按天加总（同前两步）
path = r'D:\分析数据\项目\资金流入流出预测-挑战Baseline\Purchase Redemption Data\user_balance_table.csv'
df = pd.read_csv(path)
daily = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
daily.columns = ['report_date', 'purchase', 'redeem']
daily['report_date'] = pd.to_datetime(daily['report_date'], format='%Y%m%d')
daily = daily.sort_values('report_date')

# 2) 加一列"星期几"（0=周一 ... 6=周日）
daily['weekday'] = daily['report_date'].dt.weekday

# 3) 取最近4周（28天），按星期几求平均申购/赎回
last4w = daily.tail(28)
baseline = last4w.groupby('weekday')[['purchase', 'redeem']].mean()
print('各星期几的基准值：')
print(baseline.round().astype(int))

# 4) 生成9月的30天，按每个日期对应的星期几查表预测
sep = pd.DataFrame({'report_date': pd.date_range('2014-09-01', '2014-09-30')})
sep['weekday'] = sep['report_date'].dt.weekday
sep['purchase'] = sep['weekday'].map(baseline['purchase']).round().astype(int)
sep['redeem'] = sep['weekday'].map(baseline['redeem']).round().astype(int)
sep['report_date'] = sep['report_date'].dt.strftime('%Y%m%d')

# 5) 保存成提交文件
out = r'D:\分析数据\项目\资金流入流出预测-挑战Baseline\tc_comp_predict_table.csv'
sep[['report_date', 'purchase', 'redeem']].to_csv(out, index=False, header=False)
print('改进版提交文件已保存到:', out)
print(sep.head(10))