# -*- coding: utf-8 -*-
# 资金流入流出预测：第一步，按天汇总申购/赎回，画历史曲线

import pandas as pd
import matplotlib.pyplot as plt

# 1) 读取数据（157MB，可能要等十几秒，正常）
path = r'D:\分析数据\项目\资金流入流出预测-挑战Baseline\Purchase Redemption Data\user_balance_table.csv'
df = pd.read_csv(path)

# 2) 按日期分组，把申购、赎回分别求和
daily = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
daily.columns = ['report_date', 'purchase', 'redeem']

# 3) 把 20140805 转成真正的日期，并按时间排序
daily['report_date'] = pd.to_datetime(daily['report_date'], format='%Y%m%d')
daily = daily.sort_values('report_date')

# 4) 画图：两条线，横轴日期、纵轴金额（单位：分）
plt.figure(figsize=(12, 5))
plt.plot(daily['report_date'], daily['purchase'], label='总申购 purchase', linewidth=1)
plt.plot(daily['report_date'], daily['redeem'], label='总赎回 redeem', linewidth=1)
plt.xlabel('日期')
plt.ylabel('金额（分）')
plt.title('每天总申购 vs 总赎回（2013-07 至 2014-08）')
plt.legend()
plt.tight_layout()

out = r'D:\分析数据\项目\资金流入流出预测-挑战Baseline\daily_trend.png'
plt.savefig(out, dpi=120)
print('图已保存到:', out)
print('共', len(daily), '天')
print(daily.head(10))