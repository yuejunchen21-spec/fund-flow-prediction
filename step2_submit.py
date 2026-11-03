# -*- coding: utf-8 -*-
# 资金流入流出预测：第二步，用最笨的基线（最近30天平均）生成提交文件

import pandas as pd

# 1) 读取 + 按天加总（和上一步一样）
path = r'D:\分析数据\项目\资金流入流出预测-挑战Baseline\Purchase Redemption Data\user_balance_table.csv'
df = pd.read_csv(path)
daily = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
daily.columns = ['report_date', 'purchase', 'redeem']

# 2) 用"最近30天的平均值"当作9月每天的预测（最笨的基线，先拿到分数再说）
last30 = daily.tail(30)
purchase_pred = int(last30['purchase'].mean())
redeem_pred = int(last30['redeem'].mean())
print('最近30天平均申购:', purchase_pred)
print('最近30天平均赎回:', redeem_pred)

# 3) 生成 9月1日到9月30日 的30行
import datetime
sub = pd.DataFrame({'report_date': pd.date_range('2014-09-01', '2014-09-30')})
sub['report_date'] = sub['report_date'].dt.strftime('%Y%m%d')
sub['purchase'] = purchase_pred
sub['redeem'] = redeem_pred

# 4) 保存成提交文件（不带表头，和样例 comp_predict_table.csv 一致）
out = r'D:\分析数据\项目\资金流入流出预测-挑战Baseline\tc_comp_predict_table.csv'
sub.to_csv(out, index=False, header=False)
print('提交文件已保存到:', out)
print('共', len(sub), '行')
print(sub.head())