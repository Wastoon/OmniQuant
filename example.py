'''
Author: fufeng
Description: 
Date: 2026-03-16 23:32:00
LastEditTime: 2026-03-17 00:43:06
FilePath: /quant_v3/example.py
'''
from core.datasource.datasource import DataSource

ds = DataSource()

df = ds.stock_hist("000858", "20230101", "20250101")

print(df.head())
