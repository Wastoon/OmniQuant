# 量化分析终端 v3.0

## 项目结构

```
quant_v3/
├── main.py                  ← FastAPI 入口
├── core/
│   ├── datasource.py        ← 数据源适配器（AkShare + Tushare 双源）
│   └── factors.py           ← 所有量化因子计算
├── frontend.html            ← 前端页面
└── README.md
```

---

## 快速启动

```bash
# 1. 安装依赖
pip install akshare fastapi uvicorn pandas numpy scipy

# 2. 启动（仅 AkShare，免费无需注册）
cd quant_v3
python main.py

# 3. 同时使用 Tushare（数据更全）
TUSHARE_TOKEN=你的token python main.py

# 4. 打开前端
# 直接在浏览器打开 frontend.html 文件
# 首次打开会弹出配置弹窗，填入 http://localhost:8000 即可
```

---

## 数据源切换逻辑

```
请求到达
  │
  ├─► AkShare 可用？
  │     └─► 是 → 调用 AkShare → 成功 → 返回数据 ✓
  │                            └─► 失败 → fallback ↓
  │
  └─► Tushare 可用且有 Token？
        └─► 是 → 调用 Tushare → 成功 → 返回数据 ✓
                               └─► 失败 → DataUnavailableError ✗
```

**接口响应中 `datasource_log` 字段**会显示每个方法实际使用的数据源。

---

## 5 大量化因子说明

### ① `factor_interest_free_liab` — 无息负债比例
```python
ratio = (应付账款 + 预收账款 + 合同负债) / 总负债 × 100

# 评级标准
A (>60%): 产业链话语权极强，类金融模式（茅台/格力）
B (>40%): 上下游地位较强
C (>25%): 行业平均水平
D (<25%): 依赖有息负债，债务结构差
```

### ② `factor_gross_margin_stability` — 毛利率稳定性
```python
CV = std / mean × 100   # 变异系数

# 评级标准（CV越小越好）
A (CV < 5%):  极度稳定，强定价权
B (CV < 10%): 稳定性良好
C (CV < 20%): 中等，受竞争影响
D (CV ≥ 20%): 缺乏成本转嫁能力
```

### ③ `factor_rd_capitalization` — R&D 资本化率
```python
rate = 资本化研发 / (资本化 + 费用化) × 100

# 警戒线：rate 显著高于同行业均值 1.5x → 警惕利润水分
```

### ④ `factor_dividend_payout` — 股息支付率
```python
payout_ratio = 年度分红总额 / 归母净利润 × 100

# 健康区间：30% ~ 70%
# > 100%: 吃老本，不可持续
# < 25%:  不重视股东
```

### ⑤ `factor_institution_holdings` — 机构持仓斜率
```python
slope = np.polyfit(range(n_quarters), fund_counts, 1)[0]

# 解读
slope > 2:  持续增持 → 长线资金认可
slope > 0:  小幅增持
slope > -2: 基本稳定
slope < -2: 持续减仓 → 需关注
```

---

## 多因子评分权重（可覆盖）

| 维度 | 默认权重 | API 参数名 |
|------|---------|-----------|
| 盈利质量 (ROE + 毛利稳定) | 20% | `profit_quality` |
| 资产负债结构 (无息负债 + R&D) | 20% | `balance_sheet` |
| 估值空间 (PE/PB 百分位) | 20% | `valuation` |
| 现金流/股息 | 15% | `cashflow` |
| 技术面 | 15% | `technical` |
| 风险控制 (最大回撤) | 10% | `risk` |

**自定义权重：**
```bash
# 提高估值权重，降低技术权重
curl "http://localhost:8000/api/stock/600519?weights={\"valuation\":0.35,\"technical\":0.05}"
```

---

## 行业中性化接口

```bash
# 对比白酒行业内几家公司的 ROE
curl "http://localhost:8000/api/industry/compare?codes=600519,000858,002304,603369&metric=roe"

# 对比 PE 估值（越低越好，需设 higher_is_better=false）
curl "http://localhost:8000/api/industry/compare?codes=600519,000858&metric=pe_ttm&higher_is_better=false"
```

返回每只股票在组内的百分位，`is_top20pct: true` 即为行业内 Top 20%。

---

## 技术指标说明

| 指标 | 参数 | 用途 |
|------|------|------|
| MA | 5/10/20/60 | 趋势方向 |
| MACD | 12/26/9 EMA | 动量信号 |
| RSI | 6/14 | 超买/超卖 |
| Bollinger | 20期, 2σ | 价格通道 |
| KDJ | 9/3/3 | 短期反转 |
| OBV | — | 量价配合 |

---

## 常见问题

**Q: 财务数据为空/报错？**
A: 检查响应中的 `_err_*` 字段，常见原因：
- 网络问题（AkShare 需访问东方财富等国内平台）
- 新上市公司数据不全
- Tushare 积分不足（财务数据需 120 积分）

**Q: PE/PB 历史数据拉不到？**
A: AkShare 的 `stock_a_lg_indicator` 需要网络畅通；
Tushare 需要 `daily_basic` 接口权限（免费版受限）。

**Q: 如何替换数据源？**
A: `core/datasource.py` 中添加新的适配器类，实现相同方法签名，
然后在 `DataSource._call()` 的 fallback 链中加入即可。
