"""
量化因子计算模块 v3.1
BUG FIX: calc_technical 中 tail_list(n=120) 硬编码
         导致切换 1Y/3Y/5Y 时技术面图表不变化。
         修复：tail_list 接收外部传入的 n 参数，
         由调用方根据 close 序列长度动态确定。
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("factors")


# ─────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────

def sf(v, default=None):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def percentile_rank(series: pd.Series, value: float) -> int:
    s = series.dropna()
    s = s[s > 0]
    if len(s) < 5:
        return 50
    return int((s < value).sum() / len(s) * 100)


def winsorize(s: pd.Series, lower=0.05, upper=0.95) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def z_score(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-10)


# ─────────────────────────────────────────────────────────────────
# 技术指标
# BUG FIX: tail_list 不再硬编码 n=120
#          而是输出全部数据点，让前端通过时间轴窗口来过滤显示范围。
#          这样 1Y/3Y/5Y 切换时，前端 brush/zoom 才能正确响应。
# ─────────────────────────────────────────────────────────────────

def calc_technical(close: pd.Series, volume: Optional[pd.Series] = None) -> dict:
    """
    计算主要技术指标
    修复：序列长度跟随传入的 close 全量数据，不再截断到 120 条。
    """
    if len(close) < 26:
        return {"error": "数据不足（需 ≥26 条）"}

    def last(s):
        v = s.dropna()
        return sf(v.iloc[-1]) if len(v) else None

    def to_list(s):
        """输出全量序列，保留所有精度"""
        if s is None or (hasattr(s, '__len__') and len(s) == 0):
            return []
        return s.round(4).tolist()

    # ── MA ───────────────────────────────────────────────────────
    ma5  = close.rolling(5,  min_periods=1).mean()
    ma10 = close.rolling(10, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean() if len(close) >= 60 else pd.Series(dtype=float)

    # ── MACD (12/26/9) ───────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif   = ema12 - ema26
    dea   = dif.ewm(span=9, adjust=False).mean()
    macd  = (dif - dea) * 2

    # ── RSI ──────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rsi14 = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    gain6  = delta.clip(lower=0).rolling(6, min_periods=1).mean()
    loss6  = (-delta.clip(upper=0)).rolling(6, min_periods=1).mean()
    rsi6   = 100 - 100 / (1 + gain6 / loss6.replace(0, np.nan))

    # ── Bollinger Bands (20, 2σ) ─────────────────────────────────
    bb_mid   = close.rolling(20, min_periods=1).mean()
    bb_std   = close.rolling(20, min_periods=1).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # ── KDJ (9,3,3) ──────────────────────────────────────────────
    low9  = close.rolling(9, min_periods=1).min()
    high9 = close.rolling(9, min_periods=1).max()
    rsv   = (close - low9) / (high9 - low9 + 1e-10) * 100
    k     = rsv.ewm(com=2, adjust=False).mean()
    d     = k.ewm(com=2, adjust=False).mean()
    j     = 3 * k - 2 * d

    # ── OBV ──────────────────────────────────────────────────────
    obv = None
    if volume is not None and len(volume) == len(close):
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (volume * direction).cumsum()

    # ── 趋势判断 ─────────────────────────────────────────────────
    c_last = last(close)
    ma20_v = last(ma20)
    rsi_v  = last(rsi14)
    macd_v = last(macd)

    trend      = "up"   if (ma20_v and c_last and c_last > ma20_v) else "down"
    rsi_zone   = "超卖" if rsi_v and rsi_v < 30 else "超买" if rsi_v and rsi_v > 70 else "正常"
    macd_signal = "多头" if macd_v and macd_v > 0 else "空头"

    return {
        # 当前值（标量）
        "ma5": last(ma5), "ma10": last(ma10), "ma20": last(ma20), "ma60": last(ma60),
        "dif": last(dif), "dea": last(dea), "macd_hist": last(macd),
        "rsi6": last(rsi6), "rsi14": last(rsi14),
        "bb_upper": last(bb_upper), "bb_mid": last(bb_mid), "bb_lower": last(bb_lower),
        "bb_pct_b": last(bb_pct_b), "bb_width": last(bb_width),
        "kdj_k": last(k), "kdj_d": last(d), "kdj_j": last(j),
        # 全量序列（BUG FIX: 不再 tail(120)，输出所有数据点）
        "close_series":    to_list(close),
        "ma5_series":      to_list(ma5),
        "ma20_series":     to_list(ma20),
        "ma60_series":     to_list(ma60),
        "dif_series":      to_list(dif),
        "dea_series":      to_list(dea),
        "macd_series":     to_list(macd),
        "rsi6_series":     to_list(rsi6),
        "rsi14_series":    to_list(rsi14),
        "bb_upper_series": to_list(bb_upper),
        "bb_mid_series":   to_list(bb_mid),
        "bb_lower_series": to_list(bb_lower),
        "kdj_k_series":    to_list(k),
        "kdj_d_series":    to_list(d),
        "kdj_j_series":    to_list(j),
        "obv_series":      to_list(obv) if obv is not None else [],
        # 信号
        "trend": trend,
        "rsi_zone": rsi_zone,
        "macd_signal": macd_signal,
    }


# ─────────────────────────────────────────────────────────────────
# 趋势通道 + 回撤
# ─────────────────────────────────────────────────────────────────

def calc_trend_channel(prices: list, window: int = 20, pct: float = 0.20) -> dict:
    s = pd.Series(prices, dtype=float)
    mid   = s.rolling(window, min_periods=1).mean()
    upper = mid * (1 + pct)
    lower = mid * (1 - pct)
    cur   = s.iloc[-1]
    cmid  = mid.iloc[-1]
    deviation_pct = (cur - cmid) / cmid * 100 if cmid else 0
    cup = upper.iloc[-1]; clo = lower.iloc[-1]
    if cur > cup:
        position_label = "上轨以上（超买）"
    elif cur > cmid:
        position_label = f"中轨~上轨 ({deviation_pct:+.1f}%)"
    elif cur > clo:
        position_label = f"下轨~中轨 ({deviation_pct:+.1f}%)"
    else:
        position_label = "下轨以下（超卖）"
    return {
        "mid":     mid.round(4).tolist(),
        "upper":   upper.round(4).tolist(),
        "lower":   lower.round(4).tolist(),
        "current_mid":    sf(cmid),
        "current_upper":  sf(cup),
        "current_lower":  sf(clo),
        "deviation_pct":  round(deviation_pct, 2),
        "position_label": position_label,
        "window": window, "pct": pct,
    }


def calc_drawdown(prices: list) -> dict:
    s     = pd.Series(prices, dtype=float)
    peak  = s.cummax()
    dd    = (s - peak) / peak * 100
    cur_dd = sf(dd.iloc[-1])
    max_dd = sf(dd.min())
    end_idx   = int(dd.idxmin())
    start_idx = int(s.iloc[:end_idx+1].idxmax()) if end_idx > 0 else 0
    underwater_days = int((dd < -5).sum())
    return {
        "drawdown_series":    dd.round(2).tolist(),
        "max_drawdown":       max_dd,
        "current_drawdown":   cur_dd,
        "max_dd_start_idx":   start_idx,
        "max_dd_end_idx":     end_idx,
        "underwater_days":    underwater_days,
        "recovery_needed_pct": round(-max_dd / (1 + max_dd / 100) if max_dd else 0, 2),
    }


def calc_returns(prices: list, dates: list) -> dict:
    if not prices:
        return {}
    s   = pd.Series(prices, index=pd.to_datetime(dates))
    now = s.index[-1]
    cur = s.iloc[-1]
    result = {
        "since_inception":   sf((cur / s.iloc[0] - 1) * 100),
        "annualized":        _annualized_return(s),
        "volatility_annual": _annual_volatility(s),
        "sharpe":            _sharpe_ratio(s),
        "calmar":            _calmar_ratio(s),
    }
    for label, months in [("1m",1),("3m",3),("6m",6),("1y",12),("3y",36),("5y",60)]:
        target = now - pd.DateOffset(months=months)
        sub = s[s.index >= target]
        if len(sub) > 1:
            result[label] = sf((cur / sub.iloc[0] - 1) * 100)
    return result


def _annualized_return(s):
    if len(s) < 2: return None
    years = (s.index[-1] - s.index[0]).days / 365.25
    if years < 0.1: return None
    return sf((( s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100)

def _annual_volatility(s):
    ret = s.pct_change().dropna()
    if len(ret) < 10: return None
    return sf(ret.std() * (252 ** 0.5) * 100)

def _sharpe_ratio(s, risk_free=0.02):
    ann_ret = _annualized_return(s)
    ann_vol = _annual_volatility(s)
    if ann_ret is None or ann_vol is None or ann_vol == 0: return None
    return sf((ann_ret / 100 - risk_free) / (ann_vol / 100))

def _calmar_ratio(s):
    dd = calc_drawdown(s.tolist())
    max_dd = dd.get("max_drawdown")
    ann = _annualized_return(s)
    if not max_dd or not ann or max_dd == 0: return None
    return sf(ann / abs(max_dd))


# ─────────────────────────────────────────────────────────────────
# 5 大量化因子（保持原版逻辑不变）
# ─────────────────────────────────────────────────────────────────

def factor_interest_free_liab(bs_df: pd.DataFrame) -> dict:
    if bs_df is None or bs_df.empty:
        return {"error": "资产负债表数据不可用"}
    col_map = {
        "ACCOUNTS_PAYABLE": "ap", "accounts_payable": "ap",
        "ADVANCE_RECEIVABLES": "ar", "adv_receipts": "ar",
        "CONTRACT_LIABILITIES": "cl", "contract_liab": "cl",
        "TOTAL_LIABILITIES": "tl", "total_liab": "tl",
        "REPORT_DATE": "date", "end_date": "date",
    }
    df = bs_df.rename(columns={k: v for k, v in col_map.items() if k in bs_df.columns})
    results = []
    date_col = "date" if "date" in df.columns else df.columns[0]
    df = df.sort_values(date_col, ascending=False).head(8)
    for _, row in df.iterrows():
        ap = sf(row.get("ap"), 0); ar = sf(row.get("ar"), 0)
        cl = sf(row.get("cl"), 0); tl = sf(row.get("tl"), 1)
        ifl = ap + ar + cl
        ratio = ifl / tl * 100 if tl > 0 else None
        results.append({"date": str(row.get(date_col, ""))[:10],
                        "accounts_payable": ap, "advance_receipts": ar + cl,
                        "total_liab": tl, "interest_free_liab": ifl, "ratio": sf(ratio)})
    latest = results[0] if results else {}
    ratio = latest.get("ratio")
    if ratio is None:   grade, advice = "N/A", "数据不足"
    elif ratio > 60:    grade, advice = "A",   "产业链话语权极强，类金融模式"
    elif ratio > 40:    grade, advice = "B",   "上下游地位较强，占用外部资金"
    elif ratio > 25:    grade, advice = "C",   "行业平均水平"
    else:               grade, advice = "D",   "处于弱势地位，依赖有息负债"
    return {"current_ratio": ratio, "grade": grade, "advice": advice,
            "history": results, "score": min(100, max(0, (ratio or 0) * 1.4))}


def factor_gross_margin_stability(fin_df: pd.DataFrame) -> dict:
    if fin_df is None or fin_df.empty:
        return {"error": "财务指标数据不可用"}
    gm_col = None
    for c in ["销售毛利率(%)", "grossprofit_margin", "gross_margin", "GROSS_PROFIT_RATIO"]:
        if c in fin_df.columns: gm_col = c; break
    if gm_col is None:
        return {"error": "找不到毛利率字段"}
    date_col = "报告期" if "报告期" in fin_df.columns else ("end_date" if "end_date" in fin_df.columns else fin_df.columns[0])
    df = fin_df[[date_col, gm_col]].copy()
    df[gm_col] = pd.to_numeric(df[gm_col], errors="coerce")
    df = df.dropna().sort_values(date_col)
    margins = df[gm_col].values
    dates   = df[date_col].astype(str).values
    if len(margins) < 4:
        return {"error": "历史季度数不足"}
    mean = float(np.mean(margins)); std = float(np.std(margins, ddof=1))
    cv = std / mean * 100 if mean != 0 else 999
    trend_slope = float(np.polyfit(range(len(margins)), margins, 1)[0])
    if cv < 5:   grade, advice = "A", "极度稳定，定价权极强"
    elif cv < 10: grade, advice = "B", "稳定性良好，有较强竞争壁垒"
    elif cv < 20: grade, advice = "C", "中等稳定，受成本/竞争影响有限"
    else:         grade, advice = "D", "波动较大，缺乏成本转嫁能力"
    return {"mean": round(mean, 2), "std": round(std, 2), "cv": round(cv, 2),
            "min": round(float(np.min(margins)), 2), "max": round(float(np.max(margins)), 2),
            "trend_slope": round(trend_slope, 4),
            "trend_dir": "↑ 上升" if trend_slope > 0.1 else ("↓ 下降" if trend_slope < -0.1 else "→ 平稳"),
            "grade": grade, "advice": advice,
            "score": round(max(0, min(100, 100 - cv * 2.5))),
            "history": [{"date": str(d)[:10], "value": round(float(v), 2)} for d, v in zip(dates, margins)],
            "trend": "stable" if cv < 10 else "volatile"}


def factor_rd_capitalization(inc_df: pd.DataFrame, bs_df: pd.DataFrame) -> dict:
    result = {"rd_expense": None, "rd_capitalized": None,
              "capitalization_rate": None, "grade": "N/A",
              "advice": "需与同行对比方有意义", "score": 50}
    if inc_df is not None and not inc_df.empty:
        for col in ["rd_exp", "RESEARCH_EXPENSE", "研发费用"]:
            if col in inc_df.columns:
                vals = pd.to_numeric(inc_df[col], errors="coerce").dropna()
                if len(vals): result["rd_expense"] = sf(vals.iloc[0])
                break
    if bs_df is not None and not bs_df.empty:
        for col in ["INTANGIBLE_ASSETS", "intangible_assets", "无形资产"]:
            if col in bs_df.columns:
                vals = pd.to_numeric(bs_df[col], errors="coerce").dropna()
                if len(vals) >= 2:
                    delta = float(vals.iloc[0]) - float(vals.iloc[1])
                    if delta > 0: result["rd_capitalized"] = round(delta)
                break
    rd_fee = result["rd_expense"] or 0; rd_cap = result["rd_capitalized"] or 0
    total  = rd_fee + rd_cap
    if total > 0:
        rate = rd_cap / total * 100; result["capitalization_rate"] = round(rate, 2)
        if rate < 10:   result["grade"] = "A"; result["advice"] = "研发以费用化为主，利润含金量高";     result["score"] = 85
        elif rate < 30: result["grade"] = "B"; result["advice"] = "资本化比例适中，属正常范围";         result["score"] = 65
        elif rate < 50: result["grade"] = "C"; result["advice"] = "资本化比例偏高，需对比同行均值";     result["score"] = 45
        else:           result["grade"] = "D"; result["advice"] = "⚠ 资本化比例显著偏高，警惕利润水分"; result["score"] = 20
    return result


def factor_dividend_payout(div_df: pd.DataFrame, fin_df: pd.DataFrame) -> dict:
    result = {"annual_dividend_per_share": None, "eps": None,
              "payout_ratio": None, "grade": "N/A", "advice": "数据不足", "score": 50, "history": []}
    if fin_df is not None and not fin_df.empty:
        for col in ["基本每股收益(元)", "eps", "EPS"]:
            if col in fin_df.columns:
                vals = pd.to_numeric(fin_df[col], errors="coerce").dropna()
                if len(vals): result["eps"] = sf(vals.iloc[0])
                break
    if div_df is not None and not div_df.empty:
        for col in ["每股分红", "cash_div_tax", "CASH_DIV_TAX"]:
            if col in div_df.columns:
                vals = pd.to_numeric(div_df[col], errors="coerce").dropna()
                annual_div = float(vals.head(4).sum())
                result["annual_dividend_per_share"] = round(annual_div, 4)
                result["history"] = vals.head(8).round(4).tolist()
                break
    eps = result["eps"]; div = result["annual_dividend_per_share"]
    if eps and eps > 0 and div is not None:
        ratio = div / eps * 100; result["payout_ratio"] = round(ratio, 2)
        if ratio > 100:   result["grade"] = "D"; result["advice"] = "⚠ 支付率>100%，正在吃老本，不可持续"; result["score"] = 20
        elif ratio > 70:  result["grade"] = "C"; result["advice"] = "支付率偏高，成长空间受限";           result["score"] = 55
        elif ratio >= 30: result["grade"] = "A"; result["advice"] = "支付率健康（30-70%）";              result["score"] = 90
        elif ratio > 0:   result["grade"] = "B"; result["advice"] = "支付率偏低，可能优先留存再投资";    result["score"] = 65
        else:             result["grade"] = "D"; result["advice"] = "不分红，观察资金使用效率";          result["score"] = 35
    return result


def factor_institution_holdings(holdings_history: list) -> dict:
    if not holdings_history or len(holdings_history) < 2:
        return {"latest_fund_count": None, "slope_fund_count": None,
                "trend": "数据不足", "grade": "N/A", "advice": "需至少2期数据", "score": 50, "history": []}
    hist = sorted(holdings_history, key=lambda x: x.get("date", ""))
    counts = [h["fund_count"] for h in hist]
    shares = [h.get("total_shares", 0) for h in hist]
    n = len(counts); x = list(range(n))
    slope_cnt = float(np.polyfit(x, counts, 1)[0]) if n >= 2 else 0
    slope_shr = float(np.polyfit(x, shares, 1)[0]) if n >= 2 else 0
    latest_cnt = counts[-1]
    pct_change_4q = (counts[-1] - counts[0]) / (counts[0] + 1e-10) * 100
    if slope_cnt > 2 and pct_change_4q > 10: grade, advice = "A", "机构持续增持，长线资金认可基本面"
    elif slope_cnt > 0:                       grade, advice = "B", "机构小幅增持，关注后续趋势"
    elif slope_cnt > -2:                      grade, advice = "C", "机构持仓基本稳定"
    else:                                     grade, advice = "D", "机构持续减仓，需关注基本面变化"
    return {"latest_fund_count": latest_cnt, "slope_fund_count": round(slope_cnt, 2),
            "slope_shares": round(slope_shr), "pct_change_4q": round(pct_change_4q, 2),
            "trend": "持续增持" if slope_cnt > 2 else ("持续减仓" if slope_cnt < -2 else "小幅变化"),
            "grade": grade, "advice": advice,
            "score": round(min(100, max(0, 60 + slope_cnt * 5 + pct_change_4q * 0.3))),
            "history": hist[-8:]}


# ─────────────────────────────────────────────────────────────────
# PE / PB 历史百分位（保持原版逻辑不变）
# ─────────────────────────────────────────────────────────────────

def calc_pe_pb_percentile(pe_pb_df: pd.DataFrame) -> dict:
    if pe_pb_df is None or pe_pb_df.empty:
        return {"error": "估值历史数据不可用"}
    for old, new in [("pe_ttm", "pe"), ("trade_date", "date")]:
        if old in pe_pb_df.columns and new not in pe_pb_df.columns:
            pe_pb_df = pe_pb_df.rename(columns={old: new})
    date_col = "date" if "date" in pe_pb_df.columns else pe_pb_df.columns[0]
    pe_pb_df[date_col] = pd.to_datetime(pe_pb_df[date_col])
    pe_pb_df = pe_pb_df.sort_values(date_col)
    now = pe_pb_df[date_col].max()
    result = {}
    for metric in ["pe", "pb"]:
        if metric not in pe_pb_df.columns: continue
        series = pd.to_numeric(pe_pb_df[metric], errors="coerce")
        dates  = pe_pb_df[date_col]
        valid_mask = series > 0
        series = series[valid_mask]; dates = dates[valid_mask]
        cur = sf(series.iloc[-1]) if len(series) else None
        if cur is None: continue
        result[f"{metric}_current"] = round(cur, 2)
        for yr, label in [(1, "1y"), (3, "3y"), (5, "5y")]:
            cutoff = now - pd.DateOffset(years=yr)
            sub    = series[dates >= cutoff]
            if len(sub) >= 10:
                pct = percentile_rank(sub, cur)
                result[f"{metric}_pct_{label}"]  = pct
                result[f"{metric}_mean_{label}"]  = round(sub.mean(), 2)
                result[f"{metric}_low_{label}"]   = round(sub.quantile(0.1), 2)
                result[f"{metric}_high_{label}"]  = round(sub.quantile(0.9), 2)
        result[f"{metric}_history"] = series.tail(365 * 5).round(2).tolist()
        result[f"{metric}_dates"]   = dates.tail(365 * 5).dt.strftime("%Y-%m-%d").tolist()
    pe5 = result.get("pe_pct_5y"); pb5 = result.get("pb_pct_5y")
    if pe5 is not None and pb5 is not None:
        avg = (pe5 + pb5) / 2
        result["overall_pct_5y"] = round(avg)
        result["valuation_verdict"] = ("历史低估区" if avg < 25 else "略微低估" if avg < 40 else
                                       "合理区间"   if avg < 65 else "略微高估" if avg < 80 else "历史高估区")
    return result


# ─────────────────────────────────────────────────────────────────
# 行业中性化 + 多因子评分（保持原版逻辑不变）
# ─────────────────────────────────────────────────────────────────

def calc_industry_percentile(value, industry_values, metric_name="指标", higher_is_better=True) -> dict:
    arr = pd.Series(industry_values).dropna()
    if len(arr) < 3:
        return {"percentile": None, "rank": None, "total": len(arr)}
    pct = percentile_rank(arr, value)
    if not higher_is_better: pct = 100 - pct
    rank = len(arr) - int((arr < value if higher_is_better else arr > value).sum())
    return {"value": round(value, 4), "percentile": pct, "rank": rank, "total": len(arr),
            "median": round(float(arr.median()), 4), "mean": round(float(arr.mean()), 4),
            "is_top20pct": pct >= 80, "is_bottom20pct": pct <= 20,
            "label": f"行业前 {100-pct}%" if higher_is_better else f"行业第 {pct}% 低估",
            "metric": metric_name}


DEFAULT_WEIGHTS = {"profit_quality": 0.20, "balance_sheet": 0.20,
                   "valuation": 0.20, "cashflow": 0.15,
                   "technical": 0.15, "risk": 0.10}


def multi_factor_score(fin_factors, valuation, technical, drawdown, weights=None) -> dict:
    w = weights or DEFAULT_WEIGHTS
    total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()}
    dims = {}
    roe = fin_factors.get("roe")
    gm_score = fin_factors.get("gross_margin_score", 50)
    roe_score = min(100, max(0, 50 + (roe - 10) * 2.5)) if roe is not None else 50
    dims["profit_quality"] = round((roe_score + gm_score) / 2)
    ifl_score = fin_factors.get("interest_free_liab_score", 50)
    rd_score  = fin_factors.get("rd_cap_score", 50)
    dims["balance_sheet"] = round(ifl_score * 0.7 + rd_score * 0.3)
    pe5 = valuation.get("pe_pct_5y"); pb5 = valuation.get("pb_pct_5y")
    val_scores = [100 - p for p in [pe5, pb5] if p is not None]
    dims["valuation"] = round(sum(val_scores) / len(val_scores)) if val_scores else 50
    fcf = fin_factors.get("fcf")
    div_score = fin_factors.get("dividend_score", 50)
    fcf_score = 80 if fcf and fcf > 0 else 25 if fcf is not None else 50
    dims["cashflow"] = round((fcf_score + div_score) / 2)
    rsi  = technical.get("rsi14"); macd = technical.get("macd_hist")
    rsi_score  = 90 if rsi and rsi < 30 else 40 if rsi and rsi > 70 else 70 if rsi else 50
    macd_score = 65 if macd and macd > 0 else 35 if macd and macd < 0 else 50
    trend_score = 75 if technical.get("trend") == "up" else 35
    dims["technical"] = round((rsi_score + macd_score + trend_score) / 3)
    max_dd = drawdown.get("max_drawdown")
    dims["risk"] = round(max(0, min(100, 100 + max_dd * 1.3))) if max_dd is not None else 50
    total = round(sum(dims[k] * w.get(k, 0) for k in dims))
    if total >= 70: signal, signal_cn = "buy", "推荐买入"
    elif total >= 50: signal, signal_cn = "hold", "持仓观望"
    else: signal, signal_cn = "sell", "建议减持"
    position = min(100, max(0, int((total - 30) * 100 / 70)))
    warnings = []
    if dims.get("balance_sheet", 50) < 40: warnings.append("无息负债比例偏低，注意债务风险")
    if dims.get("valuation", 50) < 30:     warnings.append("估值处于历史高位，上行空间有限")
    if dims.get("risk", 50) < 35:          warnings.append("历史最大回撤较大，注意仓位管理")
    if rsi and rsi > 75:                   warnings.append("RSI 超买，短期可能回调")
    return {"total_score": total, "signal": signal, "signal_cn": signal_cn,
            "suggested_position": position, "dimension_scores": dims,
            "weights_used": w, "warnings": warnings,
            "score_breakdown": [
                {"name": "盈利质量",   "key": "profit_quality", "score": dims.get("profit_quality", 0)},
                {"name": "资产负债",   "key": "balance_sheet",  "score": dims.get("balance_sheet", 0)},
                {"name": "估值空间",   "key": "valuation",      "score": dims.get("valuation", 0)},
                {"name": "现金流/股息","key": "cashflow",        "score": dims.get("cashflow", 0)},
                {"name": "技术面",     "key": "technical",      "score": dims.get("technical", 0)},
                {"name": "风险控制",   "key": "risk",           "score": dims.get("risk", 0)},
            ]}