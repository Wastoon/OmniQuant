import math
from typing import Optional

import numpy as np
import pandas as pd


def _finite(value, default=None):
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _clip(value, low, high):
    return max(low, min(high, value))


def generate_trade_advice(
    asset_type: str,
    prices,
    dates=None,
    technical: Optional[dict] = None,
    drawdown: Optional[dict] = None,
    valuation: Optional[dict] = None,
    quant_score: Optional[dict] = None,
) -> dict:
    close = pd.Series(prices, dtype=float).dropna()
    if len(close) < 30:
        return {
            "action": "watch",
            "action_cn": "观察",
            "confidence": 20,
            "suggested_position": 0,
            "risk_level": "unknown",
            "reasons": ["有效价格数据不足，暂不生成交易建议"],
            "warnings": ["至少需要 30 个交易日数据"],
        }

    technical = technical or {}
    drawdown = drawdown or {}
    valuation = valuation or {}
    quant_score = quant_score or {}

    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60, min_periods=20).mean().iloc[-1]
    ret20 = close.iloc[-1] / close.iloc[-20] - 1
    vol20 = close.pct_change().tail(20).std() * np.sqrt(252)

    score = 50.0
    reasons = []
    warnings = []

    total_score = _finite(quant_score.get("total_score"))
    if total_score is not None:
        score += (total_score - 50) * 0.45
        reasons.append(f"多因子综合分 {total_score:.0f}")

    rsi = _finite(technical.get("rsi14"))
    macd = _finite(technical.get("macd_hist"))
    if close.iloc[-1] > ma20:
        score += 8
        reasons.append("价格站上 20 日均线")
    else:
        score -= 8
        warnings.append("价格低于 20 日均线")

    if pd.notna(ma60):
        if ma20 > ma60:
            score += 7
            reasons.append("20 日均线位于 60 日均线上方")
        else:
            score -= 7
            warnings.append("中期均线结构偏弱")

    if rsi is not None:
        if rsi < 30:
            score += 8
            reasons.append("RSI 处于超卖区，存在修复机会")
        elif rsi > 75:
            score -= 10
            warnings.append("RSI 过热，短线回撤风险上升")
        elif 45 <= rsi <= 65:
            score += 4
            reasons.append("RSI 位于健康区间")

    if macd is not None:
        if macd > 0:
            score += 5
            reasons.append("MACD 柱线为正")
        else:
            score -= 5
            warnings.append("MACD 动能偏弱")

    pe_pct = _finite(valuation.get("pe_pct_5y"))
    pb_pct = _finite(valuation.get("pb_pct_5y"))
    pct_values = [x for x in [pe_pct, pb_pct] if x is not None]
    if asset_type == "stock" and pct_values:
        val_pct = sum(pct_values) / len(pct_values)
        if val_pct <= 35:
            score += 9
            reasons.append(f"估值处于历史较低分位 {val_pct:.0f}%")
        elif val_pct >= 75:
            score -= 12
            warnings.append(f"估值处于历史较高分位 {val_pct:.0f}%")

    max_dd = _finite(drawdown.get("max_drawdown"))
    cur_dd = _finite(drawdown.get("current_drawdown"))
    if max_dd is not None and max_dd < -35:
        score -= 8
        warnings.append("历史最大回撤较深，需降低单笔仓位")
    if cur_dd is not None and cur_dd < -20 and ret20 > 0:
        score += 4
        reasons.append("大回撤后出现阶段性企稳")

    if pd.notna(vol20) and vol20 > 0.45:
        score -= 6
        warnings.append("近 20 日波动率偏高")

    score = round(_clip(score, 0, 100))
    if score >= 72:
        action, action_cn = "buy", "分批买入"
    elif score >= 58:
        action, action_cn = "hold", "持有/等待加仓"
    elif score >= 42:
        action, action_cn = "watch", "观察"
    else:
        action, action_cn = "reduce", "减仓/回避"

    position = int(_clip((score - 30) * 1.25, 0, 90))
    if action in ("watch", "reduce"):
        position = min(position, 35 if action == "watch" else 15)

    risk_level = "high" if score < 45 or (max_dd is not None and max_dd < -35) else "medium"
    if score >= 70 and not warnings:
        risk_level = "low"

    return {
        "action": action,
        "action_cn": action_cn,
        "score": score,
        "confidence": int(_clip(abs(score - 50) * 1.4 + 35, 20, 90)),
        "suggested_position": position,
        "risk_level": risk_level,
        "reasons": reasons[:6] or ["信号中性，等待更明确的价格或基本面变化"],
        "warnings": warnings[:6],
        "stop_loss_pct": 8 if asset_type == "stock" else 5,
        "take_profit_pct": 18 if asset_type == "stock" else 12,
    }


def calc_trading_ranges(
    prices,
    dates=None,
    window: int = 20,
    band_pct: float = 0.20,
    buy_threshold: float = 0.65,
    sell_threshold: float = 0.20,
    smooth_window: int = 3,
) -> dict:
    close = pd.Series(prices, dtype=float).dropna().reset_index(drop=True)
    if len(close) < max(window, 5):
        return {
            "error": f"有效价格数据不足，至少需要 {max(window, 5)} 条",
            "window": window,
            "band_pct": band_pct,
        }

    if dates is None:
        labels = [str(i) for i in range(len(close))]
    else:
        labels = [str(d) for d in list(dates)[-len(close):]]

    mid = close.rolling(window, min_periods=1).mean()
    upper = mid * (1 + band_pct)
    lower = mid * (1 - band_pct)
    width = (upper - lower).replace(0, np.nan)

    raw_position = ((upper - close) / width).clip(0, 1)
    ref_position = raw_position.rolling(smooth_window, min_periods=1).mean().clip(0, 1)

    def signal_for(pos):
        if pos >= buy_threshold:
            return "buy"
        if pos <= sell_threshold:
            return "sell"
        return "hold"

    signals = [signal_for(float(v)) for v in ref_position]

    ranges = []
    start_idx = 0
    cur_signal = signals[0]
    for i, sig in enumerate(signals[1:], start=1):
        if sig != cur_signal:
            if cur_signal in ("buy", "sell"):
                ranges.append({
                    "type": cur_signal,
                    "start_idx": start_idx,
                    "end_idx": i - 1,
                    "start": labels[start_idx],
                    "end": labels[i - 1],
                    "avg_position": round(float(ref_position.iloc[start_idx:i].mean()), 3),
                })
            start_idx = i
            cur_signal = sig
    if cur_signal in ("buy", "sell"):
        ranges.append({
            "type": cur_signal,
            "start_idx": start_idx,
            "end_idx": len(signals) - 1,
            "start": labels[start_idx],
            "end": labels[-1],
            "avg_position": round(float(ref_position.iloc[start_idx:].mean()), 3),
        })

    latest_position = float(ref_position.iloc[-1])
    latest_signal = signal_for(latest_position)
    latest_price = float(close.iloc[-1])
    latest_mid = float(mid.iloc[-1])
    latest_upper = float(upper.iloc[-1])
    latest_lower = float(lower.iloc[-1])

    return {
        "method": "trend_channel_position",
        "window": window,
        "band_pct": band_pct,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "dates": labels,
        "mid": [round(float(x), 4) for x in mid],
        "upper": [round(float(x), 4) for x in upper],
        "lower": [round(float(x), 4) for x in lower],
        "reference_position": [round(float(x), 4) for x in ref_position],
        "signals": signals,
        "ranges": ranges,
        "latest": {
            "signal": latest_signal,
            "reference_position": round(latest_position, 3),
            "suggested_position_pct": int(round(latest_position * 100)),
            "buy_zone_price": round(latest_lower + (latest_upper - latest_lower) * (1 - buy_threshold), 4),
            "sell_zone_price": round(latest_lower + (latest_upper - latest_lower) * (1 - sell_threshold), 4),
            "lower": round(latest_lower, 4),
            "mid": round(latest_mid, 4),
            "upper": round(latest_upper, 4),
            "price": round(latest_price, 4),
        },
        "explain": [
            "趋势中轨采用滚动均线。",
            "上下轨为中轨乘以 1±band_pct。",
            "参考仓位=(上轨-当前价格)/(上轨-下轨)，并限制在 0~1。",
            "参考仓位高于 buy_threshold 视为买入/高仓位区，低于 sell_threshold 视为减仓区。",
        ],
    }


def calc_ema_trailing_strategy(df: pd.DataFrame, initial_cash: float = 100000.0, start_date: str = None, end_date: str = None) -> dict:
    if df is None or df.empty or "close" not in df.columns:
        return {}

    data = df.copy()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["close"]).reset_index(drop=True)
    if len(data) < 30:
        return {}

    close = data["close"]
    high = data.get("high", close)
    low = data.get("low", close)
    dates = data.get("date", data.index)

    # 1. 计算平滑曲线 EMA30 (更平稳的趋势判断)
    ema20 = close.ewm(span=30, adjust=False).mean()

    # 2. 计算 ATR (Average True Range) 用于判断波动和假性下跌
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # 3. 计算 10日最高价用于突破确认
    highest_10 = close.rolling(10).max().shift(1)

    # 初始化变量
    trailing_stop = pd.Series(np.nan, index=close.index, dtype=float)
    position_series = pd.Series(0.0, index=close.index, dtype=float)
    
    cash = float(initial_cash)
    shares = 0.0
    trades = []
    equity_curve = []
    
    in_position = False
    stop_price = 0.0
    
    buy_signals = [None] * len(close)
    sell_signals = [None] * len(close)
    
    # 将 start_date 和 end_date 转换为可比较的格式
    start_idx = 30
    end_idx = len(close)
    
    if start_date or end_date:
        for i in range(len(dates)):
            date_str = dates.iloc[i].strftime("%Y-%m-%d") if hasattr(dates.iloc[i], 'strftime') else str(dates.iloc[i])
            if start_date and date_str < start_date:
                start_idx = max(start_idx, i + 1)
            if end_date and date_str > end_date:
                end_idx = min(end_idx, i)

    for i in range(start_idx, end_idx):
        cur_price = float(close.iloc[i])
        cur_ema = float(ema20.iloc[i])
        prev3_ema = float(ema20.iloc[i-3]) if i >= 3 else float(ema20.iloc[i-1])
        cur_atr = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else float(cur_price * 0.02)
        cur_h10 = float(highest_10.iloc[i]) if pd.notna(highest_10.iloc[i]) else cur_price
        
        # 趋势向上的判断：EMA明显上翘 且 价格在 EMA 之上，并且突破近期高点确认启动
        ema_slope = (cur_ema - prev3_ema) / prev3_ema
        trend_up = (ema_slope > 0.002) and (cur_price > cur_ema) and (cur_price >= cur_h10)
        
        if not in_position:
            # 买入逻辑：捕获上升趋势的中段
            if trend_up:
                in_position = True
                # 初始止损线设在当前价格下方 3.0 倍 ATR 处，容忍正常的回调震荡
                stop_price = cur_price - 3.0 * cur_atr
                # 建议仓位：根据波动率动态调整，默认 90%
                suggested_pos = 0.90
                buy_amount = cash * suggested_pos
                shares = buy_amount / cur_price
                cash -= buy_amount
                
                date_str = dates.iloc[i].strftime("%Y-%m-%d") if hasattr(dates.iloc[i], 'strftime') else str(dates.iloc[i])
                trades.append({
                    "index": i, 
                    "date": date_str, 
                    "action": "buy", 
                    "price": round(cur_price, 3), 
                    "shares": round(shares, 2)
                })
                buy_signals[i] = cur_price
        else:
            # 动态抬升阶梯止损线 (Trailing Stop)
            # 使用买入以来的最高价来更新止损线
            highest_since_buy = close.iloc[trades[-1]["index"]:i+1].max()
            new_stop = highest_since_buy - 3.0 * cur_atr
            if new_stop > stop_price:
                stop_price = new_stop
                
            # 卖出逻辑：仅当跌破阶梯止损线时卖出 (避免假性下跌洗盘)
            if cur_price < stop_price:
                in_position = False
                cash += shares * cur_price
                
                date_str = dates.iloc[i].strftime("%Y-%m-%d") if hasattr(dates.iloc[i], 'strftime') else str(dates.iloc[i])
                buy_price = trades[-1]["price"]
                ret_pct = (cur_price / buy_price - 1) * 100
                trades.append({
                    "index": i, 
                    "date": date_str, 
                    "action": "sell", 
                    "price": round(cur_price, 3), 
                    "shares": round(shares, 2),
                    "return_pct": round(ret_pct, 2)
                })
                shares = 0.0
                stop_price = np.nan
                sell_signals[i] = cur_price
                
        if in_position:
            trailing_stop.iloc[i] = stop_price
            position_series.iloc[i] = 0.90
        else:
            position_series.iloc[i] = 0.0
            
        equity_curve.append(cash + shares * cur_price)

    # 补齐未计算期间的权益曲线
    full_equity = [initial_cash] * start_idx + equity_curve
    if len(full_equity) < len(close):
        full_equity.extend([full_equity[-1]] * (len(close) - len(full_equity)))
    total_return = full_equity[-1] / initial_cash - 1

    return {
        "ema20": [round(float(x), 4) if pd.notna(x) else None for x in ema20],
        "trailing_stop": [round(float(x), 4) if pd.notna(x) else None for x in trailing_stop],
        "buy_signals": [round(float(x), 4) if x is not None else None for x in buy_signals],
        "sell_signals": [round(float(x), 4) if x is not None else None for x in sell_signals],
        "position_series": [round(float(x), 2) for x in position_series],
        "trades": trades,
        "equity_curve": [round(float(x), 2) for x in full_equity],
        "final_equity": round(float(full_equity[-1]), 2),
        "total_return_pct": round(float(total_return * 100), 2)
    }


def backtest_strategy(df: pd.DataFrame, initial_cash: float = 100000.0) -> dict:
    if df is None or df.empty or "close" not in df.columns:
        raise ValueError("回测需要包含 close 列的价格数据")

    data = df.copy()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["close"]).reset_index(drop=True)
    if len(data) < 80:
        raise ValueError("回测至少需要 80 条价格数据")

    close = data["close"]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    momentum = close.pct_change(20)

    cash = float(initial_cash)
    shares = 0.0
    trades = []
    equity_curve = []

    for i in range(len(data)):
        price = float(close.iloc[i])
        if i >= 60:
            bullish = ma20.iloc[i] > ma60.iloc[i] and momentum.iloc[i] > 0
            bearish = ma20.iloc[i] < ma60.iloc[i] or momentum.iloc[i] < -0.08
            equity = cash + shares * price

            if bullish and shares == 0:
                spend = equity * 0.8
                shares = spend / price
                cash -= spend
                trades.append({"index": i, "action": "buy", "price": round(price, 4)})
            elif bearish and shares > 0:
                cash += shares * price
                trades.append({"index": i, "action": "sell", "price": round(price, 4)})
                shares = 0.0

        equity_curve.append(cash + shares * price)

    equity = pd.Series(equity_curve, dtype=float)
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / initial_cash - 1
    running_max = equity.cummax()
    max_drawdown = ((equity - running_max) / running_max).min()
    buy_hold = close.iloc[-1] / close.iloc[0] - 1
    sharpe = None
    if len(returns) > 5 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)

    return {
        "initial_cash": initial_cash,
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(float(total_return * 100), 2),
        "buy_hold_return_pct": round(float(buy_hold * 100), 2),
        "excess_return_pct": round(float((total_return - buy_hold) * 100), 2),
        "max_drawdown_pct": round(float(max_drawdown * 100), 2),
        "sharpe": round(float(sharpe), 3) if sharpe is not None else None,
        "trade_count": len(trades),
        "trades": trades[-20:],
        "equity_curve": [round(float(x), 2) for x in equity.tolist()],
    }
