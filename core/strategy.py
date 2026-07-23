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


def calc_startup_strategy(
    df: pd.DataFrame,
    initial_cash: float = 100000.0,
    ene_period: int = 10,
    ene_upper_pct: float = 11.0,
    ene_lower_pct: float = 9.0,
    dma_short: int = 10,
    dma_long: int = 50,
    dma_signal: int = 10,
    confirmation_window: int = 60,
    volume_multiplier: float = 1.05,
    volume_confirm: bool = True,
    stop_loss_pct: float = 0.08,
    trailing_stop_pct: float = 0.10,
    fee_rate: float = 0.0003,
    slippage_rate: float = 0.0005,
) -> dict:
    """前复权趋势启动策略：MA 结构 -> ENE -> DMA -> MACD 依次确认。

    df 的 close 应由调用方传入前复权收盘价。该函数只生成信号和一个
    可复现的单仓位回测，避免把未复权价格与复权指标混用。
    """
    if df is None or df.empty or "close" not in df.columns:
        return {}

    data = df.copy()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["close"]).reset_index(drop=True)
    if len(data) < max(60, dma_long + dma_signal):
        return {"error": f"有效价格数据不足，至少需要 {max(60, dma_long + dma_signal)} 条"}

    close = data["close"].astype(float)
    volume = pd.to_numeric(
        data.get("volume", pd.Series(np.nan, index=data.index)), errors="coerce"
    )
    dates = data["date"] if "date" in data.columns else pd.Series(data.index, index=data.index)

    def date_str(index):
        value = dates.iloc[index]
        return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)

    def cross_up(left, right):
        return (left > right) & (left.shift(1) <= right.shift(1))

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma30 = close.rolling(30).mean()

    ene_mid = close.rolling(ene_period).mean()
    ene_upper = ene_mid * (1 + ene_upper_pct / 100)
    ene_lower = ene_mid * (1 - ene_lower_pct / 100)

    dma_ddd = close.rolling(dma_short).mean() - close.rolling(dma_long).mean()
    dma_ama = dma_ddd.rolling(dma_signal).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_dif = ema12 - ema26
    macd_dea = macd_dif.ewm(span=9, adjust=False).mean()
    macd_hist = (macd_dif - macd_dea) * 2

    volume_ma20 = volume.rolling(20, min_periods=10).mean()
    volume_ok = (~volume.isna() & ~volume_ma20.isna() &
                 (volume >= volume_ma20 * volume_multiplier))
    if not volume_confirm:
        volume_ok = pd.Series(True, index=close.index)

    # MA20/30 只允许平缓或抬头，并且长均线仍压在短均线之上。
    slope5 = lambda series: series / series.shift(5) - 1
    long_ma_ok = (
        slope5(ma20).fillna(-1) >= -0.002
    ) & (
        slope5(ma30).fillna(-1) >= -0.002
    ) & (ma20 >= ma5) & (ma20 >= ma10) & (ma30 >= ma5) & (ma30 >= ma10)
    candidate = (
        cross_up(ma5, ma10)
        & (ma5 > ma10)
        & long_ma_ok
    ).fillna(False)

    ene_cross = (
        cross_up(close, ene_mid)
        & (close >= ene_lower)
        & (close <= ene_upper)
        & (slope5(ene_mid) >= -0.002)
    ).fillna(False)
    dma_cross = cross_up(dma_ddd, dma_ama).fillna(False)
    macd_cross = (
        cross_up(macd_dif, macd_dea)
        & (macd_hist > macd_hist.shift(1))
    ).fillna(False)

    buy_signals = [None] * len(close)
    sell_signals = [None] * len(close)
    stage_series = [0] * len(close)
    events = []
    pending_candidate = pending_ene = pending_dma = None

    for i in range(len(close)):
        if candidate.iloc[i]:
            pending_candidate = i
            pending_ene = pending_dma = None
        if pending_candidate is not None and i - pending_candidate > confirmation_window:
            pending_candidate = pending_ene = pending_dma = None
        if pending_candidate is not None:
            if pending_ene is None and i >= pending_candidate and ene_cross.iloc[i]:
                pending_ene = i
            if pending_ene is not None and pending_dma is None and i >= pending_ene and dma_cross.iloc[i]:
                pending_dma = i
            if pending_dma is not None and i >= pending_dma and macd_cross.iloc[i]:
                stage_series[i] = 4
                events.append({
                    "index": i, "type": "startup",
                    "candidate_index": pending_candidate,
                    "ene_index": pending_ene,
                    "dma_index": pending_dma,
                    "macd_index": i,
                    "volume_confirmed": bool(volume_ok.iloc[i]),
                })
                buy_signals[i] = float(close.iloc[i])
                pending_candidate = pending_ene = pending_dma = None
        if stage_series[i] == 0:
            stage_series[i] = (
                3 if pending_dma is not None else
                2 if pending_ene is not None else
                1 if pending_candidate is not None else 0
            )

    # 生成卖出信号：硬止损优先；正常卖出需要趋势/动能至少两项转弱，
    # 并用两日确认降低单日噪声。
    cash = float(initial_cash)
    shares = 0.0
    entry_price = None
    peak_price = None
    entry_index = None
    trades = []
    equity_curve = []
    weakness_days = 0
    for i, price in enumerate(close):
        price = float(price)
        if buy_signals[i] is not None and shares == 0:
            buy_price = price * (1 + slippage_rate)
            spend = cash * 0.90
            shares = spend * (1 - fee_rate) / buy_price
            cash -= spend
            entry_price, peak_price, entry_index = price, price, i
            trades.append({
                "index": i, "date": date_str(i), "action": "buy",
                "price": round(price, 4), "exec_price": round(buy_price, 4),
                "position_pct": 90.0,
                "reason": "MA→ENE→DMA→MACD 四阶段确认",
            })
        elif shares > 0:
            peak_price = max(peak_price, price)
            ma_weak = bool(ma5.iloc[i] < ma10.iloc[i] and close.iloc[i] < ma20.iloc[i])
            dma_weak = bool(dma_ddd.iloc[i] < dma_ama.iloc[i])
            macd_weak = bool(macd_dif.iloc[i] < macd_dea.iloc[i] and macd_hist.iloc[i] < 0)
            structure_weak = bool(
                (close.iloc[i] < ma30.iloc[i]) or
                (slope5(ma20).iloc[i] < -0.002 and slope5(ma30).iloc[i] < -0.002)
            )
            weakness = sum([ma_weak, dma_weak, macd_weak, structure_weak])
            weakness_days = weakness_days + 1 if weakness >= 2 else 0
            hard_stop = price <= entry_price * (1 - stop_loss_pct)
            trailing_stop = price <= peak_price * (1 - trailing_stop_pct)
            normal_exit = weakness_days >= 2 or price < ene_lower.iloc[i]
            reason = "hard_stop" if hard_stop else (
                "trailing_stop" if trailing_stop else
                "trend_and_momentum_weak" if normal_exit else None
            )
            if reason:
                sell_price = price * (1 - slippage_rate)
                cash += shares * sell_price * (1 - fee_rate)
                trades.append({
                    "index": i, "date": date_str(i), "action": "sell",
                    "price": round(price, 4), "exec_price": round(sell_price, 4),
                    "return_pct": round((sell_price / entry_price - 1) * 100, 2),
                    "holding_days": i - entry_index, "reason": reason,
                })
                sell_signals[i] = price
                shares = 0.0
                entry_price = peak_price = entry_index = None
                weakness_days = 0
        equity_curve.append(cash + shares * price)

    indicators = {
        "close": close, "volume": volume, "volume_ma20": volume_ma20,
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma30": ma30,
        "ene_mid": ene_mid, "ene_upper": ene_upper, "ene_lower": ene_lower,
        "dma_ddd": dma_ddd, "dma_ama": dma_ama,
        "macd_dif": macd_dif, "macd_dea": macd_dea, "macd_hist": macd_hist,
    }
    serial = lambda series: [round(float(x), 6) if pd.notna(x) else None for x in series]
    return {
        "method": "qfq_ma_ene_dma_macd_startup",
        "params": {
            "ene_period": ene_period, "ene_upper_pct": ene_upper_pct,
            "ene_lower_pct": ene_lower_pct, "dma_short": dma_short,
            "dma_long": dma_long, "dma_signal": dma_signal,
            "confirmation_window": confirmation_window,
            "volume_multiplier": volume_multiplier,
            "volume_confirm": volume_confirm, "stop_loss_pct": stop_loss_pct,
            "trailing_stop_pct": trailing_stop_pct,
        },
        "dates": [date_str(i) for i in range(len(close))],
        "indicators": {name: serial(series) for name, series in indicators.items()},
        "events": events,
        "candidate_signals": [float(close.iloc[i]) if candidate.iloc[i] else None for i in range(len(close))],
        "ene_signals": [float(close.iloc[i]) if ene_cross.iloc[i] else None for i in range(len(close))],
        "dma_signals": [float(close.iloc[i]) if dma_cross.iloc[i] else None for i in range(len(close))],
        "macd_signals": [float(close.iloc[i]) if macd_cross.iloc[i] else None for i in range(len(close))],
        "stage_series": stage_series,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "trades": trades,
        "equity_curve": [round(float(value), 2) for value in equity_curve],
        "final_equity": round(float(equity_curve[-1]), 2),
        "total_return_pct": round((equity_curve[-1] / initial_cash - 1) * 100, 2),
        "latest_stage": stage_series[-1],
        "latest": {name: _finite(series.iloc[-1]) for name, series in indicators.items()},
    }


def calc_ema_trailing_strategy(
    df: pd.DataFrame,
    initial_cash: float = 100000.0,
    start_date: str = None,
    end_date: str = None,
    asset_type: str = "stock",
    ema_span: int = None,
    atr_window: int = None,
    breakout_window: int = None,
    stop_atr_multiplier: float = None,
    risk_per_trade: float = None,
    min_position: float = None,
    max_position: float = None,
    max_entry_extension: float = None,
    min_holding_days: int = None,
    cooldown_days: int = None,
    signal_confirm_days: int = None,
    sell_confirm_days: int = None,
    trend_exit_atr_buffer: float = None,
    hard_stop_atr_buffer: float = None,
    min_stop_loss_pct: float = None,
    long_ema_span: int = None,
    use_long_trend_filter: bool = None,
    max_atr_pct: float = None,
    profit_protect_trigger_pct: float = None,
    profit_stop_atr_multiplier: float = None,
    fee_rate: float = None,
    slippage_rate: float = None,
    volume_confirm: bool = None,
) -> dict:
    if df is None or df.empty or "close" not in df.columns:
        return {}

    data = df.copy()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["close"]).reset_index(drop=True)
    if len(data) < 30:
        return {}

    is_fund = asset_type == "fund"
    defaults = {
        "ema_span": 45 if is_fund else 30,
        "atr_window": 20 if is_fund else 14,
        "breakout_window": 20 if is_fund else 10,
        "stop_atr_multiplier": 4.0 if is_fund else 3.0,
        "risk_per_trade": 0.025 if is_fund else 0.01,
        "min_position": 0.35 if is_fund else 0.30,
        "max_position": 0.85 if is_fund else 0.90,
        "max_entry_extension": 0.12 if is_fund else 0.12,
        "min_holding_days": 30 if is_fund else 5,
        "cooldown_days": 10 if is_fund else 3,
        "signal_confirm_days": 2 if is_fund else 1,
        "sell_confirm_days": 2 if is_fund else 1,
        "trend_exit_atr_buffer": 1.0 if is_fund else 0.5,
        "hard_stop_atr_buffer": 2.0 if is_fund else 1.0,
        "min_stop_loss_pct": 0.08 if is_fund else 0.04,
        "long_ema_span": 120,
        "use_long_trend_filter": False if is_fund else True,
        "max_atr_pct": None if is_fund else 0.08,
        "profit_protect_trigger_pct": 0.18 if is_fund else 0.12,
        "profit_stop_atr_multiplier": 3.0 if is_fund else 2.2,
        "fee_rate": 0.0015 if is_fund else 0.0003,
        "slippage_rate": 0.0 if is_fund else 0.0005,
        "volume_confirm": False if is_fund else True,
    }
    ema_span = defaults["ema_span"] if ema_span is None else ema_span
    atr_window = defaults["atr_window"] if atr_window is None else atr_window
    breakout_window = defaults["breakout_window"] if breakout_window is None else breakout_window
    stop_atr_multiplier = defaults["stop_atr_multiplier"] if stop_atr_multiplier is None else stop_atr_multiplier
    risk_per_trade = defaults["risk_per_trade"] if risk_per_trade is None else risk_per_trade
    min_position = defaults["min_position"] if min_position is None else min_position
    max_position = defaults["max_position"] if max_position is None else max_position
    max_entry_extension = defaults["max_entry_extension"] if max_entry_extension is None else max_entry_extension
    min_holding_days = defaults["min_holding_days"] if min_holding_days is None else min_holding_days
    cooldown_days = defaults["cooldown_days"] if cooldown_days is None else cooldown_days
    signal_confirm_days = defaults["signal_confirm_days"] if signal_confirm_days is None else signal_confirm_days
    sell_confirm_days = defaults["sell_confirm_days"] if sell_confirm_days is None else sell_confirm_days
    trend_exit_atr_buffer = defaults["trend_exit_atr_buffer"] if trend_exit_atr_buffer is None else trend_exit_atr_buffer
    hard_stop_atr_buffer = defaults["hard_stop_atr_buffer"] if hard_stop_atr_buffer is None else hard_stop_atr_buffer
    min_stop_loss_pct = defaults["min_stop_loss_pct"] if min_stop_loss_pct is None else min_stop_loss_pct
    long_ema_span = defaults["long_ema_span"] if long_ema_span is None else long_ema_span
    use_long_trend_filter = defaults["use_long_trend_filter"] if use_long_trend_filter is None else use_long_trend_filter
    max_atr_pct = defaults["max_atr_pct"] if max_atr_pct is None else max_atr_pct
    profit_protect_trigger_pct = defaults["profit_protect_trigger_pct"] if profit_protect_trigger_pct is None else profit_protect_trigger_pct
    profit_stop_atr_multiplier = defaults["profit_stop_atr_multiplier"] if profit_stop_atr_multiplier is None else profit_stop_atr_multiplier
    fee_rate = defaults["fee_rate"] if fee_rate is None else fee_rate
    slippage_rate = defaults["slippage_rate"] if slippage_rate is None else slippage_rate
    volume_confirm = defaults["volume_confirm"] if volume_confirm is None else volume_confirm

    close = data["close"]
    high = pd.to_numeric(data.get("high", close), errors="coerce").fillna(close)
    low = pd.to_numeric(data.get("low", close), errors="coerce").fillna(close)
    volume = pd.to_numeric(data.get("volume", pd.Series(np.nan, index=data.index)), errors="coerce")
    dates = data["date"] if "date" in data.columns else pd.Series(data.index, index=data.index)

    def _date_str(i: int) -> str:
        value = dates.iloc[i]
        return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)

    # 1. 计算平滑曲线 EMA（默认 30 日，更平稳的趋势判断；返回字段保留 ema20 兼容前端）
    ema20 = close.ewm(span=ema_span, adjust=False).mean()
    long_ema = close.ewm(span=long_ema_span, adjust=False).mean()

    # 2. 计算 ATR (Average True Range) 用于判断波动和假性下跌
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_window).mean()

    # 3. 计算近期最高价用于突破确认
    highest_10 = close.rolling(breakout_window).max().shift(1)
    volume_ma20 = volume.rolling(20, min_periods=5).mean()

    # 初始化变量
    trailing_stop = pd.Series(np.nan, index=close.index, dtype=float)
    position_series = pd.Series(0.0, index=close.index, dtype=float)
    
    cash = float(initial_cash)
    shares = 0.0
    trades = []
    equity_curve = []
    
    in_position = False
    stop_price = 0.0
    entry_idx = None
    entry_price = None
    last_exit_idx = -10**9
    pending_exit_reason = None
    pending_exit_days = 0
    
    buy_signals = [None] * len(close)
    sell_signals = [None] * len(close)
    
    # 将 start_date 和 end_date 转换为可比较的格式
    start_idx = 30
    end_idx = len(close)
    
    if start_date or end_date:
        for i in range(len(dates)):
            date_str = _date_str(i)
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
        ret20 = close.iloc[i] / close.iloc[i - 20] - 1 if i >= 20 and close.iloc[i - 20] else 0.0
        
        # 趋势向上的判断：EMA明显上翘 且 价格在 EMA 之上，并且突破近期高点确认启动。
        # 同时限制价格相对 EMA 的乖离，避免在短期急拉后追高入场。
        ema_slope = (cur_ema - prev3_ema) / prev3_ema
        atr_pct = cur_atr / cur_price if cur_price > 0 else 0.02
        entry_extension = (cur_price - cur_ema) / cur_ema if cur_ema > 0 else 0.0
        allowed_extension = min(max_entry_extension, max(0.04, 2.5 * atr_pct))
        volume_ok = True
        if volume_confirm and pd.notna(volume.iloc[i]) and pd.notna(volume_ma20.iloc[i]):
            volume_ok = float(volume.iloc[i]) >= float(volume_ma20.iloc[i]) * 1.05
        volatility_ok = max_atr_pct is None or atr_pct <= max_atr_pct
        long_trend_ok = True
        if use_long_trend_filter and i >= long_ema_span:
            long_trend_ok = (cur_price > long_ema.iloc[i]) and (long_ema.iloc[i] >= long_ema.iloc[i - 5])
        trend_up = (
            (ema_slope > 0.002)
            and (cur_price > cur_ema)
            and (cur_price >= cur_h10)
            and (entry_extension <= allowed_extension)
            and volume_ok
            and volatility_ok
            and long_trend_ok
        )
        # 基金更常见的是“沿 EMA 缓慢上行”，未必天天创阶段新高。
        # 因此基金额外允许慢趋势入场，避免因只等突破而错过大段缓慢上涨。
        fund_slow_trend = is_fund and (
            (ema_slope > 0.0005)
            and (cur_price > cur_ema)
            and (ret20 > 0.01)
            and (entry_extension <= allowed_extension)
        )
        trend_up = trend_up or fund_slow_trend
        if signal_confirm_days > 1 and trend_up:
            confirm_start = max(start_idx, i - signal_confirm_days + 1)
            trend_up = all(
                close.iloc[j] > ema20.iloc[j]
                and ema20.iloc[j] >= ema20.iloc[j - 1]
                for j in range(confirm_start, i + 1)
            )
        
        if not in_position:
            # 买入逻辑：捕获上升趋势的中段
            if trend_up and i - last_exit_idx >= cooldown_days:
                in_position = True
                # 初始止损线设在当前价格下方 N 倍 ATR 处，容忍正常的回调震荡
                stop_distance = max(stop_atr_multiplier * cur_atr, cur_price * min_stop_loss_pct)
                stop_price = cur_price - stop_distance
                # 建议仓位：按单笔风险预算动态调整，波动越大仓位越低，避免固定 90% 过度暴露。
                stop_distance_pct = max((cur_price - stop_price) / cur_price, 1e-6)
                suggested_pos = _clip(risk_per_trade / stop_distance_pct, min_position, max_position)
                buy_amount = cash * suggested_pos
                buy_exec_price = cur_price * (1 + slippage_rate)
                shares = buy_amount * (1 - fee_rate) / buy_exec_price
                cash -= buy_amount
                entry_idx = i
                entry_price = cur_price
                pending_exit_reason = None
                pending_exit_days = 0
                
                date_str = _date_str(i)
                trades.append({
                    "index": i, 
                    "date": date_str, 
                    "action": "buy", 
                    "price": round(cur_price, 3), 
                    "exec_price": round(float(buy_exec_price), 3),
                    "shares": round(shares, 2),
                    "position_pct": round(float(suggested_pos * 100), 2),
                    "stop_price": round(float(stop_price), 3),
                    "entry_extension_pct": round(float(entry_extension * 100), 2),
                    "fee_rate": fee_rate,
                    "slippage_rate": slippage_rate,
                })
                buy_signals[i] = cur_price
        else:
            # 动态抬升阶梯止损线 (Trailing Stop)
            # 使用买入以来的最高价来更新止损线
            highest_since_buy = close.iloc[trades[-1]["index"]:i+1].max()
            unrealized_pct = cur_price / entry_price - 1 if entry_price else 0.0
            active_stop_multiplier = profit_stop_atr_multiplier if unrealized_pct >= profit_protect_trigger_pct else stop_atr_multiplier
            trailing_distance = max(active_stop_multiplier * cur_atr, highest_since_buy * min_stop_loss_pct)
            new_stop = highest_since_buy - trailing_distance
            if new_stop > stop_price:
                stop_price = new_stop
                
            # 卖出逻辑：跌破阶梯止损线，或 EMA 走弱且价格有效跌破 EMA 时卖出。
            # 后者用于在趋势明显转弱时更早退出，减少只等 ATR 止损导致的利润回撤。
            holding_days = i - entry_idx if entry_idx is not None else 0
            trend_break = (ema_slope < -0.001) and (cur_price < cur_ema - trend_exit_atr_buffer * cur_atr)
            hard_stop_buffer = max(hard_stop_atr_buffer * cur_atr, cur_price * min_stop_loss_pct * 0.5)
            hard_stop = cur_price < stop_price - hard_stop_buffer
            can_normal_exit = holding_days >= min_holding_days
            exit_reason = None
            normal_exit_reason = None
            if hard_stop:
                exit_reason = "hard_stop"
            elif can_normal_exit and cur_price < stop_price:
                normal_exit_reason = "trailing_stop"
            elif can_normal_exit and trend_break:
                normal_exit_reason = "trend_break"

            if normal_exit_reason:
                if pending_exit_reason == normal_exit_reason:
                    pending_exit_days += 1
                else:
                    pending_exit_reason = normal_exit_reason
                    pending_exit_days = 1
                if pending_exit_days >= sell_confirm_days:
                    exit_reason = normal_exit_reason
            elif not hard_stop:
                pending_exit_reason = None
                pending_exit_days = 0

            if exit_reason:
                in_position = False
                sell_exec_price = cur_price * (1 - slippage_rate)
                cash += shares * sell_exec_price * (1 - fee_rate)
                last_exit_idx = i
                
                date_str = _date_str(i)
                buy_price = trades[-1]["price"]
                ret_pct = (sell_exec_price / buy_price - 1) * 100
                trades.append({
                    "index": i, 
                    "date": date_str, 
                    "action": "sell", 
                    "price": round(cur_price, 3), 
                    "exec_price": round(float(sell_exec_price), 3),
                    "shares": round(shares, 2),
                    "return_pct": round(ret_pct, 2),
                    "reason": exit_reason,
                    "stop_price": round(float(stop_price), 3) if pd.notna(stop_price) else None,
                    "holding_days": holding_days,
                    "fee_rate": fee_rate,
                    "slippage_rate": slippage_rate,
                })
                shares = 0.0
                stop_price = np.nan
                entry_idx = None
                entry_price = None
                pending_exit_reason = None
                pending_exit_days = 0
                sell_signals[i] = cur_price
                
        if in_position:
            trailing_stop.iloc[i] = stop_price
            position_series.iloc[i] = shares * cur_price / (cash + shares * cur_price)
        else:
            position_series.iloc[i] = 0.0
            
        equity_curve.append(cash + shares * cur_price)

    # 补齐未计算期间的权益曲线
    full_equity = [initial_cash] * start_idx + equity_curve
    if len(full_equity) < len(close):
        full_equity.extend([full_equity[-1]] * (len(close) - len(full_equity)))
    total_return = full_equity[-1] / initial_cash - 1

    return {
        "params": {
            "asset_type": asset_type,
            "ema_span": ema_span,
            "atr_window": atr_window,
            "breakout_window": breakout_window,
            "stop_atr_multiplier": stop_atr_multiplier,
            "risk_per_trade": risk_per_trade,
            "min_position": min_position,
            "max_position": max_position,
            "max_entry_extension": max_entry_extension,
            "min_holding_days": min_holding_days,
            "cooldown_days": cooldown_days,
            "signal_confirm_days": signal_confirm_days,
            "sell_confirm_days": sell_confirm_days,
            "trend_exit_atr_buffer": trend_exit_atr_buffer,
            "hard_stop_atr_buffer": hard_stop_atr_buffer,
            "min_stop_loss_pct": min_stop_loss_pct,
            "long_ema_span": long_ema_span,
            "use_long_trend_filter": use_long_trend_filter,
            "max_atr_pct": max_atr_pct,
            "profit_protect_trigger_pct": profit_protect_trigger_pct,
            "profit_stop_atr_multiplier": profit_stop_atr_multiplier,
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "volume_confirm": volume_confirm,
        },
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


def summarize_strategy_result(strategy_result: dict, prices, initial_cash: float = 100000.0) -> dict:
    close = pd.Series(prices, dtype=float).dropna().reset_index(drop=True)
    if strategy_result is None or len(close) < 2:
        return {}

    equity = pd.Series(strategy_result.get("equity_curve") or [], dtype=float).dropna().reset_index(drop=True)
    if equity.empty:
        final_equity = float(initial_cash)
        total_return = 0.0
        max_drawdown = 0.0
        annualized_return = None
    else:
        final_equity = float(equity.iloc[-1])
        total_return = final_equity / initial_cash - 1
        running_max = equity.cummax().replace(0, np.nan)
        max_drawdown = ((equity - running_max) / running_max).min()
        annualized_return = None
        if len(equity) > 20 and final_equity > 0:
            annualized_return = (final_equity / initial_cash) ** (252 / len(equity)) - 1

    buy_hold = close.iloc[-1] / close.iloc[0] - 1
    trades = strategy_result.get("trades") or []
    sells = [t for t in trades if t.get("action") == "sell"]
    closed_returns = [float(t.get("return_pct", 0)) for t in sells if t.get("return_pct") is not None]
    holding_days = [float(t.get("holding_days", 0)) for t in sells if t.get("holding_days") is not None]

    return {
        "initial_cash": initial_cash,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(float(total_return * 100), 2),
        "buy_hold_return_pct": round(float(buy_hold * 100), 2),
        "excess_return_pct": round(float((total_return - buy_hold) * 100), 2),
        "annualized_return_pct": round(float(annualized_return * 100), 2) if annualized_return is not None else None,
        "max_drawdown_pct": round(float(max_drawdown * 100), 2),
        "trade_count": len(trades),
        "closed_trade_count": len(sells),
        "win_rate_pct": round(sum(1 for x in closed_returns if x > 0) / len(closed_returns) * 100, 2) if closed_returns else None,
        "avg_trade_return_pct": round(float(np.mean(closed_returns)), 2) if closed_returns else None,
        "avg_holding_days": round(float(np.mean(holding_days)), 1) if holding_days else None,
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
