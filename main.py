"""
量化分析终端 - FastAPI 主程序
==============================
启动:
    pip install akshare fastapi uvicorn pandas numpy scipy
    pip install tushare akshare
    pip install pyarrow fastparquet
    python main.py                         # 仅 AkShare
    TUSHARE_TOKEN=xxx python main.py       # AkShare + Tushare 双源

接口文档: http://localhost:8000/docs
"""

import logging
import os
import warnings
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

import json
import re

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("main")

# ── 内部模块 ─────────────────────────────────────────────────────
#from core.datasource import DataUnavailableError
from core.datasource.datasource import DataSource, DataUnavailableError
from core.factors import (
    calc_technical, calc_trend_channel, calc_drawdown, calc_returns,
    factor_interest_free_liab, factor_gross_margin_stability,
    factor_rd_capitalization, factor_dividend_payout, factor_institution_holdings,
    calc_pe_pb_percentile, calc_industry_percentile, multi_factor_score,
    DEFAULT_WEIGHTS, sf,
)
from core.datasource.tools import sanitize_dataframe, clean_for_json
from core.strategy import generate_trade_advice, calc_trading_ranges, backtest_strategy

# ═══════════════════════════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="量化分析终端 API",
    description="AkShare + Tushare 双数据源，5大量化因子 + 行业中性化",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# 尝试挂载前端静态文件（frontend.html 同目录）
try:
    app.mount("/static", StaticFiles(directory="."), name="static")
except Exception:
    pass

# 全局数据源（单例）
_ds: Optional[DataSource] = None
AMBIGUOUS_FUND_CODES = {"002610"}

def get_ds() -> DataSource:
    global _ds
    if _ds is None:
        ts_token = os.getenv("TUSHARE_TOKEN", "")
        _ds = DataSource(ts_token=ts_token or None)
    return _ds


# ── 季度日期工具 ──────────────────────────────────────────────

def last_quarter_end(n: int = 0) -> str:
    """第 n 个之前的季度末日期 YYYYMMDD"""
    now = datetime.now()
    q_ends = []
    for y in range(now.year - 2, now.year + 1):
        for m, d in [(3, 31), (6, 30), (9, 30), (12, 31)]:
            q_ends.append(datetime(y, m, d))
    q_ends = [q for q in q_ends if q < now]
    q_ends.sort(reverse=True)
    target = q_ends[n] if n < len(q_ends) else q_ends[-1]
    return target.strftime("%Y%m%d")


# ═══════════════════════════════════════════════════════════════
# 通用辅助
# ═══════════════════════════════════════════════════════════════
class SafeJSONResponse(JSONResponse):
    def render(self, content):
        return json.dumps(
            clean_for_json(content),
            ensure_ascii=False,
            allow_nan=False  # 🔥关键
        ).encode("utf-8")

def _years_to_dates(years: int):
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    return start, end


def _build_fin_factors(ds: DataSource, code: str) -> dict:
    """
    拉取并计算 5 大量化因子，返回合并字典
    每个子因子失败不会中断整体（catch-all）
    """
    result = {}

    # ① 基础财务指标（ROE / 毛利率等）
    try:
        fin_df = ds.stock_financial_indicator(code)
        gm_factor = factor_gross_margin_stability(fin_df)
        result["gross_margin"] = gm_factor
        result["gross_margin_score"] = gm_factor.get("score", 50)

        # ROE
        for col in ["净资产收益率(%)", "roe", "ROE"]:
            if col in fin_df.columns:
                vals = pd.to_numeric(fin_df[col], errors="coerce").dropna()
                if len(vals):
                    result["roe"] = sf(vals.iloc[0])
                break

        # EPS
        for col in ["基本每股收益(元)", "eps", "EPS"]:
            if col in fin_df.columns:
                vals = pd.to_numeric(fin_df[col], errors="coerce").dropna()
                if len(vals):
                    result["eps"] = sf(vals.iloc[0])
                break
    except Exception as e:
        result["_err_fin"] = str(e)

    # ② 无息负债（资产负债表）
    try:
        bs_df = ds.stock_balance_sheet(code)
        ifl = factor_interest_free_liab(bs_df)
        result["interest_free_liab"] = ifl
        result["interest_free_liab_score"] = ifl.get("score", 50)
    except Exception as e:
        result["_err_bs"] = str(e)

    # ③ R&D 资本化
    try:
        inc_df = ds.stock_income_sheet(code)
        bs_df2 = result.get("_bs_df") or ds.stock_balance_sheet(code)
        rd = factor_rd_capitalization(inc_df, bs_df2)
        result["rd_capitalization"] = rd
        result["rd_cap_score"] = rd.get("score", 50)

        # FCF
        cf_df = ds.stock_cashflow_sheet(code)
        if not cf_df.empty:
            cf_row = cf_df.iloc[0]
            for opc in ["n_cashflow_act", "NETCASH_OPERATE"]:
                if opc in cf_row:
                    result["operating_cashflow"] = sf(cf_row[opc])
                    break
            for cpc in ["c_pay_acq_const_fiolta", "PURCHASE_FIXED_ASSETS"]:
                if cpc in cf_row:
                    result["capex"] = sf(cf_row[cpc])
                    break
            opc = result.get("operating_cashflow", 0) or 0
            cap = result.get("capex", 0) or 0
            result["fcf"] = opc - cap
    except Exception as e:
        result["_err_cf"] = str(e)

    # ④ 股息支付率
    try:
        div_df = ds.stock_dividend(code)
        fin_df2 = ds.stock_financial_indicator(code)
        div = factor_dividend_payout(div_df, fin_df2)
        result["dividend"] = div
        result["dividend_score"] = div.get("score", 50)
    except Exception as e:
        result["_err_div"] = str(e)

    # ⑤ 机构持仓（多季度）
    try:
        inst_hist = []
        for i in range(6):
            qdate = last_quarter_end(i)
            try:
                df = ds.stock_fund_holdings(code, qdate)
                if not df.empty:
                    cnt = int(df.shape[0])
                    for col in ["amount", "持股数量", "mkv"]:
                        if col in df.columns:
                            total = float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
                            inst_hist.append({"date": qdate, "fund_count": cnt, "total_shares": total})
                            break
            except Exception:
                pass
        inst = factor_institution_holdings(inst_hist)
        result["institution"] = inst
    except Exception as e:
        result["_err_inst"] = str(e)

    return result


def _looks_like_a_stock_code(code: str) -> bool:
    """识别当前系统支持的 A 股常见代码段，避免把基金误按股票接口拉取。"""
    s = str(code or "").strip()
    return bool(
        re.fullmatch(r"(?:60|68)\d{4}", s)
        or re.fullmatch(r"(?:000|001|002|003|300|301)\d{3}", s)
    )


def _looks_like_fund_code(code: str) -> bool:
    """常见基金/ETF/LOF/联接基金代码段；007301 属于这里。"""
    s = str(code or "").strip()
    if s in AMBIGUOUS_FUND_CODES:
        return True
    return bool(re.fullmatch(r"(?:00[4-9]\d{3}|(?:01|02|04|05|07|08|09|15|16|18|50|51|52)\d{4})", s))


# ═══════════════════════════════════════════════════════════════
# API 路由
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    ds = get_ds()
    return {
        "status": "running",
        "version": "3.0.0",
        "datasource": ds.status,
        "docs": "/docs",
        "source_log": ds.source_log(),
    }


# ─── 搜索 ─────────────────────────────────────────────────────
def _get_display_name(r: dict) -> str:
    """尝试从各种可能的列里拿到名称"""
    for k in ["name", "名称", "基金简称"]:
        if k in r and r[k]:
            return str(r[k])
    return str(r.get("code", r.get("基金代码", "")))

@app.get("/api/search")
def search(q: str = Query(..., description="股票代码或名称"), type: str = Query("both")):
    """搜索股票 / 基金，返回匹配列表"""
    ds = get_ds()
    results = []

    if type in ("stock", "both"):
        try:
            df = ds.search_stock(q)
            for _, r in df.head(8).iterrows():
                results.append({
                    "code": str(r.get("code", r.get("symbol", ""))),
                    "name": _get_display_name(r),
                    "type": "stock",
                })
        except Exception:
            pass

    if type in ("fund", "both"):
        try:
            df = ds.search_fund(q)
            for _, r in df.head(8).iterrows():
                results.append({
                    "code": str(r.get("基金代码", r.get("code", ""))),
                    "name": _get_display_name(r),
                    "type": "fund",
                })
        except Exception:
            pass

    return {"query": q, "results": results}


@app.get("/api/market/price/{asset_type}/{code}")
def market_price(asset_type: str, code: str):
    """统一价格抓取：股票实时快照，基金最新净值，失败时走缓存/历史兜底。"""
    if asset_type not in ("stock", "fund"):
        raise HTTPException(400, "asset_type 仅支持 stock 或 fund")
    ds = get_ds()
    try:
        return clean_for_json(ds.market_price(code, asset_type))
    except DataUnavailableError as e:
        raise HTTPException(400, str(e))


# ─── 基金接口 ─────────────────────────────────────────────────

@app.get("/api/fund/{code}", response_class=SafeJSONResponse)
def get_fund(
    code: str,
    period: str = Query("成立来", description="1月|3月|6月|1年|3年|5年|成立来"),
):
    """
    基金全量分析
    返回：净值历史 + 趋势通道 + 技术指标 + 回撤 + 收益率统计
    """
    ds = get_ds()

    # 🔹 尝试通过 search_fund 获取基金名称
    try:
        df_name = ds.search_fund(code)
        if not df_name.empty:
            fund_name = df_name.iloc[0].get("基金简称") or df_name.iloc[0].get("name") or code
        else:
            fund_name = code
    except Exception:
        fund_name = code

    try:
        nav_df = ds.fund_nav(code, period)
    except DataUnavailableError as e:
        raise HTTPException(400, str(e))

    prices = nav_df["nav"].tolist()
    dates  = nav_df["date"].dt.strftime("%Y-%m-%d").tolist()

    channel = calc_trend_channel(prices)
    dd      = calc_drawdown(prices)
    returns = calc_returns(prices, dates)
    tech    = calc_technical(nav_df["nav"]) if len(prices) >= 26 else {}
    strategy = generate_trade_advice("fund", prices, dates, tech, dd, {}, {})
    trading_ranges = calc_trading_ranges(prices, dates)

    # 信号
    cur = prices[-1]
    mid = channel["current_mid"] or cur
    lo  = channel["current_lower"] or cur * 0.8
    hi  = channel["current_upper"] or cur * 1.2
    dev = channel["deviation_pct"]

    if cur < lo:
        signal, pos = "buy",  75
        reason = f"净值({cur:.4f})低于趋势下轨({lo:.4f})，超卖区，可分批建仓"
    elif cur > hi:
        signal, pos = "sell", 25
        reason = f"净值({cur:.4f})突破趋势上轨({hi:.4f})，超买，考虑止盈"
    elif dev < -8:
        signal, pos = "hold", 65
        reason = f"净值在中轨下方{abs(dev):.1f}%，趋势偏弱，观望为主"
    elif dev > 12:
        signal, pos = "hold", 50
        reason = f"净值在中轨上方{dev:.1f}%，趋势向好，维持仓位"
    else:
        signal, pos = "hold", 60
        reason = "净值位于趋势通道合理区间"

    return clean_for_json({
        "code": code, "type": "fund",
        "name": fund_name,
        "dates": dates,
        "nav":   prices,
        "daily_return": nav_df.get("daily_return", pd.Series(dtype=float)).fillna(0).tolist(),
        "channel": channel,
        "drawdown": dd,
        "returns": returns,
        "technical": tech,
        "signal": signal,
        "signal_reason": reason,
        "suggested_position": pos,
        "strategy_advice": strategy,
        "trading_ranges": trading_ranges,
        "latest": {
            "nav": cur, "date": dates[-1],
            "channel_deviation_pct": dev,
            "channel_position": channel["position_label"],
        },
    })


# ─── 股票接口 ─────────────────────────────────────────────────
def _get_stock_name(info: dict, code: str) -> str:
    for k in ["name", "名称"]:
        if k in info and info[k]:
            return info[k]
    return code

@app.get("/api/stock/{code}", response_class=SafeJSONResponse)
def get_stock(
    code: str,
    years: int = Query(3, ge=1, le=10, description="历史年数"),
    include_financial: bool = Query(True, description="是否包含财务因子（耗时较长）"),
    weights: Optional[str] = Query(None, description="JSON格式权重覆盖，如 '{\"valuation\":0.3}'"),
):
    """
    股票全量分析
    返回：行情 + 技术 + 5大因子 + PE/PB百分位 + 行业分位 + 多因子评分
    """
    ds = get_ds()
    if _looks_like_fund_code(code) and (not _looks_like_a_stock_code(code) or code in AMBIGUOUS_FUND_CODES):
        raise HTTPException(400, f"{code} 看起来是基金代码，请使用基金类型添加，或请求 /api/fund/{code}")

    start, end = _years_to_dates(years)

    # ── 行情 ──
    try:
        hist_df = ds.stock_hist(code, start, end)
        hist_df = sanitize_dataframe(hist_df)
        if "date" not in hist_df.columns:
            raise HTTPException(500, "数据源返回缺少 date 字段")
    except DataUnavailableError as e:
        raise HTTPException(400, str(e))

    prices = hist_df["close"].tolist()
    dates  = hist_df["date"].dt.strftime("%Y-%m-%d").tolist()
    vols   = hist_df.get("volume", pd.Series(dtype=float))

    # ── 指标计算 ──
    tech    = calc_technical(hist_df["close"], vols)
    channel = calc_trend_channel(prices)
    dd      = calc_drawdown(prices)
    returns = calc_returns(prices, dates)

    # ── 实时估值 ──
    info    = ds.stock_info(code)
    pe_ttm  = info.get("pe_ttm")
    pb      = info.get("pb")

    # ── PE/PB 历史百分位 ──
    valuation = {"pe_ttm": pe_ttm, "pb": pb}
    try:
        pe_pb_df = ds.stock_pe_pb_history(code)
        val_pct  = calc_pe_pb_percentile(pe_pb_df)
        valuation.update(val_pct)
    except Exception as e:
        valuation["_err_pepb"] = str(e)

    # ── 5 大量化因子（可选）──
    fin_factors = {}
    if include_financial:
        fin_factors = _build_fin_factors(ds, code)

    # ── 权重覆盖 ──
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        import json
        try:
            override = json.loads(weights)
            w.update(override)
        except Exception:
            pass

    # ── 多因子评分 ──
    quant_score = multi_factor_score(fin_factors, valuation, tech, dd, weights=w)
    strategy = generate_trade_advice("stock", prices, dates, tech, dd, valuation, quant_score)
    trading_ranges = calc_trading_ranges(prices, dates)

    # ── 整合返回 ──
    result = {
        "code": code, "type": "stock",
        "name": _get_stock_name(info, code),
        "industry": info.get("industry", ""),
        "dates": dates,
        "close": prices,
        "open":   hist_df.get("open",   pd.Series(dtype=float)).tolist(),
        "high":   hist_df.get("high",   pd.Series(dtype=float)).tolist(),
        "low":    hist_df.get("low",    pd.Series(dtype=float)).tolist(),
        "volume": vols.tolist(),
        "channel": channel,
        "drawdown": dd,
        "returns": returns,
        "technical": tech,
        "valuation": valuation,
        "financial_factors": fin_factors,
        "quant_score": quant_score,
        "strategy_advice": strategy,
        "trading_ranges": trading_ranges,
        "latest": {
            "close": prices[-1], "date": dates[-1],
            "pe_ttm": pe_ttm, "pb": pb,
            "market_cap": info.get("market_cap"),
            "channel_position": channel["position_label"],
            "channel_deviation_pct": channel["deviation_pct"],
        },
        "datasource_log": ds.source_log(),
    }

    return clean_for_json(result)


# ─── 行业中性化 ───────────────────────────────────────────────

@app.get("/api/industry/compare")
def industry_compare(
    codes: str = Query(..., description="逗号分隔的股票代码，如 600519,000858,002304"),
    metric: str = Query("roe", description="对比指标: roe | gross_margin | pe_ttm | pb"),
    higher_is_better: bool = Query(True, description="越高越好（ROE/毛利率=True，PE=False）"),
):
    """
    行业中性化对比
    给一组股票，计算每只在组内的百分位，标注 Top 20% / Bottom 20%
    """
    ds   = get_ds()
    clist = [c.strip() for c in codes.split(",") if c.strip()]

    data = []
    for code in clist:
        val = None
        try:
            if metric in ("pe_ttm", "pb"):
                info = ds.stock_info(code)
                val  = info.get(metric)
            else:
                fin  = ds.stock_financial_indicator(code)
                col_map = {
                    "roe":          ["净资产收益率(%)", "roe"],
                    "gross_margin": ["销售毛利率(%)", "grossprofit_margin"],
                }
                for col in col_map.get(metric, [metric]):
                    if col in fin.columns:
                        vals = pd.to_numeric(fin[col], errors="coerce").dropna()
                        if len(vals):
                            val = sf(vals.iloc[0])
                        break
        except Exception:
            pass
        data.append({"code": code, "value": val})

    # 有效值列表
    values = [d["value"] for d in data if d["value"] is not None]

    results = []
    for d in data:
        if d["value"] is not None:
            pct_info = calc_industry_percentile(
                d["value"], values, metric, higher_is_better
            )
            results.append({**d, **pct_info})
        else:
            results.append({**d, "percentile": None, "error": "数据获取失败"})

    results.sort(key=lambda x: x.get("percentile") or 0, reverse=True)

    return {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "group_count": len(values),
        "group_median": round(float(np.median(values)), 4) if values else None,
        "group_mean":   round(float(np.mean(values)), 4)   if values else None,
        "results": results,
        "top20pct": [r["code"] for r in results if r.get("is_top20pct")],
    }


# ─── 量化筛选 ─────────────────────────────────────────────────

@app.get("/api/screen")
def screen_stocks(
    min_roe: float = Query(12.0, description="最低 ROE (%)"),
    max_pe_pct5y: int  = Query(40,   description="PE 5年百分位上限"),
    min_ifl_ratio: float = Query(25.0, description="无息负债比例下限 (%)"),
    min_div_payout: float = Query(25.0, description="股息支付率下限 (%)"),
    max_div_payout: float = Query(75.0, description="股息支付率上限 (%)"),
    min_score: int = Query(55, description="多因子最低综合分"),
    limit: int = Query(20, description="返回数量"),
):
    """
    多因子选股接口
    基于 5 大量化指标进行粗筛，返回满足条件的标的
    注意：全量筛选耗时，建议先用 /api/stock/{code} 对目标标的精细分析
    """
    ds = get_ds()
    try:
        ak = ds.ak
        if not ak.available:
            raise HTTPException(503, "AkShare 未初始化，请检查安装")
        df_spot = ak.ak.stock_zh_a_spot_em()
    except Exception as e:
        raise HTTPException(500, f"获取股票列表失败: {e}")

    # 基础过滤
    df_spot = df_spot[~df_spot["名称"].str.contains("ST|退|*", na=False, regex=False)]
    pe_col = "市盈率-动态"
    if pe_col in df_spot.columns:
        df_spot = df_spot[pd.to_numeric(df_spot[pe_col], errors="coerce") > 0]

    results = []
    sampled = df_spot.head(limit * 5)  # 限量避免超时

    for _, row in sampled.iterrows():
        code = str(row.get("代码", ""))
        name = str(row.get("名称", ""))
        pe   = sf(row.get("市盈率-动态"))
        pb   = sf(row.get("市净率"))
        price = sf(row.get("最新价"))

        if pe and pe < 200:
            results.append({
                "code": code, "name": name,
                "price": price, "pe": pe, "pb": pb,
                "note": f"调用 /api/stock/{code} 获取完整量化分析",
            })
        if len(results) >= limit:
            break

    return {
        "total": len(results),
        "filter_summary": {
            "min_roe": min_roe,
            "max_pe_pct5y": max_pe_pct5y,
            "min_ifl_ratio": min_ifl_ratio,
            "dividend_payout_range": f"{min_div_payout}% ~ {max_div_payout}%",
            "min_score": min_score,
        },
        "usage_tip": "此接口返回候选列表，精确因子分析请调用 /api/stock/{code}",
        "results": results,
    }


@app.get("/api/strategy/backtest/{asset_type}/{code}")
def strategy_backtest(
    asset_type: str,
    code: str,
    years: int = Query(3, ge=1, le=10, description="回测历史年数"),
):
    """对股票或基金运行内置趋势动量策略回测。"""
    if asset_type not in ("stock", "fund"):
        raise HTTPException(400, "asset_type 仅支持 stock 或 fund")

    ds = get_ds()
    try:
        if asset_type == "fund":
            df = ds.fund_nav(code).rename(columns={"nav": "close"})
        else:
            start, end = _years_to_dates(years)
            df = ds.stock_hist(code, start, end)
        result = backtest_strategy(df)
    except (DataUnavailableError, ValueError) as e:
        raise HTTPException(400, str(e))

    return clean_for_json({
        "code": code,
        "type": asset_type,
        "years": years,
        "strategy": "ma20_ma60_momentum",
        "backtest": result,
    })



# ─── 缓存管理 ────────────────────────────────────────────────

@app.post("/api/cache/clear")
def clear_cache(older_than_hours: int = Query(0, description="清除N小时前的缓存，0=全部清除")):
    """
    清除旧缓存
    当数据显示截止日期不正确时（如停留在2025年底），
    请调用此接口或删除 ~/.quant_cache/ 目录后重试。
    """
    ds = get_ds()
    n = ds.clear_cache(older_than_hours)
    return {"cleared": n, "message": f"已清除 {n} 个缓存文件，请重新请求数据"}

# ─── 数据源状态 ───────────────────────────────────────────────

@app.get("/api/datasource/status")
def datasource_status():
    """查看两个数据源的可用状态和 fallback 记录"""
    ds = get_ds()
    return {
        "akshare":    ds.ak.available,
        "tushare":    ds.ts.available,
        "primary":    "akshare" if ds.ak.available else "tushare",
        "source_log": ds.source_log(),
        "ts_token_set": bool(os.getenv("TUSHARE_TOKEN", "")),
    }


@app.post("/api/datasource/set_token")
def set_tushare_token(token: str = Query(...)):
    """运行时设置 Tushare Token（无需重启）"""
    global _ds
    os.environ["TUSHARE_TOKEN"] = token
    _ds = None  # 强制重新初始化
    get_ds()
    return {"status": "ok", "tushare_available": get_ds().ts.available}


# ═══════════════════════════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 58)
    print("  量化分析终端 v3.0  FastAPI + AkShare/Tushare")
    print("─" * 58)
    print(f"  API 文档  :  http://localhost:8000/docs")
    print(f"  前端页面  :  直接打开 frontend.html")
    print(f"  Tushare   :  {'已配置' if os.getenv('TUSHARE_TOKEN') else '未配置（设置 TUSHARE_TOKEN 环境变量）'}")
    print("═" * 58 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
