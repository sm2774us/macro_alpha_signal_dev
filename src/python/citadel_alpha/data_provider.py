# citadel_alpha/data_provider.py — Design-by-Contract data provider abstraction.
# Google Python Style Guide.

"""Plugin-architecture data provider for ISCF and MGD signals.

Design-by-Contract (DbC) principle: the abstract base classes define the
interface contract that any provider must satisfy. Free-tier providers
(yfinance, FRED via pandas-datareader) are used now. When Proprietary Trading Firm
infrastructure is available, drop in HLS_ISCFProvider / HLS_MGDProvider
implementing the same ABCs without changing any signal or backtest code.

Usage:
    # Free tier (default)
    from citadel_alpha.data_provider import YFinanceISCFProvider
    provider = YFinanceISCFProvider()
    data = provider.fetch(start="2015-01-01", end="2024-12-31")

    # Proprietary Trading Firm — infra (plug-in replacement — same interface)
    from citadel_alpha.data_provider import HLSISCFProvider  # Future
    provider = HLSISCFProvider(api_key=os.environ["HLS_API_KEY"])
    data = provider.fetch(start="2015-01-01", end="2024-12-31")
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Data contracts (frozen dataclasses — immutable after construction)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ISCFMarketData:
    """Contract: data required by the ISCF signal computation.

    Any data provider that implements AbstractISCFProvider MUST return
    an instance of this dataclass with all fields populated.

    Field invariants:
        - All arrays shape (T, N), float64, no NaN after forward-fill.
        - spot[t, i] > 0 for all t, i.
        - rvol[t, i] > 0 for all t, i.
        - macro_beta[t, i] ∈ [0, 1].
        - assets: list of N ticker strings.
        - dates: pd.DatetimeIndex of length T.
    """

    spot: FloatArray  # Front-month futures price (T, N)
    deferred: FloatArray  # Deferred contract price (T, N)
    rvol: FloatArray  # Annualised realised vol 20d (T, N)
    macro_beta: FloatArray  # Market-beta to macro index (T, N)
    forward_returns: FloatArray  # Next-period returns (T, N)
    trend_baseline: FloatArray  # 12-1 momentum baseline (T, N)
    momentum_baseline: FloatArray  # 1-month momentum baseline (T, N)
    carry_baseline: FloatArray  # Roll yield carry baseline (T, N)
    assets: tuple[str, ...]
    dates: pd.DatetimeIndex


@dataclass(frozen=True)
class MGDMarketData:
    """Contract: data required by the MGD signal computation.

    Field invariants:
        - All arrays shape (T, N), float64.
        - pmi_surprise / cpi_surprise / emp_surprise: z-scored vs consensus.
        - fwd_expectation: EMA of past composite surprise.
        - roll_std[t, i] > 0.
    """

    pmi_surprise: FloatArray
    cpi_surprise: FloatArray
    emp_surprise: FloatArray
    fwd_expectation: FloatArray
    roll_std: FloatArray
    forward_returns: FloatArray
    trend_baseline: FloatArray
    momentum_baseline: FloatArray
    carry_baseline: FloatArray
    assets: tuple[str, ...]
    dates: pd.DatetimeIndex


# ---------------------------------------------------------------------------
# Abstract base classes (Design-by-Contract interfaces)
# ---------------------------------------------------------------------------


class AbstractISCFProvider(abc.ABC):
    """Abstract contract for ISCF commodity data providers.

    Concrete implementations:
        YFinanceISCFProvider  — free tier (yfinance + pandas-datareader)
        AlphaVantageISCFProvider — free tier (Alpha Vantage commodities)
        HLSISCFProvider       — Proprietary Trading Firm (LME/CME feeds, freight)
    """

    @abc.abstractmethod
    def fetch(
        self,
        start: str,
        end: str,
        assets: Optional[list[str]] = None,
    ) -> ISCFMarketData:
        """Fetch and return a fully populated ISCFMarketData.

        Preconditions:
            start < end (chronological order).
            All returned arrays must have identical first dimension T.

        Postconditions:
            No NaN in any array field.
            spot > 0 elementwise.
            rvol > 0 elementwise.
            macro_beta ∈ [0, 1] elementwise.
        """
        ...

    def validate(self, data: ISCFMarketData) -> None:
        """Assert contract postconditions (called by all implementations)."""
        assert np.all(np.isfinite(data.spot)), "spot contains non-finite values"
        assert np.all(data.spot > 0), "spot must be positive"
        assert np.all(data.rvol > 0), "rvol must be positive"
        assert np.all(
            (data.macro_beta >= 0) & (data.macro_beta <= 1)
        ), "macro_beta must be in [0, 1]"
        t = data.spot.shape[0]
        for name, arr in [
            ("deferred", data.deferred),
            ("rvol", data.rvol),
            ("macro_beta", data.macro_beta),
            ("forward_returns", data.forward_returns),
        ]:
            assert arr.shape[0] == t, f"{name} T-dimension mismatch"


class AbstractMGDProvider(abc.ABC):
    """Abstract contract for MGD FX/macro data providers.

    Concrete implementations:
        YFinanceMGDProvider   — free tier (yfinance FX + FRED macro)
        FREDMGDProvider       — free tier (pandas-datareader FRED)
        HLSMGDProvider        — Proprietary Trading Firm (flash PMI, CB feeds)
    """

    @abc.abstractmethod
    def fetch(
        self,
        start: str,
        end: str,
        assets: Optional[list[str]] = None,
    ) -> MGDMarketData:
        """Fetch and return a fully populated MGDMarketData."""
        ...

    def validate(self, data: MGDMarketData) -> None:
        """Assert contract postconditions."""
        assert np.all(
            np.isfinite(data.forward_returns)
        ), "forward_returns contains non-finite values"
        assert np.all(data.roll_std > 0), "roll_std must be positive"
        t = data.pmi_surprise.shape[0]
        for name, arr in [
            ("cpi_surprise", data.cpi_surprise),
            ("emp_surprise", data.emp_surprise),
            ("fwd_expectation", data.fwd_expectation),
            ("roll_std", data.roll_std),
        ]:
            assert arr.shape[0] == t, f"{name} T-dimension mismatch"


# ---------------------------------------------------------------------------
# Free-tier implementation: yfinance + FRED
# ---------------------------------------------------------------------------

# Commodity ETF proxies (free via yfinance) for ISCF
# Real implementation would use CME/LME front-month and deferred contracts.
ISCF_PROXY_TICKERS = {
    "WTI_CL": ("USO", "BNO"),  # WTI crude: USO (front) vs BNO (deferred proxy)
    "BRENT_CO": ("BNO", "USO"),
    "NGAS_NG": ("UNG", "BOIL"),  # Natural gas proxies
    "COPPER_HG": ("CPER", "JJC"),
    "GOLD_GC": ("GLD", "IAU"),
    "SILVER_SI": ("SLV", "SIVR"),
    "ALUMINIUM_LA": ("JJU", "DJCI"),  # Broad commodity as deferred proxy
    "ZINC_LX": ("PICK", "DJCI"),  # FIXED: Replaced delisted 'ZINC' with 'PICK'
}
# ISCF_PROXY_TICKERS = {
#     "WTI_CL":      ("USO",  "BNO"),   # WTI crude: USO (front) vs BNO (deferred proxy)
#     "BRENT_CO":    ("BNO",  "USO"),
#     "NGAS_NG":     ("UNG",  "BOIL"),  # Natural gas proxies
#     "COPPER_HG":   ("CPER", "JJC"),
#     "GOLD_GC":     ("GLD",  "IAU"),
#     "SILVER_SI":   ("SLV",  "SIVR"),
#     "ALUMINIUM_LA":("JJU",  "DJCI"),  # Broad commodity as deferred proxy
#     "ZINC_LX":     ("ZINC", "DJCI"),
# }

# FX pairs available free via yfinance
MGD_FX_TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NOKUSD": "NOKUSD=X",
    "SEKUSD": "SEKUSD=X",
}

# FRED series for macro surprises (PMI proxy: ISM, CPI, NFP)
FRED_SERIES = {
    "ISM_MFG": "MANEMP",  # Manufacturing employment (ISM proxy)
    "CPI_YOY": "CPIAUCSL",
    "NFP": "PAYEMS",
    "FED_RATE": "FEDFUNDS",
}


class YFinanceISCFProvider(AbstractISCFProvider):
    """Free-tier ISCF provider using yfinance ETF price proxies.

    Proxy methodology:
        spot[i]     = ETF_front[i] adjusted close (proxy for front-month)
        deferred[i] = ETF_deferred[i] adjusted close (proxy for deferred)
        basis       = (spot - deferred) / spot  (% basis)
        rvol        = 20-day rolling annualised realised vol of spot returns
        macro_beta  = 60-day rolling beta of spot returns to SPY

    Limitation vs. HLS infra:
        ETF prices embed management fees and do not precisely track
        CME/LME prompt-date spreads. The basis signal is directionally
        correct but magnitude-attenuated vs. physical futures data.
        Plug in HLSISCFProvider for production-grade precision.
    """

    def fetch(
        self,
        start: str = "2015-01-01",
        end: str = "2024-12-31",
        assets: Optional[list[str]] = None,
    ) -> ISCFMarketData:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("pip install yfinance") from exc

        if assets is None:
            assets = list(ISCF_PROXY_TICKERS.keys())

        logger.info(
            "YFinanceISCFProvider: fetching %d assets %s→%s", len(assets), start, end
        )

        all_front_tickers = [
            ISCF_PROXY_TICKERS[a][0] for a in assets if a in ISCF_PROXY_TICKERS
        ]
        all_defer_tickers = [
            ISCF_PROXY_TICKERS[a][1] for a in assets if a in ISCF_PROXY_TICKERS
        ]
        spy = yf.download(
            "SPY", start=start, end=end, auto_adjust=True, progress=False
        )["Close"]

        front_data = yf.download(
            list(set(all_front_tickers)),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )["Close"]
        defer_data = yf.download(
            list(set(all_defer_tickers)),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )["Close"]

        if isinstance(front_data, pd.Series):
            front_data = front_data.to_frame(name=all_front_tickers[0])
        if isinstance(defer_data, pd.Series):
            defer_data = defer_data.to_frame(name=all_defer_tickers[0])

        # Align all to common dates
        common_idx = front_data.index.intersection(defer_data.index).intersection(
            spy.index
        )
        front_data = front_data.reindex(common_idx).ffill().bfill()
        defer_data = defer_data.reindex(common_idx).ffill().bfill()
        spy_aligned = spy.reindex(common_idx).ffill().bfill()

        n = len(assets)
        t = len(common_idx)

        spot = np.zeros((t, n))
        deferred = np.zeros((t, n))
        rvol = np.zeros((t, n))
        macro_beta = np.ones((t, n)) * 0.3
        fwd_ret = np.zeros((t, n))
        trend_bl = np.zeros((t, n))
        mom_bl = np.zeros((t, n))
        carry_bl = np.zeros((t, n))

        # spy_ret = spy_aligned.pct_change().fillna(0).values
        spy_ret = (
            spy_aligned.pct_change().fillna(0).values.flatten()
        )  # added .flatten()

        for j, asset in enumerate(assets):
            if asset not in ISCF_PROXY_TICKERS:
                continue
            ft, dt = ISCF_PROXY_TICKERS[asset]

            s_col = front_data.get(ft, front_data.iloc[:, 0]).values.astype(float)
            d_col = defer_data.get(dt, defer_data.iloc[:, 0]).values.astype(float)

            spot[:, j] = np.where(s_col > 0, s_col, np.nan)
            deferred[:, j] = np.where(d_col > 0, d_col, np.nan)

            # Forward fill NaNs
            for col in [spot[:, j], deferred[:, j]]:
                for i in range(1, t):
                    if np.isnan(col[i]):
                        col[i] = col[i - 1]

            ret = np.diff(np.log(np.maximum(spot[:, j], 1e-8)))
            ret = np.concatenate([[0.0], ret])
            fwd_ret[:, j] = np.roll(ret, -1)
            fwd_ret[-1, j] = 0.0

            # Realised vol: 20-day rolling annualised
            for i in range(20, t):
                rvol[i, j] = float(np.std(ret[i - 20 : i], ddof=1)) * np.sqrt(252)
            rvol[:20, j] = max(float(np.std(ret[:20], ddof=1)), 1e-4) * np.sqrt(252)

            # Macro beta: 60-day rolling OLS vs SPY
            for i in range(60, t):
                r_chunk = ret[i - 60 : i]
                s_chunk = spy_ret[i - 60 : i]
                denom = float(np.dot(s_chunk, s_chunk))
                if denom > 1e-10:
                    macro_beta[i, j] = float(
                        np.clip(np.dot(r_chunk, s_chunk) / denom, 0, 1)
                    )

            # Baselines: trend (12-1 mom), momentum (1m), carry (roll yield proxy)
            for i in range(252, t):
                trend_bl[i, j] = float(np.sum(ret[i - 252 : i - 21]))
            for i in range(21, t):
                mom_bl[i, j] = float(np.sum(ret[i - 21 : i]))
            carry_bl[:, j] = (spot[:, j] - deferred[:, j]) / np.maximum(
                spot[:, j], 1e-8
            )

        data = ISCFMarketData(
            spot=np.maximum(spot, 1e-8),
            deferred=np.maximum(deferred, 1e-8),
            rvol=np.maximum(rvol, 1e-4),
            macro_beta=np.clip(macro_beta, 0.0, 1.0),
            forward_returns=fwd_ret,
            trend_baseline=trend_bl,
            momentum_baseline=mom_bl,
            carry_baseline=carry_bl,
            assets=tuple(assets),
            dates=common_idx,
        )
        self.validate(data)
        return data


class YFinanceMGDProvider(AbstractMGDProvider):
    """Free-tier MGD provider: yfinance FX + FRED macro via pandas-datareader.

    Proxy methodology:
        PMI surprise  ≈ ISM Manufacturing index MoM delta (FRED: NAPM)
        CPI surprise  ≈ CPI MoM minus 12-month average (FRED: CPIAUCSL)
        Employment    ≈ NFP MoM vs 6-month avg (FRED: PAYEMS)
        FWD expectation = 21-day EMA of composite surprise

    Limitation vs. HLS infra:
        FRED releases lag by ~1 month; intraday flash PMIs not available.
        Consensus-based surprise (actual-consensus) requires a paid data
        feed (Bloomberg). We proxy with deviation from rolling average.
        Plug in HLSMGDProvider for true consensus-based surprises.
    """

    # def fetch(
    #     self,
    #     start: str = "2015-01-01",
    #     end: str = "2024-12-31",
    #     assets: Optional[list[str]] = None,
    # ) -> MGDMarketData:
    #     try:
    #         import yfinance as yf
    #     except ImportError as exc:
    #         raise ImportError("pip install yfinance") from exc

    #     try:
    #         import pandas_datareader.data as web
    #         _FRED_AVAILABLE = True
    #     except ImportError:
    #         _FRED_AVAILABLE = False
    #         logger.warning("pandas-datareader not available; using yfinance-only macro proxies.")

    #     if assets is None:
    #         assets = list(MGD_FX_TICKERS.keys())

    #     logger.info("YFinanceMGDProvider: fetching FX panel %s→%s", start, end)

    #     fx_tickers = [MGD_FX_TICKERS.get(a, f"{a}=X") for a in assets]
    #     fx_data = yf.download(fx_tickers, start=start, end=end,
    #                           auto_adjust=True, progress=False)["Close"]
    #     if isinstance(fx_data, pd.Series):
    #         fx_data = fx_data.to_frame(name=fx_tickers[0])
    #     fx_data = fx_data.ffill().bfill()

    #     common_idx = fx_data.index
    #     t = len(common_idx)
    #     n = len(assets)

    #     # FRED macro series (monthly, forward-filled to daily)
    #     fred_data: dict[str, pd.Series] = {}
    #     if _FRED_AVAILABLE:
    #         for key, series in FRED_SERIES.items():
    #             try:
    #                 s = web.DataReader(series, "fred", start, end).squeeze()
    #                 s = s.reindex(common_idx, method="ffill").ffill().bfill()
    #                 fred_data[key] = s
    #                 logger.info("FRED %s fetched OK", series)
    #             except Exception as e:
    #                 logger.warning("FRED %s failed: %s", series, e)

    #     def _surprise(series_key: str, window: int = 12) -> FloatArray:
    #         """MoM delta minus rolling-average = proxy for surprise."""
    #         if series_key not in fred_data:
    #             return np.zeros((t, n))
    #         s = fred_data[series_key].values.astype(float)
    #         delta = np.diff(s, prepend=s[0])
    #         rolling_avg = np.convolve(delta, np.ones(window) / window, mode="same")
    #         surprise_1d = delta - rolling_avg
    #         return np.outer(surprise_1d, np.ones(n))

    def fetch(
        self,
        start: str = "2015-01-01",
        end: str = "2024-12-31",
        assets: Optional[list[str]] = None,
    ) -> MGDMarketData:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("pip install yfinance") from exc

        # COMPLETELY BYPASS PANDAS-DATAREADER TO AVOID DEPRECATION CRASHES
        _FRED_AVAILABLE = True

        if assets is None:
            assets = list(MGD_FX_TICKERS.keys())

        logger.info("YFinanceMGDProvider: fetching FX panel %s→%s", start, end)

        fx_tickers = [MGD_FX_TICKERS.get(a, f"{a}=X") for a in assets]
        fx_data = yf.download(
            fx_tickers, start=start, end=end, auto_adjust=True, progress=False
        )["Close"]
        if isinstance(fx_data, pd.Series):
            fx_data = fx_data.to_frame(name=fx_tickers[0])
        fx_data = fx_data.ffill().bfill()

        common_idx = fx_data.index
        t = len(common_idx)
        n = len(assets)

        # FRED macro series (monthly, downloaded natively via CSV URL endpoints)
        fred_data: dict[str, pd.Series] = {}
        if _FRED_AVAILABLE:
            for key, series in FRED_SERIES.items():
                try:
                    # Construct direct URL download bypassing pandas-datareader completely
                    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

                    # Read without setting index or parse_dates yet to isolate raw text/types
                    s_df = pd.read_csv(url)

                    # Extract the first column (dates) and data column (values) dynamically by position
                    date_col_name = s_df.columns[0]
                    value_col_name = s_df.columns[1]

                    # Convert to datetime index safely using pd.Index to avoid .iloc type alignment errors
                    s_df.index = pd.Index(
                        pd.to_datetime(s_df[date_col_name], errors="coerce")
                    )

                    # Ensure data numeric value type casting (converts placeholders like '.' to NaN)
                    s_df[value_col_name] = pd.to_numeric(
                        s_df[value_col_name], errors="coerce"
                    )
                    s = s_df[value_col_name].squeeze()

                    # Reindex and forward-fill align to framework timeline
                    s = s.reindex(common_idx, method="ffill").ffill().bfill()
                    fred_data[key] = s
                    logger.info("FRED %s fetched natively OK", series)
                except Exception as e:
                    logger.warning("Natively fetching FRED %s failed: %s", series, e)

        def _surprise(series_key: str, window: int = 12) -> FloatArray:
            """MoM delta minus rolling-average = proxy for surprise."""
            if series_key not in fred_data:
                return np.zeros((t, n))
            s = fred_data[series_key].values.astype(float)
            delta = np.diff(s, prepend=s[0])
            rolling_avg = np.convolve(delta, np.ones(window) / window, mode="same")
            surprise_1d = delta - rolling_avg
            return np.outer(surprise_1d, np.ones(n))

        pmi_surp = _surprise("ISM_MFG", 12)
        cpi_surp = _surprise("CPI_YOY", 12)
        emp_surp = _surprise("NFP", 6)

        # Normalise surprises cross-sectionally
        for arr in [pmi_surp, cpi_surp, emp_surp]:
            std = np.std(arr, axis=0, ddof=1)
            std = np.where(std < 1e-8, 1.0, std)
            arr /= std

        composite = 0.40 * pmi_surp + 0.30 * cpi_surp + 0.30 * emp_surp

        # FX returns and baselines
        fwd_ret = np.zeros((t, n))
        trend_bl = np.zeros((t, n))
        mom_bl = np.zeros((t, n))
        carry_bl = np.zeros((t, n))

        for j, asset in enumerate(assets):
            ticker = MGD_FX_TICKERS.get(asset, f"{asset}=X")
            col = fx_data.get(ticker, fx_data.iloc[:, 0]).values.astype(float)
            ret = np.diff(np.log(np.maximum(col, 1e-8)))
            ret = np.concatenate([[0.0], ret])
            fwd_ret[:, j] = np.roll(ret, -1)
            fwd_ret[-1, j] = 0.0
            for i in range(252, t):
                trend_bl[i, j] = float(np.sum(ret[i - 252 : i - 21]))
            for i in range(21, t):
                mom_bl[i, j] = float(np.sum(ret[i - 21 : i]))
            # Carry proxy: interest rate differential (use FED rate as base)
            if "FED_RATE" in fred_data:
                carry_bl[:, j] = fred_data["FED_RATE"].values * 0.01
            else:
                carry_bl[:, j] = 0.025  # Flat 2.5% proxy

        # Forward expectation: 21-day EMA of composite
        alpha_fwd = 2.0 / (21 + 1.0)
        fwd_exp = np.zeros((t, n))
        fwd_exp[0] = composite[0]
        for i in range(1, t):
            fwd_exp[i] = alpha_fwd * composite[i] + (1.0 - alpha_fwd) * fwd_exp[i - 1]

        # Rolling std (60-day)
        roll_std = np.ones((t, n)) * 0.1
        for i in range(60, t):
            chunk = composite[i - 60 : i]
            s = np.std(chunk, axis=0, ddof=1)
            roll_std[i] = np.where(s < 1e-8, 0.1, s)

        data = MGDMarketData(
            pmi_surprise=pmi_surp,
            cpi_surprise=cpi_surp,
            emp_surprise=emp_surp,
            fwd_expectation=fwd_exp,
            roll_std=np.maximum(roll_std, 1e-8),
            forward_returns=fwd_ret,
            trend_baseline=trend_bl,
            momentum_baseline=mom_bl,
            carry_baseline=carry_bl,
            assets=tuple(assets),
            dates=common_idx,
        )
        self.validate(data)
        return data


# ---------------------------------------------------------------------------
# Proprietary Trading Firm provider stubs (plug-in replacements — implement on Day 1)
# ---------------------------------------------------------------------------


class HLSISCFProvider(AbstractISCFProvider):
    """Proprietary Trading Firm ISCF provider: LME/CME prompt-date spreads + freight.

    PLUG-IN REPLACEMENT for YFinanceISCFProvider.
    Implements identical AbstractISCFProvider contract.
    Replace the free-tier proxy with:
        - CME/LME front-month and deferred contract prices
        - LME warehouse inventory reports
        - Baltic Dry Index / freight rate feeds
        - Broker consensus macro-beta estimates

    To activate: export HLS_API_KEY=<key> and use this class.
    """

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    def fetch(
        self,
        start: str = "2015-01-01",
        end: str = "2024-12-31",
        assets: Optional[list[str]] = None,
    ) -> ISCFMarketData:
        raise NotImplementedError(
            "HLSISCFProvider: connect to HLS infrastructure on Day 1. "
            "Interface contract is identical to YFinanceISCFProvider."
        )


class HLSMGDProvider(AbstractMGDProvider):
    """Proprietary Trading Firm MGD provider: flash PMIs + CB real-time feeds.

    PLUG-IN REPLACEMENT for YFinanceMGDProvider.
    Replace the free-tier FRED proxies with:
        - Bloomberg consensus PMI/CPI/NFP surprise feeds (actual-consensus)
        - Intraday flash PMI data (Markit/S&P Global)
        - Central bank speaker calendars with NLP sentiment scores
        - FX forward curve data at 1M/3M/6M tenors

    To activate: export HLS_API_KEY=<key> and use this class.
    """

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    def fetch(
        self,
        start: str = "2015-01-01",
        end: str = "2024-12-31",
        assets: Optional[list[str]] = None,
    ) -> MGDMarketData:
        raise NotImplementedError(
            "HLSMGDProvider: connect to HLS infrastructure on Day 1. "
            "Interface contract is identical to YFinanceMGDProvider."
        )


# ---------------------------------------------------------------------------
# Provider factory — single entry point for all code
# ---------------------------------------------------------------------------


def get_iscf_provider(mode: str = "yfinance", **kwargs: object) -> AbstractISCFProvider:
    """Factory: return the appropriate ISCF data provider.

    Args:
        mode: "yfinance" | "proprietary"
        **kwargs: Passed to provider constructor.

    Returns:
        Concrete AbstractISCFProvider implementation.
    """
    providers = {
        "yfinance": YFinanceISCFProvider,
        "proprietary": HLSISCFProvider,
    }
    if mode not in providers:
        raise ValueError(
            f"Unknown ISCF provider mode: {mode!r}. Choose from {list(providers)}"
        )
    return providers[mode](**kwargs)  # type: ignore[arg-type]


def get_mgd_provider(mode: str = "yfinance", **kwargs: object) -> AbstractMGDProvider:
    """Factory: return the appropriate MGD data provider.

    Args:
        mode: "yfinance" | "proprietary"
        **kwargs: Passed to provider constructor.

    Returns:
        Concrete AbstractMGDProvider implementation.
    """
    providers = {
        "yfinance": YFinanceMGDProvider,
        "proprietary": HLSMGDProvider,
    }
    if mode not in providers:
        raise ValueError(
            f"Unknown MGD provider mode: {mode!r}. Choose from {list(providers)}"
        )
    return providers[mode](**kwargs)  # type: ignore[arg-type]
