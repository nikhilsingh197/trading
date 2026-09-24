"""ML-based signal strategy using trained probabilistic classifiers."""
from __future__ import annotations

from typing import Any
import pandas as pd

from ai_crypto_trader.core.enums import MarketRegime
from ai_crypto_trader.core.interfaces import Signal
from ai_crypto_trader.models.base import BaseMLModel
from ai_crypto_trader.strategies.base import BaseStrategy


class MLSignalStrategy(BaseStrategy):
    """Trading strategy powered by a machine learning model.

    Evaluates the model's predicted probability of upward price movement:
      P(up) >= long_threshold  -> ENTER_LONG
      P(up) <= short_threshold -> ENTER_SHORT
      otherwise               -> NO_TRADE
    """

    def __init__(
        self,
        model: BaseMLModel,
        version_id: str = "ml_v1",
        name: str = "MLSignalStrategy",
        params: dict[str, Any] | None = None,
    ) -> None:
        default_params = {
            "long_threshold": 0.55,
            "short_threshold": 0.45,
            "min_confidence": 0.10,
            "atr_sl_multiplier": 2.0,
            "tp_ratio": 2.0,
            "allowed_regimes": None,
        }
        if params:
            default_params.update(params)

        super().__init__(version_id=version_id, name=name, params=default_params)
        self.model = model

    def generate_signal(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        if df.empty:
            return self._no_trade_signal(df, "Empty dataframe", regime)

        # 1. Regime filter
        allowed = self.params.get("allowed_regimes")
        if allowed is not None and regime.value not in allowed and regime not in allowed:
            return self._no_trade_signal(df, f"Regime {regime.value} not allowed", regime)

        # 2. Extract latest row for inference
        last_row = df.iloc[-1:]
        symbol = str(last_row["symbol"].iloc[0]) if "symbol" in last_row.columns else "UNKNOWN"
        close = float(last_row["close"].iloc[0])

        try:
            pred = self.model.predict(last_row)
        except Exception as e:
            return self._no_trade_signal(df, f"Model prediction error: {e}", regime)

        prob = float(pred.get("probability", 0.5))
        confidence = float(pred.get("confidence", 0.0))

        # 3. Dynamic Stop Loss sizing using ATR if available
        if "vol_atr_14" in last_row.columns and pd.notna(last_row["vol_atr_14"].iloc[0]):
            atr = float(last_row["vol_atr_14"].iloc[0])
            stop_loss_pct = (atr * self.params["atr_sl_multiplier"]) / (close + 1e-9)
        else:
            stop_loss_pct = 0.015  # Fallback 1.5%

        stop_loss_pct = max(0.005, min(0.05, stop_loss_pct))  # Clamp between 0.5% and 5%
        tp_ratio = float(self.params["tp_ratio"])

        # 4. Generate Signal based on probability threshold
        long_thresh = float(self.params["long_threshold"])
        short_thresh = float(self.params["short_threshold"])
        min_conf = float(self.params["min_confidence"])

        if prob >= long_thresh and confidence >= min_conf:
            reason = (
                f"ML bullish (P={prob:.1%}, conf={confidence:.2f}, "
                f"model={self.model.model_type})"
            )
            return self._build_long_signal(
                df=df,
                confidence=confidence,
                stop_loss_pct=stop_loss_pct,
                tp_ratio=tp_ratio,
                reason=reason,
                regime=regime,
                symbol=symbol,
            )

        if prob <= short_thresh and confidence >= min_conf:
            reason = (
                f"ML bearish (P={prob:.1%}, conf={confidence:.2f}, "
                f"model={self.model.model_type})"
            )
            return self._build_short_signal(
                df=df,
                confidence=confidence,
                stop_loss_pct=stop_loss_pct,
                tp_ratio=tp_ratio,
                reason=reason,
                regime=regime,
                symbol=symbol,
            )

        neutral_reason = (
            f"Probability {prob:.1%} within neutral band "
            f"[{short_thresh:.1%}, {long_thresh:.1%}]"
        )
        return self._no_trade_signal(df, neutral_reason, regime)
