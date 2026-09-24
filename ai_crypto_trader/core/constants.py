"""Platform-wide constants."""

# ── Supported assets ──────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# ── Supported timeframes ──────────────────────────────────────────────────────
DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]

# ── Data quality flag values ──────────────────────────────────────────────────
QUALITY_OK = 0
QUALITY_INTERPOLATED = 1
QUALITY_SUSPECT = 2
QUALITY_REJECTED = 3

# ── Max allowed price deviation (% from rolling mean) before flagging ─────────
MAX_PRICE_DEVIATION_PCT = 20.0

# ── Max allowed volume spike (× rolling mean) before flagging ─────────────────
MAX_VOLUME_SPIKE_MULTIPLIER = 50.0

# ── Minimum candles required to compute most indicators ───────────────────────
MIN_CANDLES_FOR_SIGNAL = 200

# ── Backtest defaults ─────────────────────────────────────────────────────────
DEFAULT_MAKER_FEE = 0.0002   # 0.02%
DEFAULT_TAKER_FEE = 0.0005   # 0.05%
DEFAULT_SLIPPAGE = 0.0003    # 0.03%
DEFAULT_FUNDING_RATE = 0.0001  # 0.01% per 8h

# ── Kelly fraction cap (never bet more than this fraction of Kelly) ─────────
KELLY_FRACTION_CAP = 0.5

# ── Minimum statistical sample for strategy evaluation ───────────────────────
MIN_BACKTEST_TRADES = 50
MIN_PAPER_TRADES = 100

# ── Walk-forward defaults (months) ────────────────────────────────────────────
WF_TRAIN_MONTHS = 12
WF_VAL_MONTHS = 2
WF_TEST_MONTHS = 2
WF_STEP_MONTHS = 1

# ── Drift detection ───────────────────────────────────────────────────────────
DRIFT_ROLLING_WINDOW = 30     # trades
DRIFT_CONFIDENCE_WINDOW = 100  # predictions

# ── Redis channel names ───────────────────────────────────────────────────────
CHANNEL_CANDLE = "channel:candle"
CHANNEL_TICK = "channel:tick"
CHANNEL_SIGNAL = "channel:signal"
CHANNEL_ORDER = "channel:order"
CHANNEL_ALERT = "channel:alert"
CHANNEL_REGIME = "channel:regime"

# ── API pagination ────────────────────────────────────────────────────────────
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000

# ── Health check ─────────────────────────────────────────────────────────────
HEALTH_CHECK_INTERVAL_SECONDS = 5
STALE_DATA_THRESHOLD_SECONDS = 30
