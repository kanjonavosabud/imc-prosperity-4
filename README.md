# IMC Prosperity 4

My team's algorithms and research notebooks for the IMC Prosperity 4 trading competition (April 2026). Five rounds of algorithmic trading on a simulated exchange, plus manual trading puzzles each round.

## Layout

```
.
├── datamodel.py            # Competition-provided types (Order, OrderDepth, TradingState, ...)
├── log_viewer.ipynb        # Notebook for inspecting sandbox/backtest logs
├── requirements.txt        # pandas, numpy, jsonpickle
├── tutorial/               # Pre-competition tutorial round
├── round-1/ ... round-5/   # One folder per round: data, EDA, strategies, results logs
├── manual/                 # Manual-trading puzzle screenshots and results
├── logs/                   # Submission logs grouped by submission ID
└── backtests/              # Local backtest output (gitignored)
```

Each `round-N/` folder follows the same shape:
- `ROUND_N/` — official price/trade CSVs (one per simulated day) and `eda.ipynb`.
- `round-N.py` — initial scaffold for the round.
- `round-N-final.py` (or `round-N-best.py`, `round-N-updated.py`) — the version submitted at the deadline.
- Variant files (`claude-...`, `gemini-...`, `krishi-...`, `-v4`, `-v6`, `-from-1`, `-hp-v2`, etc.) — alternate strategies developed during iteration.
- `round-N-results-logs.zip` — sandbox logs from the official submission.

## Round-by-round summary

### Tutorial — [tutorial/](tutorial/)
Warm-up round on the original Prosperity products. Two algo iterations in [tutorial-algo-1.py](tutorial/tutorial-algo-1.py) and [tutorial-algo-2.py](tutorial/tutorial-algo-2.py).

### Round 1 — [round-1/](round-1/)
Products: `ASH_COATED_OSMIUM` (stable, mean-reverts ~10000) and `INTARIAN_PEPPER_ROOT` (trending).

Final submission: [round-1-final.py](round-1/round-1-final.py). Layered market-making with EWMA fair value, primary + L2 quotes, and inventory skew. Per-product position limit = 80.

### Round 2 — [round-2/](round-2/)
Same two products as round 1, plus a Market Access Fee (MAF) auction. The bot bids 2,500 in [round-2-final.py](round-2/round-2-final.py) for a 25% volume bonus. Adds counterparty tracking by parsing `own_trades` to learn who is hitting our quotes.

`INTARIAN_PEPPER_ROOT` got an adaptive linear-ramp regime: target position scales from 0 → 80 over ~3000 timestamps when slope > threshold, with asymmetric bid/ask sizes to ride the trend.

### Round 3 — [round-3/](round-3/)
Options round. Underlying `VELVETFRUIT_EXTRACT` plus four `VEV_*` strike vouchers (4000/4500/5000/5100). Also `HYDROGEL_PACK` (mean-reverts ~9990).

[round-3-final.py](round-3/round-3-final.py) implements:
- Black-Scholes with a fitted IV smile: `IV(m_t) = a·m_t² + b·m_t + c` where `m_t = log(S/K)/√T`.
- Per-strike strategy: intrinsic FV for deep ITM (4000/4500), microprice MM for near-ATM (5000/5100).
- OU mean-reversion blend for `HYDROGEL_PACK` and `VELVETFRUIT_EXTRACT` (calibrated half-lives ~550 and ~45 ticks).
- IV scalping framework on far-OTM strikes (kept dormant — thresholds didn't fire in data).

[local_backtest.py](round-3/ROUND_3/local_backtest.py) was added for offline tuning.

### Round 4 — [round-4/](round-4/)
Combined strategy in [round-4-best.py](round-4/round-4-best.py): teammate's HYDROGEL_PACK block ($4,085 day-3) sandwiched with the VELVETFRUIT_EXTRACT + 4-voucher block ($9,127 day-3). State namespacing (`hp_*` keys vs unprefixed) keeps the two halves from colliding.

Notable additions: counterparty signals (M14/M22/M38/M49/M67) gate aggressive entries, [Path E] time-decay inventory skew, [Path B] drop-phase aggressor.

Manual puzzle solved in [round-4-manual.ipynb](round-4/round-4-manual.ipynb).

### Round 5 — [round-5/](round-5/)
50 products, position limit 10 per product. Final submission: [round-5-updated.py](round-5/round-5-updated.py).

Three layers:
- **PEBBLES basket arb.** `PEBBLES_XS + S + M + L + XL ≡ 50,000`. When the residual exceeds ~1.4σ, fire all 5 legs; per-leg MM only on `PEBBLES_S` (cleanest signal).
- **MICROCHIP buy-pressure quoting.** XIRECS is a +7% net buyer across the family; skewed quotes (tight ask, wide bid) capture the lift.
- **Trend-aware MM** for the remaining 43 products: mid + EWMA trend bias + Stoikov inventory skew + penny-inside quotes + active flatten when |pos| ≥ 8.

Two layers from an earlier draft (SNACKPACK CHOC/VAN z-score, UV_VISOR_AMBER cointegration) were removed after failing to monetize live on day 4.

Manual round 5 used a news-headline puzzle ([manual-round-5-news.jpg](round-5/manual-round-5-news.jpg), [manual-round-5.ipynb](round-5/manual-round-5.ipynb), [manual-round-5-chat-transcript.txt](round-5/manual-round-5-chat-transcript.txt)).

## Manual puzzles
[manual/](manual/) holds the puzzle screenshots and result PNGs for rounds 1–5.

## Running
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Strategies are submitted as a single Python file exposing a `Trader` class with a `run(state: TradingState)` method (see any `round-N-final.py`). `datamodel.py` contains the types the platform passes in.

Logs from each submission are stored under [logs/](logs/) keyed by submission ID; open [log_viewer.ipynb](log_viewer.ipynb) to inspect them. Backtest output goes to `backtests/` (gitignored).
