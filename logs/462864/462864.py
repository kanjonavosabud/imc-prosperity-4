from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math

# ============================================================
#  HYDROGEL_PACK V2 — heavy market maker (461401-derived + extras)
# ============================================================
# Offline 3-day backtest (take-only PnL, passive layer doesn't fill in sim):
#   Base 461401 params:        +62,471
#   + tuned skew (0.01/0.4):   +92,489
#   + dL12 signal:             +94,758  ← shipping config
# Live IMC adds the passive quote layer on top.
#
# Key mechanics (vs the simple z-score that did +16k):
#   1. Position limit 200 (was 60) — 3.3× more capacity per cycle
#   2. Microprice fair value (was rolling mean) — instant, no warmup
#   3. OU blend pulls fair toward 9990 with 30% weight
#   4. Take BOTH sides every tick (no edge buffer): lift cheap asks AND hit rich bids
#   5. Post full remaining capacity at bb+1 / ba-1 (penny the inside)
#   6. Non-linear inventory skew (linear + quadratic, blows up near limit)
#   7. L1 imbalance signal (3:1 ratio → fair ±3 ticks)
#   8. NEW: dL12 signal (|L1_mid - L2_mid| ≥ 2 → fair ±2 ticks, 97% hit rate)
# ============================================================

HP                  = "HYDROGEL_PACK"
HP_LIMIT            = 200
HP_EDGE             = 12          # passive quote distance from fair (capped to inside)

# --- Inventory skew (linear + non-linear) ---
HP_SKEW             = 0.01        # ticks per lot of position
HP_NL_COEFF         = 0.4         # quadratic blow-up factor near limit

# --- OU mean-reversion blend (HYDROGEL trades around 9990) ---
HP_LT_MEAN_INIT     = 9990.0      # historical mean
HP_LT_BLEND         = 0.30        # weight of long-term mean in fair value
HP_LT_ALPHA         = 0.001       # slow EMA update of long-term mean

# --- L1 imbalance signal (rare but high-precision) ---
HP_IMB_RATIO        = 3.0         # ratio threshold (e.g., bid_v ≥ 3 × ask_v)
HP_IMB_SHIFT        = 3.0         # ticks to shift fair on imbalance signal

# --- L1-vs-L2 mid signal (NEW) ---
# Earlier EDA: when |L1_mid - L2_mid| ≥ 2, L1 mean-reverts ~4 ticks within 1-10 ticks
# at 97.5% hit rate. We shift fair AGAINST the gap to incentivize the right takes.
HP_DL12_THRESH      = 2.0
HP_DL12_SHIFT       = 2.0

# --- Multi-tier passive quotes (NEW) ---
# Tier 1: at the inside (bb+1 / ba-1) with primary size.
# Tier 2: at the market (bb / ba) for queue-displaced fills.
HP_TIER1_FRAC       = 0.70        # primary tier gets 70% of remaining capacity
HP_TIER2_FRAC       = 0.30        # secondary tier gets 30%


# ============================================================
#  TRADER
# ============================================================

class Trader:

    def _load(self, raw: str) -> dict:
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return {}

    def _save(self, d: dict) -> str:
        return json.dumps(d)

    def _trade_hydrogel(self, state: TradingState, mem: dict) -> List[Order]:
        od = state.order_depths.get(HP)
        if od is None or not od.buy_orders or not od.sell_orders:
            return []

        bb = max(od.buy_orders.keys())
        ba = min(od.sell_orders.keys())
        bv1 = od.buy_orders[bb]
        av1 = -od.sell_orders[ba]
        mid = (bb + ba) / 2.0
        pos = state.position.get(HP, 0)

        # ---- 1. Microprice fair value ----
        denom = bv1 + av1
        if denom > 0:
            fair = (bb * av1 + ba * bv1) / denom
        else:
            fair = mid

        # ---- 2. OU mean-reversion blend ----
        lt_mean = mem.get("hp_lt_mean", HP_LT_MEAN_INIT)
        lt_mean = HP_LT_ALPHA * mid + (1 - HP_LT_ALPHA) * lt_mean
        mem["hp_lt_mean"] = lt_mean
        fair = (1 - HP_LT_BLEND) * fair + HP_LT_BLEND * lt_mean

        # ---- 3. L1 imbalance signal (3:1 ratio) ----
        if av1 > 0 and bv1 > 0:
            if bv1 >= HP_IMB_RATIO * av1:
                fair += HP_IMB_SHIFT
            elif av1 >= HP_IMB_RATIO * bv1:
                fair -= HP_IMB_SHIFT

        # ---- 4. dL12 signal (NEW) ----
        # Need L2 quotes to compute. Find second-best bid/ask.
        bids_sorted = sorted(od.buy_orders.keys(), reverse=True)
        asks_sorted = sorted(od.sell_orders.keys())
        if len(bids_sorted) >= 2 and len(asks_sorted) >= 2:
            L1_mid = (bb + ba) / 2.0
            L2_mid = (bids_sorted[1] + asks_sorted[1]) / 2.0
            dL = L1_mid - L2_mid
            if dL >= HP_DL12_THRESH:
                fair -= HP_DL12_SHIFT       # L1 too high → expect drop
            elif dL <= -HP_DL12_THRESH:
                fair += HP_DL12_SHIFT       # L1 too low → expect rise

        # ---- 5. Non-linear inventory skew ----
        pos_ratio = pos / HP_LIMIT
        fv_shift = pos * HP_SKEW + pos_ratio * abs(pos_ratio) * HP_LIMIT * HP_SKEW * HP_NL_COEFF
        fv = fair - fv_shift
        fv_int = int(round(fv))

        buy_cap = HP_LIMIT - pos
        sell_cap = HP_LIMIT + pos

        orders: List[Order] = []

        # ---- 6. TAKE both sides (no edge buffer) ----
        for ap in asks_sorted:
            if buy_cap <= 0 or ap > fv_int:
                break
            avail = -od.sell_orders[ap]
            qty = min(avail, buy_cap)
            if qty > 0:
                orders.append(Order(HP, ap, qty))
                buy_cap -= qty
        for bp in bids_sorted:
            if sell_cap <= 0 or bp < fv_int:
                break
            avail = od.buy_orders[bp]
            qty = min(avail, sell_cap)
            if qty > 0:
                orders.append(Order(HP, bp, -qty))
                sell_cap -= qty

        # ---- 7. Multi-tier passive quote layer ----
        my_bid = fv_int - HP_EDGE
        my_ask = fv_int + HP_EDGE
        # Cap to inside the spread
        my_bid = min(my_bid, int(mid) - 1)
        my_ask = max(my_ask, int(mid) + 1)
        my_bid = max(my_bid, bb + 1)
        my_ask = min(my_ask, ba - 1)
        if my_bid >= my_ask:
            my_bid = int(mid) - 1
            my_ask = int(mid) + 1

        # Tier 1 at bb+1 / ba-1 (or skewed deeper if fair pulled them in)
        # Tier 2 at bb / ba (full spread, queue-displaced)
        t1_bid_sz = int(buy_cap * HP_TIER1_FRAC)
        t2_bid_sz = buy_cap - t1_bid_sz
        t1_ask_sz = int(sell_cap * HP_TIER1_FRAC)
        t2_ask_sz = sell_cap - t1_ask_sz

        if t1_bid_sz > 0:
            orders.append(Order(HP, my_bid, t1_bid_sz))
        if t2_bid_sz > 0 and bb < my_bid:
            orders.append(Order(HP, bb, t2_bid_sz))
        if t1_ask_sz > 0:
            orders.append(Order(HP, my_ask, -t1_ask_sz))
        if t2_ask_sz > 0 and ba > my_ask:
            orders.append(Order(HP, ba, -t2_ask_sz))

        return orders

    def run(self, state: TradingState):
        mem = self._load(state.traderData)
        result: Dict[str, List[Order]] = {HP: self._trade_hydrogel(state, mem)}
        return result, 0, self._save(mem)