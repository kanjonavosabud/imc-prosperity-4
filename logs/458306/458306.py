from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import math

# ============================================================
#  HYDROGEL_PACK — MAX-EXPLOITATION MARKET MAKER
# ============================================================
# Strategy: signal-driven inventory targeting + multi-tier passive quotes +
# selective high-conviction takes.
#
# The composite microstructure signal (regression-trained on days 0+1 fut_chg_5)
# predicts the next 5-tick mid change in price ticks. Empirical:
#   |signal| ≥ 2.5  → 78% sign-correctness, ±3.5 tick mean reversion
#   |signal| ≥ 4.0  → very rare, deep tail of conviction
#
# Exploitation channels:
#   1. Position TARGET = clamp(signal × TARGET_GAIN, ±LIMIT)
#   2. Fair-price skew  = mid + signal × FAIR_GAIN
#   3. Quote sizes scale CONTINUOUSLY with signal magnitude
#   4. Three passive tiers (aggressive maker / inside / at-market) for fill coverage
#   5. Take override at |signal| ≥ TAKE_TRIGGER (rare, L1-only, strict size cap)
# ============================================================

HP                   = "HYDROGEL_PACK"
HP_LIMIT             = 60

# --- Signal weights (from OLS regression) ---
SIG_W_IMB_L1   = +4.74
SIG_W_IMB_L2   = +0.44
SIG_W_IMB_L3   = -4.56
SIG_W_MICRO    = -0.83
SIG_W_DL12     = -1.52
SIG_W_VWAP_DEV = +1.69

# --- Position TARGET driven by signal ---
SIG_TARGET_GAIN      = 12.0    # target = signal × this, clamped to ±LIMIT
                                # signal ≈ 5 → target ≈ 60 (full limit)

# --- Fair price construction ---
SIG_FAIR_GAIN        = 1.0     # quote prices shift by this × signal (in ticks)
INV_SKEW_GAIN        = 0.30    # inventory-deviation skew (push toward target)

# --- Quote tiers (multi-level passive) ---
# Tier 1: aggressive maker (close to fair, big size, max spread capture per fill)
TIER1_OFFSET         = 4       # ticks from fair (was half-spread = 8)
TIER1_SIZE_BASE      = 30
# Tier 2: at the inside (bb+1 / ba-1, queue priority, medium size)
TIER2_SIZE_BASE      = 15
# Tier 3: at-market (bb / ba, full spread when filled, small size)
TIER3_SIZE_BASE      = 10

# --- Continuous size scaling ---
# Sizes get multiplied by (1 + |signal|/SIG_SIZE_SCALE) on the signal-aligned side
SIG_SIZE_SCALE       = 3.0     # signal of 3 → 2× size on aligned side
SIZE_FAVOR_RATIO     = 0.25    # opposing-side size is fraction of aligned

# --- Aggressive take (L1 only, very high conviction) ---
HP_TAKE_TRIGGER      = 4.0     # |signal| ≥ this allows a take
HP_TAKE_MAX_QTY      = 15      # max contracts to TAKE per tick (L1 only)
HP_TAKE_COOLDOWN_TS  = 500     # ticks between takes (in timestamp units)

# --- Hard inventory protection ---
HP_INV_PROTECT_AT    = 55      # at |pos| ≥ this, suppress same-side adds


# ============================================================
#  TRADER
# ============================================================

class Trader:

    # ---- order-book helpers ----------------------------------

    def _mid(self, od: OrderDepth) -> Optional[float]:
        if od and od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    def _best_bid_ask(self, od: OrderDepth):
        bb = max(od.buy_orders) if od and od.buy_orders else None
        ba = min(od.sell_orders) if od and od.sell_orders else None
        return bb, ba

    def _level_volumes(self, od: OrderDepth):
        bids = sorted(od.buy_orders.items(), key=lambda x: -x[0])
        asks = sorted(od.sell_orders.items(), key=lambda x: x[0])
        bid_p = [None]*3; bid_v = [0]*3
        ask_p = [None]*3; ask_v = [0]*3
        for i in range(min(3, len(bids))):
            bid_p[i] = bids[i][0]; bid_v[i] = bids[i][1]
        for i in range(min(3, len(asks))):
            ask_p[i] = asks[i][0]; ask_v[i] = abs(asks[i][1])
        return bid_p, bid_v, ask_p, ask_v

    def _signal(self, od: OrderDepth) -> float:
        """Predicted 5-tick mid change in price ticks. Returns 0 if book malformed."""
        bid_p, bid_v, ask_p, ask_v = self._level_volumes(od)
        if bid_p[0] is None or ask_p[0] is None:
            return 0.0

        bv1, bv2, bv3 = bid_v
        av1, av2, av3 = ask_v
        bp1, ap1 = bid_p[0], ask_p[0]
        L1_mid = (bp1 + ap1) / 2.0
        L2_mid = (bid_p[1] + ask_p[1]) / 2.0 if bid_p[1] and ask_p[1] else L1_mid
        dL12 = L1_mid - L2_mid

        def imb(b, a):
            d = b + a
            return (b - a) / d if d > 0 else 0.0

        imb_L1 = imb(bv1, av1)
        imb_L2 = imb(bv2, av2)
        imb_L3 = imb(bv3, av3)

        denom = bv1 + av1
        micro = (bp1 * av1 + ap1 * bv1) / denom if denom > 0 else L1_mid
        micro_dev = micro - L1_mid

        total_b = bv1 + bv2 + bv3
        total_a = av1 + av2 + av3
        vwap_b = (bp1*bv1 + (bid_p[1] or bp1)*bv2 + (bid_p[2] or bp1)*bv3) / total_b if total_b > 0 else bp1
        vwap_a = (ap1*av1 + (ask_p[1] or ap1)*av2 + (ask_p[2] or ap1)*av3) / total_a if total_a > 0 else ap1
        vwap_mid = (vwap_b + vwap_a) / 2.0
        vwap_dev = L1_mid - vwap_mid

        return (SIG_W_IMB_L1 * imb_L1
              + SIG_W_IMB_L2 * imb_L2
              + SIG_W_IMB_L3 * imb_L3
              + SIG_W_MICRO  * micro_dev
              + SIG_W_DL12   * dL12
              + SIG_W_VWAP_DEV * vwap_dev)

    # ---- main strategy ---------------------------------------

    def _trade_hydrogel(self, state: TradingState, data: dict) -> List[Order]:
        od = state.order_depths.get(HP, OrderDepth())
        pos = state.position.get(HP, 0)
        mid = self._mid(od)
        bb, ba = self._best_bid_ask(od)
        if mid is None or bb is None or ba is None:
            return []

        bid_p, bid_v, ask_p, ask_v = self._level_volumes(od)
        signal = self._signal(od)

        # ---- 1. Position TARGET driven by signal (continuous) ----
        target = max(-HP_LIMIT, min(HP_LIMIT, signal * SIG_TARGET_GAIN))

        # ---- 2. Fair price (signal + inventory-deviation skew) ----
        # Inventory skew: push fair AWAY from current pos and TOWARD target
        inv_dev = pos - target          # >0 if we're more long than target
        if inv_dev > 0:
            inv_skew = -INV_SKEW_GAIN * math.sqrt(inv_dev)
        elif inv_dev < 0:
            inv_skew = +INV_SKEW_GAIN * math.sqrt(-inv_dev)
        else:
            inv_skew = 0.0

        fair = mid + SIG_FAIR_GAIN * signal + inv_skew

        # ---- 3. Aggressive take override at high conviction ----
        # Only fires at |signal| ≥ TAKE_TRIGGER, with cooldown to prevent walking.
        last_take_ts = data.get("last_take_ts", -10000)
        cooldown_ok = (state.timestamp - last_take_ts) >= HP_TAKE_COOLDOWN_TS

        orders: List[Order] = []
        if cooldown_ok and abs(signal) >= HP_TAKE_TRIGGER:
            buy_cap = HP_LIMIT - pos
            sell_cap = HP_LIMIT + pos
            if signal > 0 and buy_cap > 0:
                # BUY: lift L1 ask only
                qty = min(HP_TAKE_MAX_QTY, ask_v[0], buy_cap)
                if qty > 0:
                    orders.append(Order(HP, ask_p[0], qty))
                    data["last_take_ts"] = state.timestamp
                    pos += qty  # local update for downstream sizing
            elif signal < 0 and sell_cap > 0:
                qty = min(HP_TAKE_MAX_QTY, bid_v[0], sell_cap)
                if qty > 0:
                    orders.append(Order(HP, bid_p[0], -qty))
                    data["last_take_ts"] = state.timestamp
                    pos -= qty

        # ---- 4. Continuous size scaling ----
        sig_mag = min(abs(signal) / SIG_SIZE_SCALE, 2.0)  # cap at 2× base
        align_mult = 1.0 + sig_mag                         # signal-aligned side
        oppose_mult = max(SIZE_FAVOR_RATIO, 1.0 - sig_mag) # opposing side

        if signal >= 0:
            bid_mult, ask_mult = align_mult, oppose_mult
        else:
            bid_mult, ask_mult = oppose_mult, align_mult

        # ---- 5. Quote prices: three tiers, all signal-skewed ----
        # Tier 1: aggressive maker close to fair (big spread capture if filled)
        t1_bid_target = fair - TIER1_OFFSET
        t1_ask_target = fair + TIER1_OFFSET
        t1_bid = max(int(math.floor(t1_bid_target)), bb)
        t1_bid = min(t1_bid, ba - 1)
        t1_ask = min(int(math.ceil(t1_ask_target)), ba)
        t1_ask = max(t1_ask, bb + 1)
        if t1_bid >= t1_ask:
            t1_bid = t1_ask - 1

        # Tier 2: 1 inside the market (queue priority, less spread)
        t2_bid = max(bb + 1, t1_bid - 1)
        t2_ask = min(ba - 1, t1_ask + 1)

        # Tier 3: AT the market (full spread when filled, lowest fill probability)
        t3_bid = bb
        t3_ask = ba

        # ---- 6. Compute sizes per tier with caps, bias, and inventory protection ----
        buy_cap_total = HP_LIMIT - pos
        sell_cap_total = HP_LIMIT + pos

        # Inventory protection: at near-limit, suppress the limit-side
        if pos >= HP_INV_PROTECT_AT:
            buy_cap_total = 0
        elif pos <= -HP_INV_PROTECT_AT:
            sell_cap_total = 0

        # Tier 1 (most aggressive)
        t1_bid_sz = min(int(TIER1_SIZE_BASE * bid_mult), buy_cap_total)
        t1_ask_sz = min(int(TIER1_SIZE_BASE * ask_mult), sell_cap_total)
        buy_cap_total -= max(t1_bid_sz, 0)
        sell_cap_total -= max(t1_ask_sz, 0)

        # Tier 2 (inside)
        t2_bid_sz = min(int(TIER2_SIZE_BASE * bid_mult), buy_cap_total)
        t2_ask_sz = min(int(TIER2_SIZE_BASE * ask_mult), sell_cap_total)
        buy_cap_total -= max(t2_bid_sz, 0)
        sell_cap_total -= max(t2_ask_sz, 0)

        # Tier 3 (at market)
        t3_bid_sz = min(int(TIER3_SIZE_BASE * bid_mult), buy_cap_total)
        t3_ask_sz = min(int(TIER3_SIZE_BASE * ask_mult), sell_cap_total)

        # ---- 7. Place orders, deduplicating same-price tiers ----
        # If two tiers land at the same price, merge their sizes.
        bid_orders = {}  # price -> total size
        ask_orders = {}

        def add_bid(px, sz):
            if px is None or sz <= 0 or px >= ba:
                return
            bid_orders[px] = bid_orders.get(px, 0) + sz

        def add_ask(px, sz):
            if px is None or sz <= 0 or px <= bb:
                return
            ask_orders[px] = ask_orders.get(px, 0) + sz

        if t1_bid_sz > 0: add_bid(t1_bid, t1_bid_sz)
        if t2_bid_sz > 0: add_bid(t2_bid, t2_bid_sz)
        if t3_bid_sz > 0: add_bid(t3_bid, t3_bid_sz)
        if t1_ask_sz > 0: add_ask(t1_ask, t1_ask_sz)
        if t2_ask_sz > 0: add_ask(t2_ask, t2_ask_sz)
        if t3_ask_sz > 0: add_ask(t3_ask, t3_ask_sz)

        for px, sz in bid_orders.items():
            orders.append(Order(HP, px, sz))
        for px, sz in ask_orders.items():
            orders.append(Order(HP, px, -sz))

        return orders

    # ---- main entry point ------------------------------------

    def run(self, state: TradingState):
        # State carries last-take timestamp for cooldown
        import json
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}
        result: Dict[str, List[Order]] = {HP: self._trade_hydrogel(state, data)}
        return result, 0, json.dumps(data)