from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math

# ============================================================
#  CONFIGURABLE PARAMETERS
# ============================================================

HP                 = "HYDROGEL_PACK"
HP_LIMIT           = 60

# --- Z-score mean reversion (position target) ---------------
HP_WINDOW          = 2000
HP_MIN_WINDOW      = 200
HP_Z_ENTRY         = 2.0
HP_Z_EXIT          = 0.5
HP_TAKE_EDGE       = 1
HP_ENTRY_CHUNK     = 20

# --- Microstructure composite signal (fair-price skew) ------
# Coefficients learned by OLS regression on days 0+1, fut_chg_5 target.
# Signal = predicted 5-tick mid change (in ticks).
# Tail behavior: when |signal| >= 2.5, sign-correctness = 78% (hit rate).
# Features:
#   imb_Lk    = (bid_volume_k - ask_volume_k) / (sum)        [-1, +1]
#   micro_dev = microprice - L1_mid                          (price ticks)
#   dL12      = L1_mid - L2_mid                              (price ticks)
#   vwap_dev  = L1_mid - vwap_mid                            (price ticks)
SIG_W_IMB_L1   = +4.74
SIG_W_IMB_L2   = +0.44
SIG_W_IMB_L3   = -4.56     # contrarian: L3 bid-heavy → price falls
SIG_W_MICRO    = -0.83
SIG_W_DL12     = -1.52
SIG_W_VWAP_DEV = +1.69

# How much to skew our fair price by the signal. 1.0 = use full predicted move.
# Conservative default (0.5) — signal noisy, don't overcommit.
SIG_FAIR_GAIN  = 0.5

# --- Quote placement ---------------------------------------
# Wider quotes: post AT the market's bid/ask (not 1 inside) → MY spread
# matches market spread for max profit per fill.
HP_QUOTE_OFFSET    = 0       # 0 = post AT bb/ba; 1 = post inside (narrower MY spread)
HP_QUOTE_SIZE      = 30      # passive top-up size at target

# Inventory skew (push fair away from current position)
HP_INV_SKEW_COEFF  = 0.10    # ticks per √|pos| of skew


# ============================================================
#  TRADER
# ============================================================

class Trader:

    # ---- state persistence -----------------------------------

    def _load(self, raw: str) -> dict:
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return {}

    def _save(self, d: dict) -> str:
        return json.dumps(d)

    # ---- order-book helpers ----------------------------------

    def _mid(self, od: OrderDepth) -> Optional[float]:
        if od and od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    def _best_bid_ask(self, od: OrderDepth):
        bb = max(od.buy_orders) if od and od.buy_orders else None
        ba = min(od.sell_orders) if od and od.sell_orders else None
        return bb, ba

    def _level_volumes(self, od: OrderDepth) -> Tuple[List[int], List[int]]:
        """Return (bid_levels, ask_levels) sorted from best inward, padded to 3."""
        bids = sorted(od.buy_orders.items(), key=lambda x: -x[0])
        asks = sorted(od.sell_orders.items(), key=lambda x: x[0])
        # bids[i] = (price, volume); asks[i] = (price, -volume)
        bid_p, bid_v = [], []
        for i in range(3):
            if i < len(bids):
                bid_p.append(bids[i][0]); bid_v.append(bids[i][1])
            else:
                bid_p.append(None); bid_v.append(0)
        ask_p, ask_v = [], []
        for i in range(3):
            if i < len(asks):
                ask_p.append(asks[i][0]); ask_v.append(abs(asks[i][1]))
            else:
                ask_p.append(None); ask_v.append(0)
        return bid_p, bid_v, ask_p, ask_v

    # ---- microstructure features + composite signal ---------

    def _signal(self, od: OrderDepth) -> Optional[float]:
        """Return predicted 5-tick mid change in price ticks. None if book malformed."""
        bid_p, bid_v, ask_p, ask_v = self._level_volumes(od)
        if bid_p[0] is None or ask_p[0] is None:
            return None

        bv1, bv2, bv3 = bid_v
        av1, av2, av3 = ask_v
        bp1 = bid_p[0]; ap1 = ask_p[0]

        L1_mid = (bp1 + ap1) / 2.0

        # L2 mid (None if either L2 missing)
        if bid_p[1] is not None and ask_p[1] is not None:
            L2_mid = (bid_p[1] + ask_p[1]) / 2.0
            dL12 = L1_mid - L2_mid
        else:
            dL12 = 0.0

        # Imbalance per level
        def imb(b, a):
            denom = b + a
            return (b - a) / denom if denom > 0 else 0.0

        imb_L1 = imb(bv1, av1)
        imb_L2 = imb(bv2, av2)
        imb_L3 = imb(bv3, av3)

        # Microprice
        denom = bv1 + av1
        if denom > 0:
            micro = (bp1 * av1 + ap1 * bv1) / denom
        else:
            micro = L1_mid
        micro_dev = micro - L1_mid

        # VWAP mid (volume-weighted across all 3 levels)
        total_b = bv1 + bv2 + bv3
        total_a = av1 + av2 + av3
        if total_b > 0:
            vwap_b = (bp1*bv1 + (bid_p[1] or bp1)*bv2 + (bid_p[2] or bp1)*bv3) / total_b
        else:
            vwap_b = bp1
        if total_a > 0:
            vwap_a = (ap1*av1 + (ask_p[1] or ap1)*av2 + (ask_p[2] or ap1)*av3) / total_a
        else:
            vwap_a = ap1
        vwap_mid = (vwap_b + vwap_a) / 2.0
        vwap_dev = L1_mid - vwap_mid

        signal = (SIG_W_IMB_L1 * imb_L1
                + SIG_W_IMB_L2 * imb_L2
                + SIG_W_IMB_L3 * imb_L3
                + SIG_W_MICRO  * micro_dev
                + SIG_W_DL12   * dL12
                + SIG_W_VWAP_DEV * vwap_dev)
        return signal

    # ---- HYDROGEL_PACK strategy ------------------------------

    def _trade_hydrogel(self, state: TradingState, data: dict) -> List[Order]:
        od = state.order_depths.get(HP, OrderDepth())
        pos = state.position.get(HP, 0)
        mid = self._mid(od)
        if mid is None:
            return []

        bb, ba = self._best_bid_ask(od)
        if bb is None or ba is None:
            return []

        # ---- Rolling history for z-score target ----
        hist = data.setdefault("hp_mids", [])
        hist.append(mid)
        cap = HP_WINDOW * 2
        if len(hist) > cap:
            del hist[: len(hist) - cap]
        if len(hist) < HP_MIN_WINDOW:
            return []

        eff_n = min(len(hist), HP_WINDOW)
        recent = hist[-eff_n:]
        mu = sum(recent) / eff_n
        var = sum((x - mu) ** 2 for x in recent) / (eff_n - 1)
        sigma = math.sqrt(var) if var > 1e-9 else 1.0
        z = (mid - mu) / sigma

        # Z-score sets the position target (long-horizon mean reversion)
        target = data.get("hp_target", 0)
        if z > HP_Z_ENTRY:
            target = -HP_LIMIT
        elif z < -HP_Z_ENTRY:
            target = HP_LIMIT
        elif HP_Z_EXIT > 0 and abs(z) < HP_Z_EXIT:
            target = 0
        data["hp_target"] = target

        # ---- Microstructure-skewed fair (used for QUOTING only) ----
        signal = self._signal(od) or 0.0
        inv_sign = -1 if pos > 0 else (1 if pos < 0 else 0)
        inv_skew = HP_INV_SKEW_COEFF * inv_sign * math.sqrt(abs(pos)) if pos != 0 else 0.0
        # quote_fair: short-term, microstructure-driven
        quote_fair = mid + SIG_FAIR_GAIN * signal + inv_skew
        # take_fair: long-horizon rolling mean (z-score's mean-reversion anchor).
        # The signal AUGMENTS take threshold: if signal predicts price will rise,
        # we lower the take_fair_buy threshold (less eager to buy now).
        take_fair = mu + SIG_FAIR_GAIN * signal

        orders: List[Order] = []
        delta = target - pos

        # ---- Aggressive take, using rolling-mean based threshold ----
        if delta > 0:
            need = delta
            chunk = HP_ENTRY_CHUNK
            for px in sorted(od.sell_orders.keys()):
                if need <= 0 or chunk <= 0 or px > take_fair + HP_TAKE_EDGE:
                    break
                avail = -od.sell_orders[px]
                qty = min(avail, need, chunk)
                if qty > 0:
                    orders.append(Order(HP, px, qty))
                    need -= qty; chunk -= qty
        elif delta < 0:
            need = -delta
            chunk = HP_ENTRY_CHUNK
            for px in sorted(od.buy_orders.keys(), reverse=True):
                if need <= 0 or chunk <= 0 or px < take_fair - HP_TAKE_EDGE:
                    break
                avail = od.buy_orders[px]
                qty = min(avail, need, chunk)
                if qty > 0:
                    orders.append(Order(HP, px, -qty))
                    need -= qty; chunk -= qty

        # ---- Passive quotes: WIDER spread, signal-skewed ----
        # Post at MAX(bb, quote_fair - half_market_spread) and MIN(ba, quote_fair + half_market_spread)
        # If signal pushes quote_fair up, our bid moves up but capped at ba-1 (don't cross),
        # ask moves up too (less aggressive on sell side).
        market_spread = ba - bb
        half_market = market_spread / 2.0

        # Quote prices: signal-skewed but bounded by market inside
        bid_px_target = quote_fair - half_market + HP_QUOTE_OFFSET
        ask_px_target = quote_fair + half_market - HP_QUOTE_OFFSET
        # Cap to NOT cross the market (we are makers, not takers here)
        bid_px = min(int(math.floor(bid_px_target)), ba - 1)
        bid_px = max(bid_px, bb)            # AT or above existing best bid
        ask_px = max(int(math.ceil(ask_px_target)), bb + 1)
        ask_px = min(ask_px, ba)            # AT or below existing best ask

        buy_cap = HP_LIMIT - pos
        sell_cap = HP_LIMIT + pos

        # Quote sizes: bias toward target side
        if target > pos:
            bid_sz = min(HP_QUOTE_SIZE, buy_cap)
            ask_sz = min(HP_QUOTE_SIZE // 3, sell_cap)
        elif target < pos:
            bid_sz = min(HP_QUOTE_SIZE // 3, buy_cap)
            ask_sz = min(HP_QUOTE_SIZE, sell_cap)
        else:
            bid_sz = min(HP_QUOTE_SIZE // 2, buy_cap)
            ask_sz = min(HP_QUOTE_SIZE // 2, sell_cap)

        if bid_sz > 0 and bid_px < ask_px:
            orders.append(Order(HP, bid_px, bid_sz))
        if ask_sz > 0 and ask_px > bid_px:
            orders.append(Order(HP, ask_px, -ask_sz))

        return orders

    # ---- main entry point ------------------------------------

    def run(self, state: TradingState):
        data = self._load(state.traderData)
        result: Dict[str, List[Order]] = {HP: self._trade_hydrogel(state, data)}
        return result, 0, self._save(data)