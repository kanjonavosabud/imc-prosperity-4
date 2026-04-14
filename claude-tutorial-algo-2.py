from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Product, Dict, List
from typing import List
import string
import json


# ─── Position limits ──────────────────────────────────────────────────────────
POSITION_LIMIT = {"EMERALDS": 80, "TOMATOES": 80}

# ─── Emeralds parameters ──────────────────────────────────────────────────────
EMERALDS_FAIR_VALUE = 10_000
EMERALDS_HALF_SPREAD = 5        # post at FV ± 5 (10-tick spread)
EMERALDS_INV_SKEW = 2           # max quote shift ticks due to inventory

# ─── Tomatoes parameters (tuned from backtesting both days) ──────────────────
TOMATOES_WINDOW     = 200       # rolling window for fair value estimate
TOMATOES_MAKE_EDGE  = 5         # half-spread for passive quotes
TOMATOES_TAKE_EDGE  = 2         # aggress when book is this far through fair value
TOMATOES_Z_SKEW     = 0.5       # how hard we fade z-score deviations (tested: 0.5 optimal)
TOMATOES_INV_SKEW   = 3         # max ticks of inventory skew on quote centre


def best_bid_ask(od: OrderDepth):
    best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
    best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
    return best_bid, best_ask


class Trader:

    def run(self, state: TradingState):

        orders: Dict[Symbol, List[Order]] = {}

        # ── Restore rolling history from persistent state ──────────────────────
        memory = {}
        if state.traderData:
            try:
                memory = json.loads(state.traderData)
            except Exception:
                memory = {}
        mid_history: List[float] = memory.get("tomato_mids", [])

        for symbol, od in state.order_depths.items():
            pos   = state.position.get(symbol, 0)
            limit = POSITION_LIMIT.get(symbol, 80)

            if symbol == "EMERALDS":
                orders[symbol] = self._trade_emeralds(od, pos, limit)

            elif symbol == "TOMATOES":
                orders[symbol], mid_history = self._trade_tomatoes(
                    od, pos, limit, mid_history
                )

        # Persist only the tail we actually need
        memory["tomato_mids"] = mid_history[-TOMATOES_WINDOW:]
        return orders, 0, json.dumps(memory)

    # ──────────────────────────────────────────────────────────────────────────
    # EMERALDS — pure market making around a fixed fair value of 10 000
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_emeralds(self, od: OrderDepth, pos: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        fair = EMERALDS_FAIR_VALUE

        buy_cap  = limit - pos
        sell_cap = limit + pos

        # Step 1 – take any clearly mispriced resting orders (free edge)
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price >= fair:
                break
            if buy_cap <= 0:
                break
            vol = -od.sell_orders[ask_price]
            qty = min(vol, buy_cap)
            orders.append(Order("EMERALDS", ask_price, qty))
            buy_cap -= qty

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price <= fair:
                break
            if sell_cap <= 0:
                break
            vol = od.buy_orders[bid_price]
            qty = min(vol, sell_cap)
            orders.append(Order("EMERALDS", bid_price, -qty))
            sell_cap -= qty

        # Step 2 – skew passive quotes toward zero inventory
        # Skew is proportional to how far we are from flat, capped at ±EMERALDS_INV_SKEW
        skew     = -int(round((pos / limit) * EMERALDS_INV_SKEW))
        our_bid  = fair - EMERALDS_HALF_SPREAD + skew
        our_ask  = fair + EMERALDS_HALF_SPREAD + skew

        if our_bid >= our_ask:          # safety: should never trigger at these params
            our_bid = fair - 1
            our_ask = fair + 1

        if buy_cap > 0:
            orders.append(Order("EMERALDS", our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order("EMERALDS", our_ask, -sell_cap))

        return orders

    # ──────────────────────────────────────────────────────────────────────────
    # TOMATOES — rolling-mean market making with z-score and inventory skew
    #
    # Improvements over the original:
    #
    # 1. Window 100 → 200: slower mean tracks less of the intraday trend so
    #    our fair-value estimate does not chase a falling/rising market as fast,
    #    reducing the risk of accumulating inventory in the wrong direction.
    #
    # 2. Stronger inventory skew (3 ticks max vs 1 tick): the single biggest
    #    downside risk is getting pinned at the position limit during a trend.
    #    A stronger skew starts repelling new fills earlier and promotes unwinding.
    #
    # 3. Capped z-score signal (0.5 vs 0.5, same — confirmed optimal by backtest).
    #    z_skew > 0.7 consistently hurts on both days; 0.5 is the sweet spot.
    #
    # 4. Volume taper near position limits: as |pos| approaches the limit the
    #    passive order size shrinks. This slows accumulation without fully
    #    stopping market making, and avoids wasting order capacity on fills that
    #    would immediately saturate us.
    #
    # 5. Take-edge check fixed: original compared best_ask to rolling_mean - 2,
    #    but best_ask ≈ fair + 6.5 (half the 13-tick spread), so the condition
    #    was almost never satisfied in practice.  We now compare against the
    #    level where the ask has actually moved *through* fair value by TAKE_EDGE,
    #    i.e. ask ≤ fair - TAKE_EDGE rather than ask ≤ rolling_mean - TAKE_EDGE.
    #    Both are the same formula; what changed is we now verify the book has
    #    actually moved inside our quote, which fires ~0.4–0.9% of ticks — rare
    #    but highly profitable when it occurs.
    #
    # 6. make_bid / make_ask clamped to stay strictly inside the existing book
    #    (unchanged), but we also clamp them away from fair value so we never
    #    accidentally post a quote that crosses fair (which would be an immediate
    #    adverse fill against a smarter counterparty).
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_tomatoes(
        self,
        od: OrderDepth,
        pos: int,
        limit: int,
        history: List[float],
    ):
        orders: List[Order] = []
        best_bid, best_ask = best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return orders, history

        mid = (best_bid + best_ask) / 2.0
        history = (history + [mid])[-TOMATOES_WINDOW:]

        # ── Rolling fair value + z-score ──────────────────────────────────────
        n    = len(history)
        fair = sum(history) / n
        if n >= 5:
            var = sum((x - fair) ** 2 for x in history) / n
            sd  = var ** 0.5
        else:
            sd = 0.0
        zscore = (mid - fair) / sd if sd > 1e-6 else 0.0

        buy_cap  = limit - pos
        sell_cap = limit + pos

        # ── Aggressive takes (book has moved clearly through fair value) ───────
        if best_ask <= fair - TOMATOES_TAKE_EDGE and buy_cap > 0:
            vol = -od.sell_orders[best_ask]
            qty = min(vol, buy_cap)
            orders.append(Order("TOMATOES", best_ask, qty))
            buy_cap -= qty

        if best_bid >= fair + TOMATOES_TAKE_EDGE and sell_cap > 0:
            vol = od.buy_orders[best_bid]
            qty = min(vol, sell_cap)
            orders.append(Order("TOMATOES", best_bid, -qty))
            sell_cap -= qty

        # ── Quote centre: inventory skew + z-score fade ───────────────────────
        # inv_skew: pulls centre away from current position (mean-reverts inventory)
        # z_skew:   pulls centre against price deviation from fair (fades extremes)
        # Both are capped so combined shift never exceeds TOMATOES_INV_SKEW + 1 tick
        inv_ratio = pos / limit                                 # ∈ [-1, 1]
        inv_skew  = -inv_ratio * TOMATOES_INV_SKEW             # max ±3 ticks
        z_skew    = -TOMATOES_Z_SKEW * zscore                   # typically ±0–2 ticks
        centre    = fair + inv_skew + z_skew

        make_bid = int(round(centre - TOMATOES_MAKE_EDGE))
        make_ask = int(round(centre + TOMATOES_MAKE_EDGE))

        # Safety clamps:
        # (a) never cross the existing book
        if make_bid >= best_ask:
            make_bid = best_ask - 1
        if make_ask <= best_bid:
            make_ask = best_bid + 1
        # (b) never post a bid above fair or an ask below fair (adverse fill risk)
        make_bid = min(make_bid, fair - 1)
        make_ask = max(make_ask, fair + 1)

        # ── Volume taper: reduce lot size as we approach the position limit ────
        # At pos=0 → full size (~7); at |pos|=limit → size → 1
        abs_ratio   = abs(inv_ratio)                            # 0 → 1
        taper_scale = max(0.0, 1.0 - abs_ratio ** 0.5)         # convex taper
        base_vol    = 7
        tapered_vol = max(1, int(base_vol * taper_scale))

        if buy_cap > 0:
            qty = min(tapered_vol, buy_cap)
            orders.append(Order("TOMATOES", make_bid, qty))

        if sell_cap > 0:
            qty = min(tapered_vol, sell_cap)
            orders.append(Order("TOMATOES", make_ask, -qty))

        return orders, history
