"""Round-5 strategy — revamped.

Rebuilt on top of the v1 (round-5.py) by:
  - fixing the position limit (50 -> 10; correct exchange limit),
  - dropping the two data-fitted layers (SNACKPACK CHOC/VAN z-score and
    the UV_VISOR_AMBER cointegration pairs) — both showed in-sample
    edge but no live monetization on day 4,
  - replacing the simple MM with a trend-aware microprice MM borrowed
    and improved from krishi.py (which scored +$26.9k on day 4),
  - replacing the basket-only PEBBLES recipe with krishi's residual +
    per-leg MM hybrid,
  - keeping the MICROCHIP buy-bias recipe (it's grounded in the EDA
    finding that XIRECS is a +7% net buyer, not a fitted threshold).

Layers:

  A. PEBBLES (mechanical, +$5,004 on day 4 in krishi).
     PEBBLES_XS+S+M+L+XL ≡ 50,000. Two modes:
       (i)  basket trade — when |sum(mids) - 50000| ≥ 4 ticks, fire all
            5 legs same direction (capped to high-σ legs that drive
            the residual: skip M+L);
       (ii) per-leg MM on PEBBLES_S only (mid-σ leg with the cleanest
            implied-FV signal and tightest spread).
     Per-leg lots = 5; QTY=10 was a backtest artifact that lost on
     PEBBLES_XL live.

  B. MICROCHIP buy-pressure quoting.
     XIRECS is consistently a buyer above mid on this family. Skewed
     edges (tight ask, wider bid) so we capture the lift without
     paying for the wider bid ourselves.

  C. Trend-aware MM for everything else (43 products, was 33).
     Plain mid as FV base + EWMA trend bias + Stoikov inventory skew +
     penny-inside quote + cross-take when book crosses skewed FV +
     active-flatten when |pos| ≥ 8.

REMOVED (failed to monetize on day 4):
  • SNACKPACK CHOC/VAN z-score pair (data-fitted thresholds).
  • UV_VISOR_AMBER cross-cluster cointegration (data-fitted, $0 live).

Position limit: 10 per product across all 50 products.
"""
import json
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple


# ============================================================
#  CONSTANTS
# ============================================================

POS_LIMIT = 10

# === PEBBLES ===
PEBBLES_LEGS = ['PEBBLES_XS', 'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L', 'PEBBLES_XL']
PEBBLES_TARGET_SUM = 50_000
PEBBLE_BASKET_THR = 4              # residual ticks; ~1.4σ on observed std=2.8
PEBBLE_BASKET_QTY_PER_LEG = 5      # live-validated; QTY=10 lost on XL day 4
PEBBLE_TAKE_EDGE = 3
PEBBLE_QUOTE_EDGE = 3
SKIP_PEBBLE_BASKET = {'PEBBLES_M', 'PEBBLES_L'}      # low-σ legs trade noise
SKIP_PEBBLE_PER_LEG = {'PEBBLES_M', 'PEBBLES_XS',    # only S has clean signal
                       'PEBBLES_L', 'PEBBLES_XL'}
SKIP_MM = {'PEBBLES_M'}             # diagnose_robust passed all 3 gates

# === MICROCHIP ===
MICROCHIP_PRODUCTS = {
    'MICROCHIP_CIRCLE', 'MICROCHIP_OVAL', 'MICROCHIP_RECTANGLE',
    'MICROCHIP_SQUARE', 'MICROCHIP_TRIANGLE',
}
MICROCHIP_BID_EDGE = 3              # wider bid (don't accumulate longs)
MICROCHIP_ASK_EDGE = 1              # tight ask (lift the bot's buy pressure)

# === MM ===
DEFAULT_BID_EDGE = 2
DEFAULT_ASK_EDGE = 2
INVENTORY_SKEW = 0.15               # Stoikov reservation-price shift / lot

# Trend-aware FV. EWMA span ~30 ticks captures the persistent 500-800
# tick drifts that dominated day-4 losses without overfitting noise.
EWMA_ALPHA = 0.065
TREND_GAIN = 0.5                    # FV = mid + GAIN*(mid - ewma); 0.5 is
                                    # defensive (half-pull toward extrapolation)

# Active flatten: when stuck near the limit and the book is friendly,
# eat the half-spread to escape inventory before the trend extends.
FLATTEN_TRIGGER = 8
FLATTEN_TARGET = 5

ALL_PRODUCTS = [
    'GALAXY_SOUNDS_BLACK_HOLES', 'GALAXY_SOUNDS_DARK_MATTER',
    'GALAXY_SOUNDS_PLANETARY_RINGS', 'GALAXY_SOUNDS_SOLAR_FLAMES',
    'GALAXY_SOUNDS_SOLAR_WINDS',
    'MICROCHIP_CIRCLE', 'MICROCHIP_OVAL', 'MICROCHIP_RECTANGLE',
    'MICROCHIP_SQUARE', 'MICROCHIP_TRIANGLE',
    'OXYGEN_SHAKE_CHOCOLATE', 'OXYGEN_SHAKE_EVENING_BREATH',
    'OXYGEN_SHAKE_GARLIC', 'OXYGEN_SHAKE_MINT', 'OXYGEN_SHAKE_MORNING_BREATH',
    'PANEL_1X2', 'PANEL_1X4', 'PANEL_2X2', 'PANEL_2X4', 'PANEL_4X4',
    'PEBBLES_L', 'PEBBLES_M', 'PEBBLES_S', 'PEBBLES_XL', 'PEBBLES_XS',
    'ROBOT_DISHES', 'ROBOT_IRONING', 'ROBOT_LAUNDRY', 'ROBOT_MOPPING',
    'ROBOT_VACUUMING',
    'SLEEP_POD_COTTON', 'SLEEP_POD_LAMB_WOOL', 'SLEEP_POD_NYLON',
    'SLEEP_POD_POLYESTER', 'SLEEP_POD_SUEDE',
    'SNACKPACK_CHOCOLATE', 'SNACKPACK_PISTACHIO', 'SNACKPACK_RASPBERRY',
    'SNACKPACK_STRAWBERRY', 'SNACKPACK_VANILLA',
    'TRANSLATOR_ASTRO_BLACK', 'TRANSLATOR_ECLIPSE_CHARCOAL',
    'TRANSLATOR_GRAPHITE_MIST', 'TRANSLATOR_SPACE_GRAY', 'TRANSLATOR_VOID_BLUE',
    'UV_VISOR_AMBER', 'UV_VISOR_MAGENTA', 'UV_VISOR_ORANGE',
    'UV_VISOR_RED', 'UV_VISOR_YELLOW',
]


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

    @staticmethod
    def _bb_ba(od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        bb = max(od.buy_orders) if od and od.buy_orders else None
        ba = min(od.sell_orders) if od and od.sell_orders else None
        return bb, ba

    # ====================================================================
    # ===== RECIPE A: PEBBLES — residual basket arb + per-leg MM =========
    # ====================================================================

    def _trade_pebbles(self, state: TradingState, ewmas: dict) -> Dict[str, List[Order]]:
        """Mode A: basket trade on |residual| ≥ THR.
        Mode B: per-leg passive MM around implied FV (PEBBLES_S only).
        """
        out: Dict[str, List[Order]] = {p: [] for p in PEBBLES_LEGS}

        mids: Dict[str, float] = {}
        ods: Dict[str, OrderDepth] = {}
        bbs: Dict[str, int] = {}
        bas: Dict[str, int] = {}
        for p in PEBBLES_LEGS:
            od = state.order_depths.get(p)
            if od is None or not od.buy_orders or not od.sell_orders:
                return out
            bb, ba = self._bb_ba(od)
            mids[p] = (bb + ba) / 2
            ods[p] = od
            bbs[p] = bb
            bas[p] = ba

        residual = sum(mids.values()) - PEBBLES_TARGET_SUM

        # --- Mode A ---
        if residual >= PEBBLE_BASKET_THR:
            for p in PEBBLES_LEGS:
                if p in SKIP_PEBBLE_BASKET:
                    continue
                pos = state.position.get(p, 0)
                sell_cap = POS_LIMIT + pos
                bid_vol = ods[p].buy_orders[bbs[p]]
                qty = min(sell_cap, PEBBLE_BASKET_QTY_PER_LEG, bid_vol)
                if qty > 0:
                    out[p].append(Order(p, bbs[p], -qty))
        elif residual <= -PEBBLE_BASKET_THR:
            for p in PEBBLES_LEGS:
                if p in SKIP_PEBBLE_BASKET:
                    continue
                pos = state.position.get(p, 0)
                buy_cap = POS_LIMIT - pos
                ask_vol = -ods[p].sell_orders[bas[p]]
                qty = min(buy_cap, PEBBLE_BASKET_QTY_PER_LEG, ask_vol)
                if qty > 0:
                    out[p].append(Order(p, bas[p], qty))

        # --- Mode B (PEBBLES_S only) ---
        for p in PEBBLES_LEGS:
            if p in SKIP_PEBBLE_PER_LEG:
                continue
            others = sum(mids[q] for q in PEBBLES_LEGS if q != p)
            implied_fv = PEBBLES_TARGET_SUM - others
            ewma = ewmas.get(p, mids[p])
            implied_fv += TREND_GAIN * (mids[p] - ewma)
            fv_int = int(round(implied_fv))

            pos = state.position.get(p, 0)
            in_flight = sum(o.quantity for o in out[p])
            eff = pos + in_flight
            buy_cap = POS_LIMIT - eff
            sell_cap = POS_LIMIT + eff
            od = ods[p]

            # Take on extreme single-leg mispricing
            if buy_cap > 0:
                for ask in sorted(od.sell_orders):
                    if ask <= fv_int - PEBBLE_TAKE_EDGE:
                        qty = min(-od.sell_orders[ask], buy_cap)
                        if qty > 0:
                            out[p].append(Order(p, ask, qty))
                            buy_cap -= qty
                    else:
                        break
            if sell_cap > 0:
                for bid in sorted(od.buy_orders, reverse=True):
                    if bid >= fv_int + PEBBLE_TAKE_EDGE:
                        qty = min(od.buy_orders[bid], sell_cap)
                        if qty > 0:
                            out[p].append(Order(p, bid, -qty))
                            sell_cap -= qty
                    else:
                        break

            skew = eff * INVENTORY_SKEW
            fv_skewed = fv_int - int(round(skew))
            qb = min(fv_skewed - PEBBLE_QUOTE_EDGE, bas[p] - 1)
            qa = max(fv_skewed + PEBBLE_QUOTE_EDGE, bbs[p] + 1)
            if qb >= qa:
                qb, qa = fv_int - 1, fv_int + 1

            if buy_cap > 0:
                out[p].append(Order(p, qb, buy_cap))
            if sell_cap > 0:
                out[p].append(Order(p, qa, -sell_cap))

        return out

    # ====================================================================
    # ===== RECIPE B: MICROCHIP buy-pressure quoting =====================
    # ====================================================================

    def _trade_microchip(self, state: TradingState, sym: str, ewmas: dict) -> List[Order]:
        return self._mm_core(
            state, sym, ewmas,
            bid_edge=MICROCHIP_BID_EDGE,
            ask_edge=MICROCHIP_ASK_EDGE,
        )

    # ====================================================================
    # ===== RECIPE C: Trend-aware MM core ================================
    # ====================================================================

    def _mm_core(self, state: TradingState, prod: str, ewmas: dict,
                 bid_edge: int = DEFAULT_BID_EDGE,
                 ask_edge: int = DEFAULT_ASK_EDGE) -> List[Order]:
        od = state.order_depths.get(prod)
        if od is None or not od.buy_orders or not od.sell_orders:
            return []

        pos = state.position.get(prod, 0)
        bb = max(od.buy_orders)
        ba = min(od.sell_orders)
        spread = ba - bb
        mid = (bb + ba) / 2

        # Trend-aware FV (the day-4 fix). Plain mid + continuation of the
        # short-window drift. Microprice was REJECTED here because it
        # gets pulled by depth — bid-side depth thins ahead of leg-downs
        # so microprice ticks up just before the price falls.
        ewma = ewmas.get(prod, mid)
        fv = mid + TREND_GAIN * (mid - ewma)

        # Stoikov inventory skew on top of the trend-adjusted FV.
        fv_skewed = fv - pos * INVENTORY_SKEW
        fv_int = int(round(fv_skewed))

        orders: List[Order] = []
        buy_cap = POS_LIMIT - pos
        sell_cap = POS_LIMIT + pos

        # --- TAKE: cross when book crosses skewed FV ---
        for ask in sorted(od.sell_orders):
            if ask <= fv_int and buy_cap > 0:
                qty = min(-od.sell_orders[ask], buy_cap)
                if qty > 0:
                    orders.append(Order(prod, ask, qty))
                    buy_cap -= qty
            else:
                break
        for bid in sorted(od.buy_orders, reverse=True):
            if bid >= fv_int and sell_cap > 0:
                qty = min(od.buy_orders[bid], sell_cap)
                if qty > 0:
                    orders.append(Order(prod, bid, -qty))
                    sell_cap -= qty
            else:
                break

        # --- ACTIVE FLATTEN: deleverage when stuck near limit ---
        if pos >= FLATTEN_TRIGGER:
            unwind = pos - FLATTEN_TARGET
            if unwind > 0 and bb >= fv_int - 1:
                qty = min(unwind, od.buy_orders[bb], sell_cap)
                if qty > 0:
                    orders.append(Order(prod, bb, -qty))
                    sell_cap -= qty
        elif pos <= -FLATTEN_TRIGGER:
            unwind = -pos - FLATTEN_TARGET
            if unwind > 0 and ba <= fv_int + 1:
                qty = min(unwind, -od.sell_orders[ba], buy_cap)
                if qty > 0:
                    orders.append(Order(prod, ba, qty))
                    buy_cap -= qty

        # --- QUOTE: penny inside on tight markets, edge-based otherwise ---
        if spread > 1:
            quote_bid = max(bb + 1, fv_int - bid_edge)
            quote_ask = min(ba - 1, fv_int + ask_edge)
        else:
            quote_bid = bb
            quote_ask = ba

        # Position-limit safety: keep accumulating side passive at the edge.
        if pos >= POS_LIMIT - 2:
            quote_bid = bb - 1
        elif pos <= -(POS_LIMIT - 2):
            quote_ask = ba + 1

        if quote_bid >= quote_ask:
            quote_bid, quote_ask = bb, ba

        if buy_cap > 0:
            orders.append(Order(prod, quote_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(prod, quote_ask, -sell_cap))

        return orders

    # ====================================================================
    # ===== MAIN ENTRY ====================================================
    # ====================================================================

    def run(self, state: TradingState):
        mem = self._load(state.traderData)
        ewmas = mem.get('ewma', {})

        # Update EWMAs first so all layers see the same up-to-date trend.
        for prod, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                continue
            mid = (max(od.buy_orders) + min(od.sell_orders)) / 2
            prev = ewmas.get(prod, mid)
            ewmas[prod] = prev + EWMA_ALPHA * (mid - prev)
        mem['ewma'] = ewmas

        result: Dict[str, List[Order]] = {}

        # Recipe A — PEBBLES
        for sym, orders in self._trade_pebbles(state, ewmas).items():
            if orders:
                result[sym] = orders

        # Recipe B — MICROCHIP buy-pressure quoting
        for sym in MICROCHIP_PRODUCTS:
            if sym in result or sym not in state.order_depths:
                continue
            orders = self._trade_microchip(state, sym, ewmas)
            if orders:
                result[sym] = orders

        # Recipe C — Trend-aware MM for everything else
        for sym in ALL_PRODUCTS:
            if sym in result or sym in SKIP_MM:
                continue
            if sym in MICROCHIP_PRODUCTS:
                continue
            if sym not in state.order_depths:
                continue
            orders = self._mm_core(state, sym, ewmas)
            if orders:
                result[sym] = orders

        return result, 0, json.dumps(mem)