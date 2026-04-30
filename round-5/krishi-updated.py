"""IMC Prosperity Round 5 — krishi-updated strategy.

Improvements over krishi.py based on day-4 log analysis (PnL +$26,951.55):

  DIAGNOSIS: every losing product had POSITIVE per-fill edge vs mid:
    GALAXY_DARK_MATTER       +$383 edge → -$3,344 PnL  (mid drift -407)
    GALAXY_PLANETARY_RINGS   +$399 edge → -$3,281 PnL  (mid drift -778)
    TRANSLATOR_SPACE_GRAY    +$222 edge → -$4,234 PnL  (mid drift -669)
    SLEEP_POD_LAMB_WOOL      +$231 edge → -$3,550 PnL  (mid drift -591)
  Winners showed the same +edge but with positive drift.
  The symmetric microprice MM kept rebuilding long inventory into the
  one-sided down-moves: pure adverse selection on trending tape.

  FIXES (priority-ordered by expected impact):

    1) Trend-aware FV (the main fix).
       Maintain per-product EWMA mid (alpha=0.06 ≈ span 30 ticks).
       trend = mid - ewma; bias FV by +TREND_GAIN * trend so the FV
       leads in the direction of drift. On a down-trend, FV is pulled
       BELOW mid → our buy quote sits below the falling mid → fills
       stop, inventory drains via the unwind side.

    2) Active position-flatten at |pos| >= 8.
       Original code only widened the accumulating side by 1 tick
       (insufficient on trending tape). New: when long-and-stuck
       AND best_bid >= adjusted_fv - 1, hit the bid for (pos - 5)
       lots; symmetric on the short side. We accept the half-spread
       cost in exchange for not riding the trend further.

    3) Plain mid replaces microprice as FV base.
       Microprice (volume-weighted) gets pulled by depth; when bid-side
       depth thins ahead of a leg-down it ticks UP right before the
       decline. (bb+ba)/2 is unbiased.

  UNCHANGED from krishi.py (these worked on day 4):
    - PEBBLES basket arb (residual ≥ 4, 5 lots/leg, SKIP {M, L})
    - PEBBLES per-leg MM on S only (SKIP_PEBBLE_PER_LEG = {XS, M, L, XL})
    - SKIP_MM = {PEBBLES_M}
    - Stoikov 0.15 ticks/lot inventory skew
    - PEBBLE_BASKET_QTY_PER_LEG = 5 (live-validated; QTY=10 was a
      backtest artifact)

Position limit: 10 per product across all 50 products.
"""
import json
from datamodel import TradingState, Order
from typing import Dict, List


class Trader:
    POS_LIMIT = 10

    PEBBLES = ['PEBBLES_XS', 'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L', 'PEBBLES_XL']
    PEBBLES_SUM = 50_000

    PEBBLE_BASKET_THR = 4
    PEBBLE_BASKET_QTY_PER_LEG = 5

    PEBBLE_TAKE_EDGE = 3
    PEBBLE_QUOTE_EDGE = 3

    SKIP_PEBBLE_BASKET = {'PEBBLES_M', 'PEBBLES_L'}
    SKIP_PEBBLE_PER_LEG = {'PEBBLES_M', 'PEBBLES_XS', 'PEBBLES_L', 'PEBBLES_XL'}
    SKIP_MM = {'PEBBLES_M'}

    ALL_PRODUCTS = [
        'GALAXY_SOUNDS_DARK_MATTER', 'GALAXY_SOUNDS_BLACK_HOLES',
        'GALAXY_SOUNDS_PLANETARY_RINGS', 'GALAXY_SOUNDS_SOLAR_WINDS',
        'GALAXY_SOUNDS_SOLAR_FLAMES',
        'SLEEP_POD_SUEDE', 'SLEEP_POD_LAMB_WOOL', 'SLEEP_POD_POLYESTER',
        'SLEEP_POD_NYLON', 'SLEEP_POD_COTTON',
        'MICROCHIP_CIRCLE', 'MICROCHIP_OVAL', 'MICROCHIP_SQUARE',
        'MICROCHIP_RECTANGLE', 'MICROCHIP_TRIANGLE',
        'PEBBLES_XS', 'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L', 'PEBBLES_XL',
        'ROBOT_VACUUMING', 'ROBOT_MOPPING', 'ROBOT_DISHES',
        'ROBOT_LAUNDRY', 'ROBOT_IRONING',
        'UV_VISOR_YELLOW', 'UV_VISOR_AMBER', 'UV_VISOR_ORANGE',
        'UV_VISOR_RED', 'UV_VISOR_MAGENTA',
        'TRANSLATOR_SPACE_GRAY', 'TRANSLATOR_ASTRO_BLACK',
        'TRANSLATOR_ECLIPSE_CHARCOAL', 'TRANSLATOR_GRAPHITE_MIST',
        'TRANSLATOR_VOID_BLUE',
        'PANEL_1X2', 'PANEL_2X2', 'PANEL_1X4', 'PANEL_2X4', 'PANEL_4X4',
        'OXYGEN_SHAKE_MORNING_BREATH', 'OXYGEN_SHAKE_EVENING_BREATH',
        'OXYGEN_SHAKE_MINT', 'OXYGEN_SHAKE_CHOCOLATE', 'OXYGEN_SHAKE_GARLIC',
        'SNACKPACK_CHOCOLATE', 'SNACKPACK_VANILLA', 'SNACKPACK_PISTACHIO',
        'SNACKPACK_STRAWBERRY', 'SNACKPACK_RASPBERRY',
    ]

    DEFAULT_SKEW = 0.15

    # --- Mean-reversion FV parameters ---
    # EWMA span ~30 (alpha ≈ 0.065).
    # NEGATIVE GAIN: FV is pulled BACK toward the EWMA. Round-5-updated
    # ran GAIN = +0.5 first and lost $293k on day-4 replay with 4.3x
    # the trade volume — every fill landed just before a reversal. The
    # tape mean-reverts on per-tick scale. So:
    #   mid below ewma → fv > mid → bid the dip (buy)
    #   mid above ewma → fv < mid → ask the rip (sell)
    EWMA_ALPHA = 0.065
    TREND_GAIN = -0.5

    # --- Active flatten ---
    # When |pos| >= FLATTEN_TRIGGER and the inside book has crossed our
    # skewed FV, hit it down to FLATTEN_TARGET. Lower trigger → more
    # cost, less trend exposure. 8/5 means we accept the half-spread on
    # 3 lots to deleverage from limit toward neutral.
    FLATTEN_TRIGGER = 8
    FLATTEN_TARGET = 5

    def _load_state(self, state: TradingState) -> dict:
        if state.traderData:
            try:
                return json.loads(state.traderData)
            except Exception:
                return {}
        return {}

    def run(self, state: TradingState):
        mem = self._load_state(state)
        ewmas = mem.get('ewma', {})
        result: Dict[str, List[Order]] = {}

        # Update EWMAs for every product before any logic uses them.
        for prod, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                continue
            mid = (max(od.buy_orders) + min(od.sell_orders)) / 2
            prev = ewmas.get(prod, mid)
            ewmas[prod] = prev + self.EWMA_ALPHA * (mid - prev)
        mem['ewma'] = ewmas

        # 1) PEBBLES basket arb + per-leg MM
        pebble_orders = self._pebble_arb(state, ewmas)
        for p, orders in pebble_orders.items():
            if orders:
                result[p] = orders

        # 2) Standard MM for everything else
        for prod in self.ALL_PRODUCTS:
            if prod in result:
                continue
            if prod not in state.order_depths:
                continue
            if prod in self.SKIP_MM:
                continue
            result[prod] = self._market_make(state, prod, ewmas)

        return result, 0, json.dumps(mem)

    # ------------------------------------------------------------------ pebbles

    def _pebble_arb(self, state: TradingState, ewmas: dict) -> Dict[str, List[Order]]:
        out: Dict[str, List[Order]] = {p: [] for p in self.PEBBLES}

        mids: Dict[str, float] = {}
        ods = {}
        bbs: Dict[str, int] = {}
        bas: Dict[str, int] = {}
        for p in self.PEBBLES:
            if p not in state.order_depths:
                return out
            od = state.order_depths[p]
            if not od.buy_orders or not od.sell_orders:
                return out
            bb = max(od.buy_orders.keys())
            ba = min(od.sell_orders.keys())
            mids[p] = (bb + ba) / 2
            ods[p] = od
            bbs[p] = bb
            bas[p] = ba

        residual = sum(mids.values()) - self.PEBBLES_SUM

        # === Mode A: basket trade ===
        if residual >= self.PEBBLE_BASKET_THR:
            for p in self.PEBBLES:
                if p in self.SKIP_PEBBLE_BASKET:
                    continue
                position = state.position.get(p, 0)
                sell_cap = self.POS_LIMIT + position
                bid_vol = ods[p].buy_orders[bbs[p]]
                qty = min(sell_cap, self.PEBBLE_BASKET_QTY_PER_LEG, bid_vol)
                if qty > 0:
                    out[p].append(Order(p, bbs[p], -qty))
        elif residual <= -self.PEBBLE_BASKET_THR:
            for p in self.PEBBLES:
                if p in self.SKIP_PEBBLE_BASKET:
                    continue
                position = state.position.get(p, 0)
                buy_cap = self.POS_LIMIT - position
                ask_vol = -ods[p].sell_orders[bas[p]]
                qty = min(buy_cap, self.PEBBLE_BASKET_QTY_PER_LEG, ask_vol)
                if qty > 0:
                    out[p].append(Order(p, bas[p], qty))

        # === Mode B: per-leg MM (PEBBLES_S only) ===
        for p in self.PEBBLES:
            if p in self.SKIP_PEBBLE_PER_LEG:
                continue
            others_sum = sum(mids[q] for q in self.PEBBLES if q != p)
            implied_fv = self.PEBBLES_SUM - others_sum
            # Trend bias on the per-leg implied FV too.
            ewma = ewmas.get(p, mids[p])
            trend = mids[p] - ewma
            implied_fv = implied_fv + self.TREND_GAIN * trend
            fv_int = int(round(implied_fv))

            position = state.position.get(p, 0)
            in_flight = sum(o.quantity for o in out[p])
            effective_pos = position + in_flight
            buy_cap = self.POS_LIMIT - effective_pos
            sell_cap = self.POS_LIMIT + effective_pos

            od = ods[p]
            best_bid = bbs[p]
            best_ask = bas[p]

            if buy_cap > 0:
                for ask_price in sorted(od.sell_orders.keys()):
                    if ask_price <= fv_int - self.PEBBLE_TAKE_EDGE:
                        qty = min(-od.sell_orders[ask_price], buy_cap)
                        if qty > 0:
                            out[p].append(Order(p, ask_price, qty))
                            buy_cap -= qty
                    else:
                        break
            if sell_cap > 0:
                for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                    if bid_price >= fv_int + self.PEBBLE_TAKE_EDGE:
                        qty = min(od.buy_orders[bid_price], sell_cap)
                        if qty > 0:
                            out[p].append(Order(p, bid_price, -qty))
                            sell_cap -= qty
                    else:
                        break

            skew = effective_pos * self.DEFAULT_SKEW
            fv_skewed = fv_int - int(round(skew))

            quote_bid = fv_skewed - self.PEBBLE_QUOTE_EDGE
            quote_ask = fv_skewed + self.PEBBLE_QUOTE_EDGE

            quote_bid = min(quote_bid, best_ask - 1)
            quote_ask = max(quote_ask, best_bid + 1)
            if quote_bid >= quote_ask:
                quote_bid = fv_int - 1
                quote_ask = fv_int + 1

            if buy_cap > 0:
                out[p].append(Order(p, quote_bid, buy_cap))
            if sell_cap > 0:
                out[p].append(Order(p, quote_ask, -sell_cap))

        return out

    # ---------------------------------------------------------------- MM logic

    def _market_make(self, state: TradingState, prod: str, ewmas: dict) -> List[Order]:
        od = state.order_depths[prod]
        if not od.buy_orders or not od.sell_orders:
            return []

        position = state.position.get(prod, 0)

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2

        # Trend-aware FV: continuation of recent drift.
        ewma = ewmas.get(prod, mid)
        trend = mid - ewma
        fv = mid + self.TREND_GAIN * trend

        # Stoikov inventory skew on top of the trend-adjusted FV.
        skew = position * self.DEFAULT_SKEW
        fv_skewed = fv - skew
        fv_int = int(round(fv_skewed))

        orders: List[Order] = []
        buy_cap = self.POS_LIMIT - position
        sell_cap = self.POS_LIMIT + position

        # === TAKE: cross when prices cross our trend-skewed FV ===
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price <= fv_int and buy_cap > 0:
                qty = min(-od.sell_orders[ask_price], buy_cap)
                if qty > 0:
                    orders.append(Order(prod, ask_price, qty))
                    buy_cap -= qty
            else:
                break

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price >= fv_int and sell_cap > 0:
                qty = min(od.buy_orders[bid_price], sell_cap)
                if qty > 0:
                    orders.append(Order(prod, bid_price, -qty))
                    sell_cap -= qty
            else:
                break

        # === ACTIVE FLATTEN: when stuck near the limit and the book is
        #     friendly, eat the half-spread to escape the inventory before
        #     the trend extends further. This is the day-4 fix — passive
        #     widening was not enough to drain stuck longs into a -700
        #     mid-drift. ===
        if position >= self.FLATTEN_TRIGGER:
            unwind_qty = position - self.FLATTEN_TARGET
            if unwind_qty > 0 and best_bid >= fv_int - 1:
                bid_avail = od.buy_orders[best_bid]
                qty = min(unwind_qty, bid_avail, sell_cap)
                if qty > 0:
                    orders.append(Order(prod, best_bid, -qty))
                    sell_cap -= qty
        elif position <= -self.FLATTEN_TRIGGER:
            unwind_qty = -position - self.FLATTEN_TARGET
            if unwind_qty > 0 and best_ask <= fv_int + 1:
                ask_avail = -od.sell_orders[best_ask]
                qty = min(unwind_qty, ask_avail, buy_cap)
                if qty > 0:
                    orders.append(Order(prod, best_ask, qty))
                    buy_cap -= qty

        # === QUOTE: penny inside the spread ===
        if spread > 1:
            quote_bid = best_bid + 1
            quote_ask = best_ask - 1
        else:
            quote_bid = best_bid
            quote_ask = best_ask

        # Position-limit safety: when at the edge, keep the accumulating
        # side passive so we don't pile in further while the active flatten
        # above does the work.
        if position >= self.POS_LIMIT - 2:
            quote_bid = best_bid - 1
        elif position <= -(self.POS_LIMIT - 2):
            quote_ask = best_ask + 1

        if quote_bid >= quote_ask:
            quote_bid = best_bid
            quote_ask = best_ask

        if buy_cap > 0:
            orders.append(Order(prod, quote_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(prod, quote_ask, -sell_cap))

        return orders
