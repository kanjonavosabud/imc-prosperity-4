"""Round-4 HYDROGEL_PACK-only strategy.

Same core MM mechanics as the integrated round-4.py HP path, plus explicit
A/B test knobs based on R4 EDA findings. Defaults match current round-4.py
behavior so the baseline PnL is unchanged — flip ONE knob at a time to test.


# ============================================================================
# A/B TEST KNOBS (priority order from EDA recommendations)
# ============================================================================

  [B] EDGE                — passive half-spread.   Baseline 12.   Test: 14, 16.
      R4 HP std is ~15% wider than R3 (34.6 vs ~30); wider edge captures more
      $/fill at the cost of fewer fills.  Note: the spread-clamp
      `my_ask = best_ask - 1` may dominate when the book is tight, so EDGE
      mostly takes effect when the NPC quotes spread wide.

  [A] LT_MEAN_BLEND       — OU mean-reversion weight. Baseline 0.30.
      Test: 0.10, 0.20, 0.40.   Comment in original code says "peak via local
      sweep" but that may have been on R3 data.  Worth re-sweeping on R4.

  [D] M38_TIGHTEN_AMOUNT  — extra ticks of price improvement when Mark 38
      traded HP recently.  Baseline 0 (off).   Test: 1.
      Hypothesis: M38 is uninformed on HP (post-fill mid move ±0.06 to ±0.12,
      n>500 in R4 historical) AND pays +8.5 average half-spread (highest in
      the dataset).  Tighter quotes during M38-active windows should win
      more of their flow with low adverse-selection cost.

  [E] OBI_EXTRA_TILT      — extra weight on L1 OBI beyond what L1 microprice
      already encodes.  Baseline 0.0 (off).   Test: 1.0, 2.0.
      L1 microprice already applies an implicit OBI tilt of
      `half_spread × OBI`; this knob ADDS to that.  Almost certainly
      double-counts the signal — a non-zero value will probably hurt.
      Useful as a sanity check.

# ============================================================================
# ALREADY VALIDATED (do not remove)
# ============================================================================

  [C] M14_SELL_QTY_THR=6 → pull bid for SIZE_DEFENSE_TICKS (50) ticks.
      Verified on logs/510439:
        4 firings, mean Δmid = -4.5 ticks over the next 5000 ts (50 ticks),
        50% pct_down at h=500.
      Verified on R4 historical (3 days):
        M14 SELL HP qty≥6 ⇒ -0.47 avg @ h=100, 47% wr-down.
      Defensive: stops us accumulating longs into the forecast drop.
"""
import json
from datamodel import TradingState, Order
from typing import List


class Trader:

    PRODUCT = 'HYDROGEL_PACK'
    LIMIT = 200

    # ----- Core MM tunables ----------------------------------------------
    EDGE = 14        # [B] passive half-spread.  Test 14, 16.
    SKEW = 0.02           # linear position skew on FV
    NL_COEFF = 0.8        # nonlinear inventory term

    # ----- OU mean-reversion ---------------------------------------------
    LT_MEAN_PRIOR = 9995.0   # R4 empirical mean = 9994.65
    LT_MEAN_BLEND = 0.25     # [A] blend weight.  Test 0.10, 0.20, 0.40.
    LT_MEAN_ALPHA = 0.001    # adaptive μ EMA

    # ----- OBI extra tilt (off by default; L1 microprice already has OBI) -
    OBI_EXTRA_TILT = 0.0     # [E] extra k·OBI_L1 added to FV.  Test 1.0, 2.0.

    # ----- M38 quote-tightening (off by default) -------------------------
    M38_TIGHTEN_AMOUNT = 0   # [D] extra ticks of price improvement when M38
                              #     traded HP within M38_ACTIVE_TICKS.  Test 1.
    M38_ACTIVE_TICKS = 20    # decay window (in ticks; ×100 for ts)

    # ----- M14 large-sell defense (validated, ON) ------------------------
    SIZE_DEFENSE_TICKS = 50      # [C] pull bid for this many ticks
    M14_SELL_QTY_THR = 6         #     when M14 sells qty≥this

    def _load_state(self, state: TradingState) -> dict:
        if state.traderData:
            try:
                return json.loads(state.traderData)
            except Exception:
                return {}
        return {}

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        mem = self._load_state(state)

        if self.PRODUCT not in state.order_depths:
            return result, conversions, json.dumps(mem)

        # ----- Counterparty trigger detection -----
        # Reactive: only fires after a CP trade has already crossed the book.
        # state.market_trades = trades involving SUB; same for own_trades.
        size_defense_ts = self.SIZE_DEFENSE_TICKS * 100
        m38_active_ts = self.M38_ACTIVE_TICKS * 100
        for src in (state.market_trades, state.own_trades):
            for tr in src.get(self.PRODUCT, []):
                qty = abs(getattr(tr, 'quantity', 0))
                seller = getattr(tr, 'seller', None)
                buyer = getattr(tr, 'buyer', None)

                # [C] M14 large sell → pull bid (validated)
                if seller == 'Mark 14' and qty >= self.M14_SELL_QTY_THR:
                    mem['m14_high_sell_until'] = max(
                        mem.get('m14_high_sell_until', 0),
                        tr.timestamp + size_defense_ts,
                    )

                # [D] M38 just traded HP → tighten quotes window
                if 'Mark 38' in (seller, buyer):
                    mem['m38_active_until'] = max(
                        mem.get('m38_active_until', 0),
                        tr.timestamp + m38_active_ts,
                    )

        result[self.PRODUCT] = self._market_make(state, mem)
        return result, conversions, json.dumps(mem)

    def _market_make(self, state: TradingState, mem: dict) -> List[Order]:
        order_depth = state.order_depths[self.PRODUCT]
        position = state.position.get(self.PRODUCT, 0)
        orders: List[Order] = []

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        bid_vol_l1 = order_depth.buy_orders[best_bid]
        ask_vol_l1 = -order_depth.sell_orders[best_ask]

        # ----- L1 microprice as FV baseline (already encodes OBI) -----
        fair_value = (best_bid * ask_vol_l1 + best_ask * bid_vol_l1) / (bid_vol_l1 + ask_vol_l1)

        # [E] Optional extra OBI tilt on top of microprice (default 0 = off).
        if self.OBI_EXTRA_TILT != 0.0:
            obi_l1 = (bid_vol_l1 - ask_vol_l1) / (bid_vol_l1 + ask_vol_l1)
            fair_value += self.OBI_EXTRA_TILT * obi_l1

        # ----- OU mean-reversion blend -----
        prev_mean = mem.get('lt_mean', self.LT_MEAN_PRIOR)
        new_mean = self.LT_MEAN_ALPHA * mid + (1 - self.LT_MEAN_ALPHA) * prev_mean
        mem['lt_mean'] = new_mean
        fair_value = (1 - self.LT_MEAN_BLEND) * fair_value + self.LT_MEAN_BLEND * new_mean

        # ----- Inventory skew (linear + nonlinear) -----
        pos_ratio = position / self.LIMIT
        fv_shift = position * self.SKEW + pos_ratio * abs(pos_ratio) * self.LIMIT * self.SKEW * self.NL_COEFF
        fv = fair_value - fv_shift
        fv_int = int(round(fv))

        buy_cap = self.LIMIT - position
        sell_cap = position + self.LIMIT

        # ----- Aggressive crosses when EV+ -----
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price <= fv_int and buy_cap > 0:
                vol = -order_depth.sell_orders[ask_price]
                qty = min(vol, buy_cap)
                orders.append(Order(self.PRODUCT, ask_price, qty))
                buy_cap -= qty

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_price >= fv_int and sell_cap > 0:
                vol = order_depth.buy_orders[bid_price]
                qty = min(vol, sell_cap)
                orders.append(Order(self.PRODUCT, bid_price, -qty))
                sell_cap -= qty

        # ----- Passive quotes -----
        my_bid = fv_int - self.EDGE
        my_ask = fv_int + self.EDGE

        # Always at least 1 tick away from mid
        my_bid = min(my_bid, int(mid) - 1)
        my_ask = max(my_ask, int(mid) + 1)

        # [D] Penny-improvement clamp; tightens by extra ticks when M38 active.
        m38_active = state.timestamp < mem.get('m38_active_until', 0)
        inner_offset = 1 + (self.M38_TIGHTEN_AMOUNT if m38_active else 0)
        my_bid = max(my_bid, best_bid + inner_offset)
        my_ask = min(my_ask, best_ask - inner_offset)

        # If clamps crossed (book too tight), fall back to mid ± 1
        if my_bid >= my_ask:
            my_bid = int(mid) - 1
            my_ask = int(mid) + 1

        # [C] Pull bid when M14 just dumped large size — let the drop play out.
        skip_bid = state.timestamp < mem.get('m14_high_sell_until', 0)

        if buy_cap > 0 and not skip_bid:
            orders.append(Order(self.PRODUCT, my_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(self.PRODUCT, my_ask, -sell_cap))

        return orders