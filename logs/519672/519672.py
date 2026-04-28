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

  [F] M22_BUY_QTY_THR=4 → pull bid for M22_BUY_DEFENSE_TICKS (50) ticks.
      M22 is a structural seller across the option strikes.  Their HP buys
      (almost always hitting SUBMISSION's offer) precede a fast drop.
      Verified across logs/510439, 512282, 497869:
        n=9–10 per run, d100 ≈ −4.4, d500 ≈ −4.7, 89–90% pct_down(d500).
      Stronger and more frequent than M14 SELL trigger.  Same defensive
      action: pull our bid so we don't accumulate longs into the drop.

  [G] M14_BUY_QTY_THR=6 → pull ASK for SIZE_DEFENSE_TICKS (50) ticks. NEW.
      The biggest loss center in 510439/512282/512667 audits:
        n=10–15 per run, qty avg ~12, half_paid +7-8 (we sell BELOW mid),
        fwd500 ≈ −1100 to −1400 per run.  M14 BUYS HP from us at our (clamped)
        ask, then mid rises strongly.  Direct mirror of [C] but on the
        opposite side: pull our ask so we don't accumulate shorts into the
        forecast rise.

  [H] MIN_QUOTE_EDGE=7 → smart spread clamp. NEW.
      Old behavior `my_bid = max(my_bid, best_bid + 1)` always penny-improves
      into the inside spread, which collapses our EDGE buffer when the book
      is tight.  Mark 14 specifically times their fills against our clamped
      quotes during regime shifts.
      New rule: penny-improve only if it preserves at least MIN_QUOTE_EDGE
      ticks of distance from FV.  Otherwise match at best_bid/best_ask, and
      if even that is too close, stay at fv ± EDGE (will not fill but will
      not lose).  Three modes:
        wide book  → penny-improve (case 1, current behavior)
        mid book   → match (case 2, new)
        tight book → step away (case 3, new) — accept fewer fills

  [I] LT_MEAN_PRIOR=10010 → reduce structural short bias. NEW.
      Was 9995.  Audits show position is short 97-99% of the time across
      all 3 logs and ends ~−170.  With LT_MEAN=9995 and typical mid 10025,
      OU pull is 0.25 × 30 = 7.5 ticks DOWN, biasing us to sell.  Bumping
      prior to 10010 halves the early-game pull (3.75 ticks).  After ~5000
      ticks the EMA converges to actual mean regardless of prior, so this
      mostly affects the opening 50 ticks of trading.

  [J] STALE_FV_GUARD (off by default).  Test: 5, 8.
      Skip aggressor crosses when |fv − mid| > this many ticks.  During
      regime shifts microprice → FV briefly disagrees with mid; aggressor
      crosses then fire on stale FV and we get adverse-selected.

  [K] INVENTORY_SOFT_CAP (off by default = LIMIT).  Test: 150.
      When |position| ≥ this, disable aggressor adds on the loaded side.
      Stops runaway accumulation against momentum.
"""
import json
from datamodel import TradingState, Order
from typing import List


class Trader:

    PRODUCT = 'HYDROGEL_PACK'
    LIMIT = 200

    # ----- Core MM tunables ----------------------------------------------
    EDGE = 14             # [B] passive half-spread.  Test 14, 16.
    SKEW = 0.02           # linear position skew on FV
    NL_COEFF = 0.8        # nonlinear inventory term
    MIN_QUOTE_EDGE = 0    # [H] only block penny-improve when it would put us
                          #     on the WRONG SIDE of FV (i.e. bid > FV / ask < FV).
                          #     Was 7 then 4 — both killed normal bid posts because
                          #     the OU pull keeps FV ≈ best_bid+1, leaving < 4 ticks
                          #     of buffer. With 0 we match 512282's coverage but
                          #     still skip the M14-attack regime-shift case.

    # ----- OU mean-reversion ---------------------------------------------
    LT_MEAN_PRIOR = 9995.0   # [I] reverted from 10010 — short bias is a FEATURE
                              #     (mid mean-reverts down → short profits on reversion)
    LT_MEAN_BLEND = 0.25     # [A] blend weight.  Test 0.10, 0.20, 0.40.
    LT_MEAN_ALPHA = 0.001    # adaptive μ EMA

    # ----- OBI extra tilt (off by default; L1 microprice already has OBI) -
    OBI_EXTRA_TILT = 0.0     # [E] extra k·OBI_L1 added to FV.  Test 1.0, 2.0.

    # ----- M38 quote-tightening (off by default) -------------------------
    M38_TIGHTEN_AMOUNT = 0   # [D] extra ticks of price improvement when M38
                              #     traded HP within M38_ACTIVE_TICKS.  Test 1.
    M38_ACTIVE_TICKS = 20    # decay window (in ticks; ×100 for ts)

    # ----- Aggressor-cross safety knobs (off by default) -----------------
    STALE_FV_GUARD = 0       # [J] skip cross when |fv−mid| > this. Test 5, 8.
    INVENTORY_SOFT_CAP = 200 # [K] = LIMIT (off). Test 150.

    # ----- M14 defense (both sides; both validated, ON) ------------------
    SIZE_DEFENSE_TICKS = 50      # window in ticks (×100 for ts)
    M14_SELL_QTY_THR = 6         # [C] pull BID when M14 sells qty≥this
    M14_BUY_QTY_THR = 6          # [G] pull ASK when M14 buys  qty≥this

    # ----- M22 buy defense (validated, ON) -------------------------------
    M22_BUY_QTY_THR = 4          # [F] pull bid when M22 buys HP qty≥this
    M22_BUY_DEFENSE_TICKS = 50   #     for this many ticks

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
        m14_defense_ts = self.SIZE_DEFENSE_TICKS * 100
        m22_defense_ts = self.M22_BUY_DEFENSE_TICKS * 100
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
                        tr.timestamp + m14_defense_ts,
                    )

                # [G] M14 large buy → pull ask (validated; biggest loss center)
                if buyer == 'Mark 14' and qty >= self.M14_BUY_QTY_THR:
                    mem['m14_high_buy_until'] = max(
                        mem.get('m14_high_buy_until', 0),
                        tr.timestamp + m14_defense_ts,
                    )

                # [F] M22 buys HP → pull bid (validated; bear signal d100≈-4.4)
                if buyer == 'Mark 22' and qty >= self.M22_BUY_QTY_THR:
                    mem['m22_buy_until'] = max(
                        mem.get('m22_buy_until', 0),
                        tr.timestamp + m22_defense_ts,
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

        # ----- Defensive trigger flags (computed early to gate BOTH aggressor
        # crosses AND passive quotes — keeps the predictions consistent) -----
        # [C/F] predict DROP: M14 just dumped large size, OR M22 just bought.
        predict_drop = (state.timestamp < mem.get('m14_high_sell_until', 0)
                        or state.timestamp < mem.get('m22_buy_until', 0))
        # [G]   predict RISE: M14 just bought large size.
        predict_rise = state.timestamp < mem.get('m14_high_buy_until', 0)

        # [J] Stale-FV guard: skip aggressor crosses when FV strays from mid.
        stale_fv = (self.STALE_FV_GUARD > 0
                    and abs(fv_int - mid) > self.STALE_FV_GUARD)

        # ----- Aggressor crosses gated by:
        #   stale_fv  : both sides off
        #   soft cap  : loaded side off
        #   predict_*: opposite-direction side off (so we don't cross UP to buy
        #              right after a M14 SELL that forecasts a drop, etc.)
        allow_buy_cross  = ((not stale_fv)
                            and (position <  self.INVENTORY_SOFT_CAP)
                            and (not predict_drop))
        allow_sell_cross = ((not stale_fv)
                            and (position > -self.INVENTORY_SOFT_CAP)
                            and (not predict_rise))

        # ----- Aggressive crosses when EV+ -----
        if allow_buy_cross:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price <= fv_int and buy_cap > 0:
                    vol = -order_depth.sell_orders[ask_price]
                    qty = min(vol, buy_cap)
                    orders.append(Order(self.PRODUCT, ask_price, qty))
                    buy_cap -= qty

        if allow_sell_cross:
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

        # [D] Penny-improvement target; M38-active widens the offset.
        m38_active = state.timestamp < mem.get('m38_active_until', 0)
        inner_offset = 1 + (self.M38_TIGHTEN_AMOUNT if m38_active else 0)

        # [H] Smart clamp: only penny-improve if it preserves MIN_QUOTE_EDGE
        # of distance from FV.  Otherwise match at best, or step away.
        # ----- Bid side -----
        inner_bid_target = best_bid + inner_offset
        if fv_int - inner_bid_target >= self.MIN_QUOTE_EDGE:
            my_bid = max(my_bid, inner_bid_target)        # case 1: penny-improve
        elif fv_int - best_bid >= self.MIN_QUOTE_EDGE:
            my_bid = max(my_bid, best_bid)                # case 2: match
        # case 3: book too tight to safely improve — leave my_bid at fv-EDGE
        # ----- Ask side -----
        inner_ask_target = best_ask - inner_offset
        if inner_ask_target - fv_int >= self.MIN_QUOTE_EDGE:
            my_ask = min(my_ask, inner_ask_target)        # case 1: penny-improve
        elif best_ask - fv_int >= self.MIN_QUOTE_EDGE:
            my_ask = min(my_ask, best_ask)                # case 2: match
        # case 3: stay at fv+EDGE

        # If clamps crossed (book too tight), fall back to mid ± 1
        if my_bid >= my_ask:
            my_bid = int(mid) - 1
            my_ask = int(mid) + 1

        # Passive quotes use the same predict_drop/predict_rise flags computed
        # above, so they stay consistent with the aggressor-cross gating.
        if buy_cap > 0 and not predict_drop:
            orders.append(Order(self.PRODUCT, my_bid, buy_cap))
        if sell_cap > 0 and not predict_rise:
            orders.append(Order(self.PRODUCT, my_ask, -sell_cap))

        return orders