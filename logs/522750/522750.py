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

  [L] OU calibration update (R4 fit on 30k samples):
      φ = 0.998020,  half-life = 350 ticks,  μ = 9995.4.
      Old BLEND=0.25 implied τ=145 ticks (much shorter than half-life).
      BLEND=0.30 implies τ=180 — better aligned without over-pricing reversion.
      ALPHA=0.0005 (was 0.001) keeps lt_mean anchored to historical 9995
      longer instead of drifting toward current run mean by tick 1000.

  [M] traderData: cash + mid-history ledger (NEW — infrastructure).
      Persists across calls:
        total_cash       running sum of -signed_qty × price from own_trades.
                          MTM = total_cash + position × mid at any tick.
        mid_hist[≤200]   rolling window for adaptive sigma / regime detection.

  [N] Take-profit FV boost — DISABLED (kept as code for revisit).
      521635 audit: caused aggressor BUYs at +8 above mid during M14 attacks.

  [O] Aggressive trough-cover, PASSIVE-ONLY (NEW).
      Detects "high-MTM + heavy-short + mid at recent trough" regime via:
        position ≤ −COVER_MIN_SHORT
        MTM       > COVER_MIN_MTM
        z(mid vs last 50 mids) < −COVER_Z_THRESHOLD
      When active: posts passive bid at `best_bid + COVER_INNER_OFFSET`
      (4 ticks inside the spread, vs default 1).  Designed to capture cheap
      cover from M38/NPC sell flow that fills our deeper bid.  Critically:
      ONLY overrides passive bid placement.  Aggressor BUY threshold uses
      the un-modified fv_int — cannot lift attack offers like [N] did.
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

    # ----- OU mean-reversion (calibrated on R4 historical, half-life=350) -
    LT_MEAN_PRIOR = 9995.0   # matches AR(1) μ̂ = 9995.4 on R4 historical
    LT_MEAN_BLEND = 0.25     # 521635 with 0.30 underperformed by $1309 vs 520782.
                              # Reverted; OU theory says 0.30, data disagrees.
    LT_MEAN_ALPHA = 0.001    # reverted from 0.0005 (same A/B isolation)

    # ----- OBI extra tilt (off by default; L1 microprice already has OBI) -
    OBI_EXTRA_TILT = 0.0     # [E] extra k·OBI_L1 added to FV.  Test 1.0, 2.0.

    # ----- M38 quote-tightening (off by default) -------------------------
    M38_TIGHTEN_AMOUNT = 0   # [D] extra ticks of price improvement when M38
                              #     traded HP within M38_ACTIVE_TICKS.  Test 1.
    M38_ACTIVE_TICKS = 20    # decay window (in ticks; ×100 for ts)

    # ----- Aggressor-cross safety knobs (off by default) -----------------
    STALE_FV_GUARD = 0       # [J] skip cross when |fv−mid| > this. Test 5, 8.
    INVENTORY_SOFT_CAP = 200 # [K] = LIMIT (off). Test 150.

    # ----- TraderData history config [M] ---------------------------------
    MID_HIST_LEN = 200       # rolling window of mids in mem (~1.6KB at 200)

    # ----- [Path A] Per-side inner offset for the penny-improve clamp ----
    # Default 1 = penny-improve inside spread (current behavior).
    # 0 = MATCH at NPC's best level (share queue, gain +1 tick per fill).
    # Theoretical motivation: the clamp `min(my_ask, best_ask−1)` always
    # leaves us inside the spread by 1 tick, capturing 1 less tick per fill
    # than NPC.  Matching shares queue but recovers that tick.
    INNER_OFFSET_BID = 1     # keep penny-improve on bid (M38 cover flow)
    INNER_OFFSET_ASK = 0     # MATCH on ask side — Path A

    # ----- [Path B] Drop-phase aggressive cover (uses mid_hist) -----------
    # During a confirmed drop (mid moved down ≥ DROP_THR over LOOKBACK ticks)
    # AND we're heavily short AND already winning MTM, raise the aggressor
    # BUY threshold from fv_int to (mid + DROP_PREMIUM).  CAPPED at mid+3
    # so attack offers (typically at mid+8) cannot trip it — that was the
    # 521635 failure mode.  Useful for catching brief discount offers
    # (M38 / NPC L2/L3) during a drop without crossing into attack territory.
    VELOCITY_LOOKBACK   = 20    # ticks to compute mid velocity over
    VELOCITY_DROP_THR   = -5    # mid must drop ≥ 5 ticks over lookback
    DROP_COVER_MIN_POS  = 100   # only when |position| ≥ this
    DROP_COVER_PREMIUM  = 3     # max ticks above mid for aggressor BUY

    # ----- [Path D] Multi-level passive quotes ---------------------------
    # Reserve a small chunk of capacity for a deeper "probe" order at a wider
    # offset.  Catches flash sells that sweep the bid book (or flash buys
    # spiking the ask).  Free option: same total capacity, just split between
    # the inside quote and a defensive deep level.
    PROBE_SIZE   = 10        # lots reserved for the probe (5% of LIMIT)
    PROBE_OFFSET = 5         # ticks beyond the inside quote

    # ----- Take-profit FV boost [N] — DISABLED after 521635 audit ---------
    # The +5 FV boost caused us to AGGRESSOR-BUY at 10031+ during the rally
    # tail (M14 attacks).  Booking gains at the WRONG side of the trade.
    # Cost ~$1,309 vs 520782 baseline.  Set BOOST=0 to disable.
    TAKE_PROFIT_THRESHOLD = 99999 # effectively off
    TAKE_PROFIT_MIN_POS   = 100
    TAKE_PROFIT_FV_BOOST  = 0     # off

    # ----- [O] Aggressive trough-cover (PASSIVE only, no aggressor) ------
    # When MTM is positive AND we're meaningfully short AND mid is depressed
    # vs recent rolling mean (z < −threshold), post our passive bid much
    # closer to mid (`best_bid + COVER_INNER_OFFSET`) so M38 / NPCs fill us
    # at a low price.  Covers some short cheaply, frees position for the
    # next reversal.  Critically, this overrides ONLY the passive bid clamp;
    # the aggressor BUY threshold uses the unboosted fv_int, so we cannot
    # accidentally lift expensive offers (the 521635 trap).
    # 522059 audit: [O] fired 81/1000 ticks but added 0 fills — bumped the
    # avg buy price by +1 tick at no benefit (NPC seller flow already saturated
    # at best_bid+1).  Disabled with high MIN_SHORT to no-op.
    COVER_MIN_SHORT      = 9999 # OFF (was 60).  Set 60 to re-enable.
    COVER_MIN_MTM        = 500
    COVER_Z_THRESHOLD    = 0.2
    COVER_LOOKBACK       = 50
    COVER_INNER_OFFSET   = 4

    # ----- CP defenses — DISABLED by default after 519672 audit -----------
    # Setting thresholds to 999 effectively turns off [C], [F], [G].
    #
    # Why disabled: this strategy is structurally SHORT-BIASED (OU pulls FV
    # below mid → we sell at peaks).  Final mid in every run is ~10017, well
    # below the rally peaks (10040+).  The "M14 attack" fills I worried about
    # earlier (selling at mid−7 during a rally) are actually the strategy's
    # most profitable trades — we hold the short through the mean-reversion
    # and end up +17 per share.
    #
    # 519672 audit:  M14 BUY trigger blocked 5 of 11 expected M14 SELL fills.
    # Each blocked fill cost the strategy ~$200 of long-run profit.
    # Net cost of all triggers in 519672: ~$1,400 vs 512282 baseline.
    #
    # To re-enable any trigger, lower its threshold.
    SIZE_DEFENSE_TICKS = 50      # window in ticks (×100 for ts)
    M14_SELL_QTY_THR = 999       # [C] OFF (was 6).  Set 6 to re-enable.
    M14_BUY_QTY_THR = 999        # [G] OFF (was 6).  Set 6 to re-enable.

    M22_BUY_QTY_THR = 999        # [F] OFF (was 4).  Set 4 to re-enable.
    M22_BUY_DEFENSE_TICKS = 50   #     window when re-enabled

    def _load_state(self, state: TradingState) -> dict:
        if state.traderData:
            try:
                return json.loads(state.traderData)
            except Exception:
                return {}
        return {}

    def _cover_active(self, mid: float, position: int, mem: dict) -> bool:
        """[O] Detect aggressive-cover regime: heavy short + winning MTM +
        mid is at a recent trough vs the rolling mean."""
        if position > -self.COVER_MIN_SHORT:
            return False
        mtm = mem.get('total_cash', 0.0) + position * mid
        if mtm < self.COVER_MIN_MTM:
            return False
        hist = mem.get('mid_hist', [])
        if len(hist) < self.COVER_LOOKBACK:
            return False
        recent = hist[-self.COVER_LOOKBACK:]
        n = len(recent)
        rmean = sum(recent) / n
        rvar = sum((x - rmean) ** 2 for x in recent) / n
        rstd = max(rvar ** 0.5, 0.5)
        z = (mid - rmean) / rstd
        return z < -self.COVER_Z_THRESHOLD

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        mem = self._load_state(state)

        if self.PRODUCT not in state.order_depths:
            return result, conversions, json.dumps(mem)

        # ----- [M] Cash ledger from own_trades (since last call) -----
        # Tracks running cash so MTM = total_cash + position × mid at any tick.
        # IMC delivers each fill once in own_trades, so no double-counting.
        cash = mem.get('total_cash', 0.0)
        for tr in state.own_trades.get(self.PRODUCT, []):
            if getattr(tr, 'buyer', None) == 'SUBMISSION':
                cash -= tr.price * abs(tr.quantity)         # we paid out
            elif getattr(tr, 'seller', None) == 'SUBMISSION':
                cash += tr.price * abs(tr.quantity)         # we received
        mem['total_cash'] = cash

        # ----- [M] Mid-history rolling window ------------------------
        od = state.order_depths[self.PRODUCT]
        if od.buy_orders and od.sell_orders:
            cur_mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2
            hist = mem.setdefault('mid_hist', [])
            hist.append(cur_mid)
            if len(hist) > self.MID_HIST_LEN:
                del hist[:-self.MID_HIST_LEN]

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

        # ----- [N] Take-profit FV boost ----------------------------------
        # When MTM is well in the green and position is heavy, bias FV toward
        # neutral.  This loosens the smart-clamp and inventory-skew interaction
        # so the passive bid (or ask, if long) gets posted closer to mid,
        # encouraging cover.  Books gains and frees cap for the next reversal.
        mtm = mem.get('total_cash', 0.0) + position * mid
        if mtm > self.TAKE_PROFIT_THRESHOLD:
            if position <= -self.TAKE_PROFIT_MIN_POS:
                fair_value += self.TAKE_PROFIT_FV_BOOST     # nudge toward buying
            elif position >= self.TAKE_PROFIT_MIN_POS:
                fair_value -= self.TAKE_PROFIT_FV_BOOST     # symmetric (rare for HP)

        # ----- Inventory skew (linear + nonlinear) -----
        pos_ratio = position / self.LIMIT
        fv_shift = position * self.SKEW + pos_ratio * abs(pos_ratio) * self.LIMIT * self.SKEW * self.NL_COEFF
        fv = fair_value - fv_shift
        fv_int = int(round(fv))

        buy_cap = self.LIMIT - position
        sell_cap = position + self.LIMIT

        # ----- Defensive trigger flags (gate PASSIVE quotes only) ----------
        # The CP triggers are off by default (thresholds at 999); these flags
        # are False unless thresholds are lowered.  We do NOT gate aggressor
        # crosses on predict_*: empirically that blocks long-term-profitable
        # short entries (519672 audit).
        predict_drop = (state.timestamp < mem.get('m14_high_sell_until', 0)
                        or state.timestamp < mem.get('m22_buy_until', 0))
        predict_rise = state.timestamp < mem.get('m14_high_buy_until', 0)

        # [J] Stale-FV guard: skip aggressor crosses when FV strays from mid.
        stale_fv = (self.STALE_FV_GUARD > 0
                    and abs(fv_int - mid) > self.STALE_FV_GUARD)

        # ----- Aggressor crosses gated only by stale_fv and inventory cap.
        allow_buy_cross  = (not stale_fv) and (position <  self.INVENTORY_SOFT_CAP)
        allow_sell_cross = (not stale_fv) and (position > -self.INVENTORY_SOFT_CAP)

        # [Path B] Drop-phase aggressor BUY threshold raise.  When the
        # market has dropped meaningfully over recent ticks AND we're heavy
        # short AND winning MTM, raise the BUY threshold up to mid+PREMIUM
        # so we lift any unusually low offers.  CAPPED at mid+PREMIUM so
        # attack offers above mid+PREMIUM cannot fire it (521635 trap fix).
        buy_threshold = fv_int
        hist = mem.get('mid_hist', [])
        if (len(hist) > self.VELOCITY_LOOKBACK
                and position <= -self.DROP_COVER_MIN_POS
                and mtm > 0):
            velocity = mid - hist[-self.VELOCITY_LOOKBACK]
            if velocity <= self.VELOCITY_DROP_THR:
                buy_threshold = max(fv_int, int(mid) + self.DROP_COVER_PREMIUM)

        # ----- Aggressive crosses when EV+ -----
        if allow_buy_cross:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price <= buy_threshold and buy_cap > 0:
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
        m38_extra = self.M38_TIGHTEN_AMOUNT if m38_active else 0

        # [Path A] Per-side inner offsets — match NPC on ask, penny-improve on bid.
        inner_offset_bid = self.INNER_OFFSET_BID + m38_extra
        inner_offset_ask = self.INNER_OFFSET_ASK + m38_extra

        # [H] Smart clamp: only post if it preserves MIN_QUOTE_EDGE
        # of distance from FV.  Otherwise match at best, or step away.
        # ----- Bid side -----
        inner_bid_target = best_bid + inner_offset_bid
        if fv_int - inner_bid_target >= self.MIN_QUOTE_EDGE:
            my_bid = max(my_bid, inner_bid_target)        # case 1: at target
        elif fv_int - best_bid >= self.MIN_QUOTE_EDGE:
            my_bid = max(my_bid, best_bid)                # case 2: fallback to match
        # case 3: book too tight — leave my_bid at fv-EDGE
        # ----- Ask side -----
        inner_ask_target = best_ask - inner_offset_ask
        if inner_ask_target - fv_int >= self.MIN_QUOTE_EDGE:
            my_ask = min(my_ask, inner_ask_target)        # case 1: at target
        elif best_ask - fv_int >= self.MIN_QUOTE_EDGE:
            my_ask = min(my_ask, best_ask)                # case 2: fallback to match
        # case 3: stay at fv+EDGE

        # If clamps crossed (book too tight), fall back to mid ± 1
        if my_bid >= my_ask:
            my_bid = int(mid) - 1
            my_ask = int(mid) + 1

        # [O] Aggressive trough-cover override — bumps the passive bid only.
        # Aggressor BUY threshold (fv_int above) is left untouched, so this
        # cannot trigger the expensive cross-up-into-attack-offers failure
        # mode that 521635 take-profit had.
        if self._cover_active(mid, position, mem):
            cover_bid = best_bid + self.COVER_INNER_OFFSET
            cover_bid = min(cover_bid, int(mid) - 1)
            if cover_bid > my_bid:
                my_bid = cover_bid
                # Re-check clamp crossing after override
                if my_bid >= my_ask:
                    my_ask = my_bid + 1

        # [Path D] Multi-level passive quotes: split into inside + deep probe.
        # The probe sits PROBE_OFFSET ticks beyond the inside quote and only
        # fills on a sweep / flash event.  Free option, same total capacity.
        probe_size = self.PROBE_SIZE
        probe_off  = self.PROBE_OFFSET

        # Bid side
        if buy_cap > 0 and not predict_drop:
            inside_size = max(0, buy_cap - probe_size)
            actual_probe = min(probe_size, buy_cap - inside_size)
            if inside_size > 0:
                orders.append(Order(self.PRODUCT, my_bid, inside_size))
            if actual_probe > 0 and my_bid > probe_off + 1:
                orders.append(Order(self.PRODUCT, my_bid - probe_off, actual_probe))

        # Ask side
        if sell_cap > 0 and not predict_rise:
            inside_size = max(0, sell_cap - probe_size)
            actual_probe = min(probe_size, sell_cap - inside_size)
            if inside_size > 0:
                orders.append(Order(self.PRODUCT, my_ask, -inside_size))
            if actual_probe > 0:
                orders.append(Order(self.PRODUCT, my_ask + probe_off, -actual_probe))

        return orders