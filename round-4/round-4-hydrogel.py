"""Round-4 HYDROGEL_PACK-only strategy — consolidated "best" version.

Combines every knob and feature from the prior HP-only files:
  - round-4-hydrogel.py (simple)               EDGE=12, BLEND=0.30, M14_SELL=6
  - round-4-hydro-tentative-best.py (simple)   identical to above
  - hydrobest.py (mid; A/B framework)          EDGE=14, BLEND=0.25, [H] smart clamp,
                                                [J] stale-FV guard, [K] inventory cap
  - round-4-best-hydro.py (advanced; HP_-prefixed for combination)
  - round-4-hydrogel-advanced.py (advanced; unprefixed)
                                                + [Path E] time-decay inventory skew
                                                + [Path F] target-position framework
                                                + [Path A] per-side inner offsets
                                                + [Path B] drop-phase aggressor
                                                + [Path D] multi-level passive quotes
                                                + [N] take-profit FV boost
                                                + [O] aggressive trough-cover
                                                + [M] cash + mid-history ledger

Defaults follow the advanced (most-audited) tuning. Flip a single knob at a
time to A/B test against the running baseline.

# ============================================================================
# A/B TEST KNOBS (priority order from EDA recommendations)
# ============================================================================

  [B] EDGE                — passive half-spread.   Baseline 14.   Test: 12, 16.
      R4 HP std is ~15% wider than R3 (34.6 vs ~30); wider edge captures more
      $/fill at the cost of fewer fills.

  [A] LT_MEAN_BLEND       — OU mean-reversion weight. Baseline 0.25.
      Test: 0.10, 0.20, 0.30, 0.40. 521635 with 0.30 underperformed by $1309
      vs 520782 baseline.

  [D] M38_TIGHTEN_AMOUNT  — extra ticks of price improvement when Mark 38
      traded HP recently. Baseline 0 (off). Test: 1.

  [E] OBI_EXTRA_TILT      — extra weight on L1 OBI beyond the implicit
      microprice tilt. Baseline 0.0 (off). Test: 1.0, 2.0.

# ============================================================================
# ALREADY VALIDATED (do not remove)
# ============================================================================

  [C] M14_SELL_QTY_THR=6  → pull bid for SIZE_DEFENSE_TICKS (50) ticks.
      Verified on logs/510439 (4 firings, mean Δmid=-4.5 over 5000 ts) and
      R4 historical (3 days; -0.47 avg @ h=100, 47% wr-down). DEFAULT here is
      999 (OFF) — the advanced strategy's structural short bias makes M14
      attack fills profitable on average; lower this to 6 to re-enable.

  [F] M22_BUY_QTY_THR=4   → pull bid; M22 HP buys precede a fast drop
      (n=9–10/run, d100≈-4.4, 89-90% pct_down). DEFAULT 999 (OFF).

  [G] M14_BUY_QTY_THR=6   → pull ASK; biggest loss center pre-defense
      (fwd500 ≈ -1100 to -1400 / run). DEFAULT 999 (OFF) for same reason as [C].

  [H] MIN_QUOTE_EDGE      → smart spread clamp; only post if it preserves
      MIN_QUOTE_EDGE distance from FV. Baseline 0 (always allow penny-improve).

  [I] LT_MEAN_PRIOR=9995  → matches AR(1) μ̂=9995.4 on R4 historical.
      Bumping to 10010 was tested and reverted: structural short bias is a
      FEATURE here, since mid mean-reverts down and we profit on reversion.

  [J] STALE_FV_GUARD       — skip aggressor crosses when |fv−mid|>this.
      Baseline 0 (off). Test 5, 8.

  [K] INVENTORY_SOFT_CAP   — disable aggressor adds on the loaded side once
      |position|>=this. Baseline 200 (=LIMIT, off). Test 150.

  [L] OU calibration       φ=0.998020, half-life=350 ticks, μ=9995.4 on R4.

  [M] traderData ledger    total_cash + mid_hist[≤200], persisted across calls.
      MTM = total_cash + position × mid at any tick.

  [N] Take-profit FV boost — DISABLED (THRESHOLD=99999) after 521635 audit.

  [O] Aggressive trough-cover, PASSIVE-only. DISABLED (MIN_SHORT=9999) after
      522059 audit (fired 81/1000 ticks, 0 added fills).

# ============================================================================
# ALTERNATE FRAMEWORKS (opt-in)
# ============================================================================

  [Path E] Time-decay inventory skew (DEFAULT). Inventory aversion ramps
      from 1.0 at run start to 1+TIME_DECAY_BOOST at run end. At pos=-180,
      end-of-run skew is ~3.6× the open. Set TIME_DECAY_BOOST=0 to disable.

  [Path F] OU optimal target-position framework. Set USE_PATH_F=True to
      replace the linear+nonlinear skew with deviation-from-target skew:
        target_pos = -LIMIT × tanh((mid-μ)/TARGET_SCALE)
        fv_shift   = (position - target_pos) × TARGETED_SKEW

  [Path A] Per-side inner offset for the penny-improve clamp.
      INNER_OFFSET_*=0 → match NPC; =1 → penny-improve (default).

  [Path B] Drop-phase aggressor BUY threshold raise. Catches discount offers
      during confirmed drops; capped at mid+DROP_COVER_PREMIUM so attack
      offers can't trip it (521635 trap fix).

  [Path D] Multi-level passive quotes. Reserves PROBE_SIZE for a deeper
      probe at PROBE_OFFSET ticks beyond the inside quote. OFF by default.
"""
import json
import math
from datamodel import TradingState, Order
from typing import List


class Trader:

    PRODUCT = 'HYDROGEL_PACK'
    LIMIT = 200

    # ----- Core MM tunables ----------------------------------------------
    EDGE = 14             # [B] passive half-spread
    SKEW = 0.02           # linear position skew on FV
    NL_COEFF = 0.8        # nonlinear inventory term

    # [Path E] Time-decay on inventory aversion (A-S inspired). Ramps from 1.0
    # at run start to (1 + TIME_DECAY_BOOST) at run end. As run ends, every
    # unit of |position| matters more (less time for reversion to bail us out).
    TIME_DECAY_BOOST    = 2.0
    TIME_DECAY_TS_TOTAL = 1_000_000

    # [Path F] OU optimal target-position framework — opt-in alternative to
    # the (linear+nonlinear) inventory skew + Path E time-decay.
    USE_PATH_F        = False
    TARGET_SCALE      = 30.0      # ticks per σ in tanh denominator (~R4 σ̂=34.6)
    TARGETED_SKEW     = 0.05

    # [H] Smart-clamp threshold. 0 = only block penny-improve when it would
    # put us on the wrong side of FV. Higher values reduce penny-improvement
    # under tight books at the cost of fewer fills.
    MIN_QUOTE_EDGE = 0

    # ----- OU mean-reversion (calibrated on R4 historical, half-life=350) -
    LT_MEAN_PRIOR = 9995.0   # AR(1) μ̂ = 9995.4 on R4 historical
    LT_MEAN_BLEND = 0.25     # 521635 with 0.30 underperformed by $1309 vs 520782
    LT_MEAN_ALPHA = 0.001

    # ----- OBI extra tilt (off by default; L1 microprice already has OBI) -
    OBI_EXTRA_TILT = 0.0

    # ----- M38 quote-tightening (off by default) -------------------------
    M38_TIGHTEN_AMOUNT = 0
    M38_ACTIVE_TICKS = 20

    # ----- Aggressor-cross safety knobs (off by default) -----------------
    STALE_FV_GUARD = 0
    INVENTORY_SOFT_CAP = 200

    # ----- TraderData history config [M] ---------------------------------
    MID_HIST_LEN = 200

    # ----- [Path A] Per-side inner offset for the penny-improve clamp ----
    INNER_OFFSET_BID = 1
    INNER_OFFSET_ASK = 1     # 522750: 0 lost 10/12 M38 SELL fills to NPC queue

    # ----- [Path B] Drop-phase aggressive cover (uses mid_hist) -----------
    VELOCITY_LOOKBACK   = 20
    VELOCITY_DROP_THR   = -5
    DROP_COVER_MIN_POS  = 100
    DROP_COVER_PREMIUM  = 3

    # ----- [Path D] Multi-level passive quotes ---------------------------
    PROBE_SIZE   = 0         # 522750: probes never fired (deep, no flash events)
    PROBE_OFFSET = 5

    # ----- Take-profit FV boost [N] — DISABLED after 521635 audit ---------
    TAKE_PROFIT_THRESHOLD = 99999
    TAKE_PROFIT_MIN_POS   = 100
    TAKE_PROFIT_FV_BOOST  = 0

    # ----- [O] Aggressive trough-cover (PASSIVE only, no aggressor) ------
    # 522059 audit: fired 81/1000 ticks, 0 added fills. Disabled via MIN_SHORT.
    COVER_MIN_SHORT      = 9999
    COVER_MIN_MTM        = 500
    COVER_Z_THRESHOLD    = 0.2
    COVER_LOOKBACK       = 50
    COVER_INNER_OFFSET   = 4

    # ----- CP defenses — DISABLED by default after 519672 audit -----------
    # Why disabled: structural SHORT-BIAS (OU pulls FV below mid → sell at
    # peaks). 519672 audit: M14 BUY trigger blocked 5/11 expected M14 SELL
    # fills, each costing ~$200 of long-run profit. Net cost ~$1,400 vs
    # 512282 baseline. Lower thresholds to re-enable individually.
    SIZE_DEFENSE_TICKS = 50
    M14_SELL_QTY_THR = 999       # [C] OFF (was 6)
    M14_BUY_QTY_THR = 999        # [G] OFF (was 6)
    M22_BUY_QTY_THR = 999        # [F] OFF (was 4)
    M22_BUY_DEFENSE_TICKS = 50

    # ====================================================================
    # ===== STATE =========================================================
    # ====================================================================

    def _load_state(self, state: TradingState) -> dict:
        if state.traderData:
            try:
                return json.loads(state.traderData)
            except Exception:
                return {}
        return {}

    # ====================================================================
    # ===== HELPERS =======================================================
    # ====================================================================

    def _target_position(self, mid: float, lt_mean: float) -> float:
        """[Path F] OU optimal target position.

        Returns target ∈ [-LIMIT, +LIMIT] based on tanh of standardized
        deviation from mean. At mid=μ → 0; at mid≫μ → -LIMIT (max short);
        at mid≪μ → +LIMIT (max long). Sign-correct for OU mean-reversion.
        """
        z = (mid - lt_mean) / self.TARGET_SCALE
        return -self.LIMIT * math.tanh(z)

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

    # ====================================================================
    # ===== MAIN ENTRY ====================================================
    # ====================================================================

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        mem = self._load_state(state)

        if self.PRODUCT not in state.order_depths:
            return result, conversions, json.dumps(mem)

        # ----- [M] Cash ledger from own_trades (since last call) -----
        # MTM = total_cash + position × mid. IMC delivers each fill once in
        # own_trades, so no double-counting.
        cash = mem.get('total_cash', 0.0)
        for tr in state.own_trades.get(self.PRODUCT, []):
            if getattr(tr, 'buyer', None) == 'SUBMISSION':
                cash -= tr.price * abs(tr.quantity)
            elif getattr(tr, 'seller', None) == 'SUBMISSION':
                cash += tr.price * abs(tr.quantity)
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

        # ----- [N] Take-profit FV boost -----
        # When MTM is well in the green and position is heavy, bias FV toward
        # neutral. Books gains and frees cap. DISABLED by THRESHOLD=99999.
        mtm = mem.get('total_cash', 0.0) + position * mid
        if mtm > self.TAKE_PROFIT_THRESHOLD:
            if position <= -self.TAKE_PROFIT_MIN_POS:
                fair_value += self.TAKE_PROFIT_FV_BOOST
            elif position >= self.TAKE_PROFIT_MIN_POS:
                fair_value -= self.TAKE_PROFIT_FV_BOOST

        # ----- Inventory skew -----
        if self.USE_PATH_F:
            target_pos = self._target_position(mid, new_mean)
            deviation = position - target_pos
            fv_shift = deviation * self.TARGETED_SKEW
        else:
            elapsed_frac = min(1.0, state.timestamp / max(1, self.TIME_DECAY_TS_TOTAL))
            time_factor = 1.0 + self.TIME_DECAY_BOOST * elapsed_frac
            pos_ratio = position / self.LIMIT
            fv_shift = (position * self.SKEW
                        + pos_ratio * abs(pos_ratio) * self.LIMIT * self.SKEW * self.NL_COEFF)
            fv_shift *= time_factor
        fv = fair_value - fv_shift
        fv_int = int(round(fv))

        buy_cap = self.LIMIT - position
        sell_cap = position + self.LIMIT

        # ----- Defensive trigger flags (gate PASSIVE quotes only) ----------
        # We do NOT gate aggressor crosses on predict_*: empirically that
        # blocks long-term-profitable short entries (519672 audit).
        predict_drop = (state.timestamp < mem.get('m14_high_sell_until', 0)
                        or state.timestamp < mem.get('m22_buy_until', 0))
        predict_rise = state.timestamp < mem.get('m14_high_buy_until', 0)

        # [J] Stale-FV guard: skip aggressor crosses when FV strays from mid.
        stale_fv = (self.STALE_FV_GUARD > 0
                    and abs(fv_int - mid) > self.STALE_FV_GUARD)

        allow_buy_cross  = (not stale_fv) and (position <  self.INVENTORY_SOFT_CAP)
        allow_sell_cross = (not stale_fv) and (position > -self.INVENTORY_SOFT_CAP)

        # [Path B] Drop-phase aggressor BUY threshold raise. CAPPED at mid+
        # PREMIUM so attack offers above mid+PREMIUM cannot fire it (521635
        # trap fix).
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
        my_bid = min(my_bid, int(mid) - 1)
        my_ask = max(my_ask, int(mid) + 1)

        # [D] Penny-improvement target; M38-active widens the offset.
        m38_active = state.timestamp < mem.get('m38_active_until', 0)
        m38_extra = self.M38_TIGHTEN_AMOUNT if m38_active else 0
        # [Path A] Per-side inner offsets.
        inner_offset_bid = self.INNER_OFFSET_BID + m38_extra
        inner_offset_ask = self.INNER_OFFSET_ASK + m38_extra

        # [H] Smart clamp: only post if it preserves MIN_QUOTE_EDGE distance
        # from FV. Otherwise match at best, or step away.
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
                if my_bid >= my_ask:
                    my_ask = my_bid + 1

        # [Path D] Multi-level passive quotes: split into inside + deep probe.
        # The probe sits PROBE_OFFSET ticks beyond the inside quote and only
        # fills on a sweep / flash event. Free option, same total capacity.
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
