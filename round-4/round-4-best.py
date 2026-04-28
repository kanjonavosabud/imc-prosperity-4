"""Combined Round-4 strategy.

Sandwiched from two parts, each preserved verbatim with namespaced state:

  HP block:   teammate's 525232.py (HYDROGEL_PACK, $4,085 official)
              — extensive A/B-tested. Includes [Path E] time-decay inventory
              skew, [Path B] drop-phase aggressor, calibrated OU (μ=9995,
              half-life=350), cash+mid-history ledger.
              All knobs preserved at teammate's tuned defaults.

  VE+OPT:     krishi's 524643.py (VELVETFRUIT_EXTRACT + 4 vouchers, $9,127)
              — M67/M49/M22 counterparty signals, intrinsic FV for VEV_4000/
              4500/5000, microprice for VEV_5100, asymmetric inventory cap.
              VEV_5000 intrinsic FV breakthrough verified (-$57 → +$126).

  Expected combined Day-3 PnL: ~$13,212 ($4,085 HP + $9,127 VE+options).

State namespacing: HP block uses 'hp_*' keys (hp_lt_mean, hp_total_cash,
hp_mid_hist, hp_m14_*, hp_m22_buy_until, hp_m38_active_until). VE block uses
unprefixed keys (lt_mean dict, m67_buy_until, m49_sell_until, m22_sell_until).
No collisions.
"""
import json
import math
from datamodel import TradingState, Order
from typing import List


# ============================================================================
# Helpers (used by VE+OPT block)
# ============================================================================

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def implied_vol(target: float, S: float, K: float, T: float) -> float:
    """Bisection IV solver. Bounded, deterministic, no numpy."""
    if T <= 0 or S <= 0 or target <= 0:
        return 0.20
    intrinsic = max(0.0, S - K)
    if target < intrinsic:
        return 0.05
    lo, hi = 0.01, 3.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_quadratic(xs, ys):
    """Least-squares quadratic y = a*x^2 + b*x + c via normal equations."""
    n = len(xs)
    if n < 3:
        return None
    Sx = sum(xs); Sx2 = sum(x*x for x in xs)
    Sx3 = sum(x**3 for x in xs); Sx4 = sum(x**4 for x in xs)
    Sy = sum(ys); Sxy = sum(x*y for x, y in zip(xs, ys))
    Sx2y = sum(x*x*y for x, y in zip(xs, ys))
    M = [[Sx4, Sx3, Sx2], [Sx3, Sx2, Sx], [Sx2, Sx, float(n)]]
    v = [Sx2y, Sxy, Sy]

    def det3(A):
        return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))

    D = det3(M)
    if abs(D) < 1e-9:
        return None
    out = []
    for j in range(3):
        Mj = [row[:] for row in M]
        for i in range(3):
            Mj[i][j] = v[i]
        out.append(det3(Mj) / D)
    return tuple(out)


class Trader:

    # ====================================================================
    # ===== HP BLOCK (teammate's strategy, verbatim with HP_ prefix) =====
    # ====================================================================
    HP_PRODUCT = 'HYDROGEL_PACK'
    HP_LIMIT = 200

    HP_EDGE = 14
    HP_SKEW = 0.02
    HP_NL_COEFF = 0.8
    HP_TIME_DECAY_BOOST = 2.0
    HP_TIME_DECAY_TS_TOTAL = 1_000_000

    HP_USE_PATH_F = False
    HP_TARGET_SCALE = 30.0
    HP_TARGETED_SKEW = 0.05

    HP_MIN_QUOTE_EDGE = 0

    HP_LT_MEAN_PRIOR = 9995.0
    HP_LT_MEAN_BLEND = 0.25
    HP_LT_MEAN_ALPHA = 0.001

    HP_OBI_EXTRA_TILT = 0.0

    HP_M38_TIGHTEN_AMOUNT = 0
    HP_M38_ACTIVE_TICKS = 20

    HP_STALE_FV_GUARD = 0
    HP_INVENTORY_SOFT_CAP = 200

    HP_MID_HIST_LEN = 200

    HP_INNER_OFFSET_BID = 1
    HP_INNER_OFFSET_ASK = 1

    HP_VELOCITY_LOOKBACK = 20
    HP_VELOCITY_DROP_THR = -5
    HP_DROP_COVER_MIN_POS = 100
    HP_DROP_COVER_PREMIUM = 3

    HP_PROBE_SIZE = 0
    HP_PROBE_OFFSET = 5

    HP_TAKE_PROFIT_THRESHOLD = 99999
    HP_TAKE_PROFIT_MIN_POS = 100
    HP_TAKE_PROFIT_FV_BOOST = 0

    HP_COVER_MIN_SHORT = 9999
    HP_COVER_MIN_MTM = 500
    HP_COVER_Z_THRESHOLD = 0.2
    HP_COVER_LOOKBACK = 50
    HP_COVER_INNER_OFFSET = 4

    HP_SIZE_DEFENSE_TICKS = 10  # SHORT window — signal strongest at h=10 in EDA
    HP_M14_SELL_QTY_THR = 1   # re-enabled: EDA shows |t|=29, 88% wr, +$8.21 mean
    HP_M14_BUY_QTY_THR = 1
    HP_M22_BUY_QTY_THR = 999
    HP_M22_BUY_DEFENSE_TICKS = 50

    # M14 FV-bias signals — SHORT-DURATION (10 ticks, matches data peak signal).
    # Earlier attempt (531358 official) used 50-tick window and bias=$5 → caught
    # -$5k of non-M14 flow at biased prices. Theory: shorter window = less time
    # for non-M14 flow to be caught, while still capturing the directional move.
    # Take-only test (today, local) hurt -$28k because exit lag > signal window
    # given HP edge=14. Conclusion: bias must be small AND short. EDA supports
    # this — submitting to IMC official to validate.
    HP_M14_BUY_BIAS = 3.0     # 36% of mean shift (was 60%, more conservative)
    HP_M14_SELL_BIAS = 3.0
    HP_M14_TAKE_AGG = 0       # take-only mode disabled (FV bias does the lifting)
    HP_M14_BLOCK_POSTS = False  # never block opposite-side posts (lost flow in 531358)

    # ====================================================================
    # ===== VE + OPTIONS BLOCK (krishi's strategy, verbatim) =============
    # ====================================================================
    SMILE_A = 0.142979
    SMILE_B = 0.002469
    SMILE_C = 0.235495
    TTE_INITIAL_DAYS = 7.0
    DIFF_EMA_ALPHA = 0.05
    ABS_EMA_ALPHA = 0.01

    INFORMED_SIGNAL_TICKS = 50
    # M67 BUY VE → +$1.45 / 10t (t=+6.49, 66.7% wr) — primary VE signal.
    # M49 SELL VE and M22 SELL VE are 85% / 74% the SAME TRADE as M67 BUY
    # (M67 buys, M49/M22 sell to it). Including them stacks bias on the same
    # event → over-bias → BT loss of ~$25k. Disabled to prevent double-counting.
    # The sign in code applies the bias as +bias for "expects price up";
    # M49/M22 sells were previously coded with -bias which both:
    #   (a) inverted the direction (data says price UP, not down)
    #   (b) double-counted M67 events
    # Fix: only M67. The 16 / 26 unique M49/M22-only events (without M67)
    # are too few (n<30) to justify a separate signal.
    MARK67_BUY_BIAS = 1.5
    MARK49_SELL_BIAS = 0.0  # disabled: 85% redundant with M67_BUY
    MARK22_SELL_BIAS = 0.0  # disabled: 74% redundant with M67_BUY

    # M14 follow-the-insider bias per product. Verified:
    #   M14 BUY  HP       → +$8.21 / 10t (t=+29.03, 88.5% wr)
    #   M14 BUY  VE       → +$2.18 / 10t (t=+14.06, 77.2% wr)
    #   M14 BUY  VEV_4000 → +$10.54 / 10t (t=+49.02, 100% wr)
    # M14 does NOT trade VEV_4500, 5000, 5100 (n=0 over 3 days), so those
    # entries are removed — they would never fire.
    MARK14_SIGNAL_TICKS = 10    # SHORT window — minimize non-M14 flow caught
    # FV-bias re-enabled at SMALLER magnitude + shorter duration (10 ticks).
    # 531358 used 50-tick @ $1.5 bias → caught -$5k non-M14 flow.
    # Theory: 10-tick @ $1.0 bias → 5x less window, ~33% less magnitude.
    # Submitting to IMC to validate against official matching.
    MARK14_BIAS_PER_PRODUCT = {
        'VELVETFRUIT_EXTRACT': 1.0,  # data: +$2.18 / 10t, ~46% capture
        'VEV_4000': 4.0,             # data: +$10.54 / 10t, ~38% capture (leveraged)
    }
    MARK14_TAKE_AGGRESSION_PER_PRODUCT = {}  # take-only didn't work either

    MM_PARAMS = {
        # Deep ITM: intrinsic FV. M38 regime defense applied to VEV_4500 only.
        # Mark 22 vs Mark 38 footprints are LITERALLY non-overlapping on these
        # deep-ITM products (M22 at compressed spread + high imbal, M38 at full
        # NPC spread + perfectly balanced book). Defense candidate identified
        # via 526010 audit (M38 cost -$8 V4000, -$14 V4500).
        # WHY V4500 ONLY: applying to V4000 dropped local 3-day by -$7k (local
        # matcher fills V4000 abundantly at the spread=21 regime). V4500 local
        # was unchanged by the defense (regime is a no-op locally), making it
        # a "free option" — zero local downside, +$14 expected official save.
        'VEV_4000':            {'limit': 300, 'edge': 12, 'skew': 0.0, 'strike': 4000, 'use_bs_fv': True},
        'VEV_4500':            {'limit': 300, 'edge': 4, 'skew': 0.0, 'strike': 4500, 'm38_skip_spread': 16, 'm38_skip_imbal': 0.05, 'use_bs_fv': True},
        # VEV_5000: REVERTED from intrinsic FV. The 'strike': 5000 setting
        # gave +$126 on Day 3 official but -$1,151 / -$1,622 on Day 1/2 local
        # (overfit to Day 3 down-drift). Microprice is robust: +$468 / +$260 /
        # +$438 across all 3 local days. Cost on Day 3 official ~$183, gained
        # robustness ~$3,866 across other-day-like conditions.
        'VEV_5000':            {'limit': 50, 'edge': 2,  'skew': 0.05, 'asym_pos_thr': 15},
        'VEV_5100':            {'limit': 50, 'edge': 2,  'skew': 0.05, 'asym_pos_thr': 15},
        # OTM strikes — quote only when book widens enough that our quotes
        # land inside the existing spread. Book-spread filter prevents the
        # cross-fill spiral that killed limit=300 in R3 and edge=1 in earlier
        # tests. min_book_spread = edge*2 + 1 ensures quotes are inside.
        'VEV_5200':            {'limit': 25, 'edge': 2, 'skew': 0.06, 'asym_pos_thr': 10, 'min_book_spread': 5, 'take_strict': True},
        'VEV_5300':            {'limit': 20, 'edge': 2, 'skew': 0.08, 'asym_pos_thr': 8,  'min_book_spread': 5, 'take_strict': True},
        'VEV_5400':            {'limit': 15, 'edge': 2, 'skew': 0.10, 'asym_pos_thr': 6,  'min_book_spread': 5, 'take_strict': True},
        'VEV_5500':            {'limit': 15, 'edge': 2, 'skew': 0.10, 'asym_pos_thr': 6,  'min_book_spread': 5, 'take_strict': True},
        'VELVETFRUIT_EXTRACT': {'limit': 200, 'edge': 4,  'skew': 0.003},
    }

    NL_COEFF = 0.8

    # OU params for VE only (HP OU is in HP_LT_MEAN_*)
    LT_MEAN_PRIOR = {'VELVETFRUIT_EXTRACT': 5250.0}
    LT_MEAN_BLEND_PER_PRODUCT = {
        # Bumped from 0.07 → 0.10 after 3-day audit. Local total +$4,526
        # (D1 +$4,876, D2 -$3,808, D3 +$3,460). Theoretically: blend=0.10
        # implies τ≈53 ticks holding horizon (vs 36 at 0.07), closer to
        # realistic fill-to-unwind cycle. With θ≈0.002/tick, blend = 1-exp(-θτ).
        'VELVETFRUIT_EXTRACT': 0.10,
    }
    LT_MEAN_ALPHA = 0.001

    IMBALANCE_FV_SHIFT = {}
    IMBALANCE_RATIO_THR = 999.0

    SCALP_PARAMS = {}
    BUY_PARAMS = {}

    # === TAKE-PROFIT on options ===
    # R3 captured only 3% of peak option PnL ($1.5k of $46k). R4 currently
    # captures 51% ($616 of $1,209). The mechanism missing in BOTH rounds:
    # exit long positions when option mid runs above our avg cost by enough
    # ticks that subsequent mean-reversion would erase the gains.
    # 526010 audit: VEV_5000 peak +$24 → final -$58 (-$83 give-back).
    # Per-share threshold (in option ticks):
    OPT_TP_PRICE_THR = {
        'VEV_4000': 8,    # spread=21, big moves; need wider TP
        'VEV_4500': 5,    # spread=16
        'VEV_5000': 3,    # spread=6
        'VEV_5100': 3,    # spread=4
    }
    OPT_TP_POS_THR = 5     # only fire when meaningfully loaded (5+ lots)

    # ====================================================================
    # ===== SHARED HELPERS ===============================================
    # ====================================================================

    def _load_state(self, state: TradingState) -> dict:
        if state.traderData:
            try:
                return json.loads(state.traderData)
            except Exception:
                return {}
        return {}

    def _track_time(self, mem: dict, state: TradingState) -> float:
        last_ts = mem.get('last_ts', None)
        day = mem.get('day', 0)
        if last_ts is not None and state.timestamp < last_ts:
            day += 1
        mem['day'] = day
        mem['last_ts'] = state.timestamp
        elapsed_within_day = state.timestamp / 1_000_000.0
        days_remaining = max(0.5, self.TTE_INITIAL_DAYS - day - elapsed_within_day)
        return days_remaining / 365.0

    def _smile_iv(self, S: float, K: float, T: float) -> float:
        if S <= 0 or K <= 0 or T <= 0:
            return 0.20
        m_t = math.log(S / K) / math.sqrt(T)
        iv = self.SMILE_A * m_t * m_t + self.SMILE_B * m_t + self.SMILE_C
        return max(0.05, iv)

    # Per-tick smile fit. Backs out IV from each option's mid, fits a
    # quadratic through (m_t, IV) pairs, returns coefficients (A, B, C).
    # Caches per tick to avoid re-fitting once per option call.
    _OPT_STRIKE_LIST = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500)

    def _get_tick_smile(self, state: TradingState, S: float, T: float, mem: dict):
        if mem is None or S is None or S <= 0 or T <= 0:
            return None
        cache_key = state.timestamp
        cached = mem.get('_smile_cache_ts')
        if cached == cache_key:
            return mem.get('_smile_cache_coef')
        xs, ys = [], []
        for K in self._OPT_STRIKE_LIST:
            sym = f'VEV_{K}'
            od = state.order_depths.get(sym)
            if od is None or not od.buy_orders or not od.sell_orders:
                continue
            mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
            iv = implied_vol(mid, S, float(K), T)
            if iv <= 0.05 or iv >= 2.5:
                continue  # skip degenerate fits (deep ITM with intrinsic-only)
            m_t = math.log(S / K) / math.sqrt(T)
            xs.append(m_t)
            ys.append(iv)
        coef = fit_quadratic(xs, ys) if len(xs) >= 4 else None
        mem['_smile_cache_ts'] = cache_key
        mem['_smile_cache_coef'] = coef
        return coef

    def _underlying_mid(self, state: TradingState):
        if 'VELVETFRUIT_EXTRACT' not in state.order_depths:
            return None
        od = state.order_depths['VELVETFRUIT_EXTRACT']
        if not od.buy_orders or not od.sell_orders:
            return None
        bb = max(od.buy_orders.keys())
        ba = min(od.sell_orders.keys())
        bv = od.buy_orders[bb]
        av = -od.sell_orders[ba]
        return (bb * av + ba * bv) / (bv + av)

    # ====================================================================
    # ===== HP BLOCK METHODS (teammate's, prefix-namespaced) =============
    # ====================================================================

    def _hp_target_position(self, mid: float, lt_mean: float) -> float:
        z = (mid - lt_mean) / self.HP_TARGET_SCALE
        return -self.HP_LIMIT * math.tanh(z)

    def _hp_cover_active(self, mid: float, position: int, mem: dict) -> bool:
        if position > -self.HP_COVER_MIN_SHORT:
            return False
        mtm = mem.get('hp_total_cash', 0.0) + position * mid
        if mtm < self.HP_COVER_MIN_MTM:
            return False
        hist = mem.get('hp_mid_hist', [])
        if len(hist) < self.HP_COVER_LOOKBACK:
            return False
        recent = hist[-self.HP_COVER_LOOKBACK:]
        n = len(recent)
        rmean = sum(recent) / n
        rvar = sum((x - rmean) ** 2 for x in recent) / n
        rstd = max(rvar ** 0.5, 0.5)
        z = (mid - rmean) / rstd
        return z < -self.HP_COVER_Z_THRESHOLD

    def _hp_run(self, state: TradingState, mem: dict) -> List[Order]:
        if self.HP_PRODUCT not in state.order_depths:
            return []

        # ----- [M] Cash ledger from own_trades -----
        cash = mem.get('hp_total_cash', 0.0)
        for tr in state.own_trades.get(self.HP_PRODUCT, []):
            if getattr(tr, 'buyer', None) == 'SUBMISSION':
                cash -= tr.price * abs(tr.quantity)
            elif getattr(tr, 'seller', None) == 'SUBMISSION':
                cash += tr.price * abs(tr.quantity)
        mem['hp_total_cash'] = cash

        # ----- [M] Mid-history rolling window -----
        od = state.order_depths[self.HP_PRODUCT]
        if od.buy_orders and od.sell_orders:
            cur_mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2
            hist = mem.setdefault('hp_mid_hist', [])
            hist.append(cur_mid)
            if len(hist) > self.HP_MID_HIST_LEN:
                del hist[:-self.HP_MID_HIST_LEN]

        # ----- HP counterparty triggers (HP-only feeds) -----
        m14_defense_ts = self.HP_SIZE_DEFENSE_TICKS * 100
        m22_defense_ts = self.HP_M22_BUY_DEFENSE_TICKS * 100
        m38_active_ts = self.HP_M38_ACTIVE_TICKS * 100
        for src in (state.market_trades, state.own_trades):
            for tr in src.get(self.HP_PRODUCT, []):
                qty = abs(getattr(tr, 'quantity', 0))
                seller = getattr(tr, 'seller', None)
                buyer = getattr(tr, 'buyer', None)

                if seller == 'Mark 14' and qty >= self.HP_M14_SELL_QTY_THR:
                    mem['hp_m14_high_sell_until'] = max(
                        mem.get('hp_m14_high_sell_until', 0),
                        tr.timestamp + m14_defense_ts,
                    )
                if buyer == 'Mark 14' and qty >= self.HP_M14_BUY_QTY_THR:
                    mem['hp_m14_high_buy_until'] = max(
                        mem.get('hp_m14_high_buy_until', 0),
                        tr.timestamp + m14_defense_ts,
                    )
                if buyer == 'Mark 22' and qty >= self.HP_M22_BUY_QTY_THR:
                    mem['hp_m22_buy_until'] = max(
                        mem.get('hp_m22_buy_until', 0),
                        tr.timestamp + m22_defense_ts,
                    )
                if 'Mark 38' in (seller, buyer):
                    mem['hp_m38_active_until'] = max(
                        mem.get('hp_m38_active_until', 0),
                        tr.timestamp + m38_active_ts,
                    )

        return self._hp_market_make(state, mem)

    def _hp_market_make(self, state: TradingState, mem: dict) -> List[Order]:
        order_depth = state.order_depths[self.HP_PRODUCT]
        position = state.position.get(self.HP_PRODUCT, 0)
        orders: List[Order] = []

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        bid_vol_l1 = order_depth.buy_orders[best_bid]
        ask_vol_l1 = -order_depth.sell_orders[best_ask]

        # L1 microprice as FV baseline
        fair_value = (best_bid * ask_vol_l1 + best_ask * bid_vol_l1) / (bid_vol_l1 + ask_vol_l1)

        if self.HP_OBI_EXTRA_TILT != 0.0:
            obi_l1 = (bid_vol_l1 - ask_vol_l1) / (bid_vol_l1 + ask_vol_l1)
            fair_value += self.HP_OBI_EXTRA_TILT * obi_l1

        # OU mean-reversion (HP namespace)
        prev_mean = mem.get('hp_lt_mean', self.HP_LT_MEAN_PRIOR)
        new_mean = self.HP_LT_MEAN_ALPHA * mid + (1 - self.HP_LT_MEAN_ALPHA) * prev_mean
        mem['hp_lt_mean'] = new_mean
        fair_value = (1 - self.HP_LT_MEAN_BLEND) * fair_value + self.HP_LT_MEAN_BLEND * new_mean

        # M14 directional FV bias (insider follow) — short-duration (10 ticks
        # via HP_SIZE_DEFENSE_TICKS). EDA: M14 BUY HP → +$8.21 mean shift, t=29.
        # Bias of +3 captures ~36% of move while limiting collateral on posts.
        if (self.HP_M14_BUY_BIAS != 0 and
                state.timestamp < mem.get('hp_m14_high_buy_until', 0)):
            fair_value += self.HP_M14_BUY_BIAS
        if (self.HP_M14_SELL_BIAS != 0 and
                state.timestamp < mem.get('hp_m14_high_sell_until', 0)):
            fair_value -= self.HP_M14_SELL_BIAS

        # [N] Take-profit FV boost (effectively off)
        mtm = mem.get('hp_total_cash', 0.0) + position * mid
        if mtm > self.HP_TAKE_PROFIT_THRESHOLD:
            if position <= -self.HP_TAKE_PROFIT_MIN_POS:
                fair_value += self.HP_TAKE_PROFIT_FV_BOOST
            elif position >= self.HP_TAKE_PROFIT_MIN_POS:
                fair_value -= self.HP_TAKE_PROFIT_FV_BOOST

        # Inventory skew with [Path E] time-decay or [Path F] target-position
        if self.HP_USE_PATH_F:
            target_pos = self._hp_target_position(mid, new_mean)
            deviation = position - target_pos
            fv_shift = deviation * self.HP_TARGETED_SKEW
        else:
            elapsed_frac = min(1.0, state.timestamp / max(1, self.HP_TIME_DECAY_TS_TOTAL))
            time_factor = 1.0 + self.HP_TIME_DECAY_BOOST * elapsed_frac
            pos_ratio = position / self.HP_LIMIT
            fv_shift = (position * self.HP_SKEW
                        + pos_ratio * abs(pos_ratio) * self.HP_LIMIT * self.HP_SKEW * self.HP_NL_COEFF)
            fv_shift *= time_factor
        fv = fair_value - fv_shift
        fv_int = int(round(fv))

        buy_cap = self.HP_LIMIT - position
        sell_cap = position + self.HP_LIMIT

        # CP defenses — only block opposite-side posts when HP_M14_BLOCK_POSTS
        # is True (legacy mode). Take-only mode (current default) keeps posts
        # active and uses HP_M14_TAKE_AGG to cross aggressively instead.
        m14_buy_active  = state.timestamp < mem.get('hp_m14_high_buy_until', 0)
        m14_sell_active = state.timestamp < mem.get('hp_m14_high_sell_until', 0)
        m22_buy_active  = state.timestamp < mem.get('hp_m22_buy_until', 0)
        if self.HP_M14_BLOCK_POSTS:
            predict_drop = m14_sell_active or m22_buy_active
            predict_rise = m14_buy_active
        else:
            predict_drop = False
            predict_rise = False

        stale_fv = (self.HP_STALE_FV_GUARD > 0
                    and abs(fv_int - mid) > self.HP_STALE_FV_GUARD)

        allow_buy_cross = (not stale_fv) and (position < self.HP_INVENTORY_SOFT_CAP)
        allow_sell_cross = (not stale_fv) and (position > -self.HP_INVENTORY_SOFT_CAP)

        # [Path B] Drop-phase aggressor BUY threshold raise
        buy_threshold = fv_int
        sell_threshold = fv_int
        hist = mem.get('hp_mid_hist', [])
        if (len(hist) > self.HP_VELOCITY_LOOKBACK
                and position <= -self.HP_DROP_COVER_MIN_POS
                and mtm > 0):
            velocity = mid - hist[-self.HP_VELOCITY_LOOKBACK]
            if velocity <= self.HP_VELOCITY_DROP_THR:
                buy_threshold = max(fv_int, int(mid) + self.HP_DROP_COVER_PREMIUM)

        # M14 take-only aggression — boost cross thresholds when signal active.
        # Bid posting / ask posting unaffected (clean FV±edge → captures
        # ordinary M01/M55/M22 flow at proper edge).
        if m14_buy_active and self.HP_M14_TAKE_AGG > 0:
            buy_threshold = max(buy_threshold, fv_int + self.HP_M14_TAKE_AGG)
        if m14_sell_active and self.HP_M14_TAKE_AGG > 0:
            sell_threshold = min(sell_threshold, fv_int - self.HP_M14_TAKE_AGG)

        # Aggressor crosses
        if allow_buy_cross:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price <= buy_threshold and buy_cap > 0:
                    vol = -order_depth.sell_orders[ask_price]
                    qty = min(vol, buy_cap)
                    orders.append(Order(self.HP_PRODUCT, ask_price, qty))
                    buy_cap -= qty

        if allow_sell_cross:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price >= sell_threshold and sell_cap > 0:
                    vol = order_depth.buy_orders[bid_price]
                    qty = min(vol, sell_cap)
                    orders.append(Order(self.HP_PRODUCT, bid_price, -qty))
                    sell_cap -= qty

        # Passive quotes
        my_bid = fv_int - self.HP_EDGE
        my_ask = fv_int + self.HP_EDGE
        my_bid = min(my_bid, int(mid) - 1)
        my_ask = max(my_ask, int(mid) + 1)

        m38_active = state.timestamp < mem.get('hp_m38_active_until', 0)
        m38_extra = self.HP_M38_TIGHTEN_AMOUNT if m38_active else 0
        inner_offset_bid = self.HP_INNER_OFFSET_BID + m38_extra
        inner_offset_ask = self.HP_INNER_OFFSET_ASK + m38_extra

        # [H] Smart clamp
        inner_bid_target = best_bid + inner_offset_bid
        if fv_int - inner_bid_target >= self.HP_MIN_QUOTE_EDGE:
            my_bid = max(my_bid, inner_bid_target)
        elif fv_int - best_bid >= self.HP_MIN_QUOTE_EDGE:
            my_bid = max(my_bid, best_bid)

        inner_ask_target = best_ask - inner_offset_ask
        if inner_ask_target - fv_int >= self.HP_MIN_QUOTE_EDGE:
            my_ask = min(my_ask, inner_ask_target)
        elif best_ask - fv_int >= self.HP_MIN_QUOTE_EDGE:
            my_ask = min(my_ask, best_ask)

        if my_bid >= my_ask:
            my_bid = int(mid) - 1
            my_ask = int(mid) + 1

        # [O] Aggressive trough-cover (off by default)
        if self._hp_cover_active(mid, position, mem):
            cover_bid = best_bid + self.HP_COVER_INNER_OFFSET
            cover_bid = min(cover_bid, int(mid) - 1)
            if cover_bid > my_bid:
                my_bid = cover_bid
                if my_bid >= my_ask:
                    my_ask = my_bid + 1

        # [Path D] Multi-level passive quotes (probe off by default)
        probe_size = self.HP_PROBE_SIZE
        probe_off = self.HP_PROBE_OFFSET

        if buy_cap > 0 and not predict_drop:
            inside_size = max(0, buy_cap - probe_size)
            actual_probe = min(probe_size, buy_cap - inside_size)
            if inside_size > 0:
                orders.append(Order(self.HP_PRODUCT, my_bid, inside_size))
            if actual_probe > 0 and my_bid > probe_off + 1:
                orders.append(Order(self.HP_PRODUCT, my_bid - probe_off, actual_probe))

        if sell_cap > 0 and not predict_rise:
            inside_size = max(0, sell_cap - probe_size)
            actual_probe = min(probe_size, sell_cap - inside_size)
            if inside_size > 0:
                orders.append(Order(self.HP_PRODUCT, my_ask, -inside_size))
            if actual_probe > 0:
                orders.append(Order(self.HP_PRODUCT, my_ask + probe_off, -actual_probe))

        return orders

    # ====================================================================
    # ===== VE + OPTIONS BLOCK (krishi's, verbatim) ======================
    # ====================================================================

    def buy_flow(self, state: TradingState, product: str, params: dict) -> List[Order]:
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        buy_cap = params['limit'] - position
        if buy_cap <= 0:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_px = best_bid + 1 if (best_ask - best_bid) > 1 else best_bid
        orders.append(Order(product, bid_px, buy_cap))
        return orders

    def iv_scalp(self, state: TradingState, product: str, params: dict,
                 S: float, T: float, mem: dict) -> List[Order]:
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        bb = max(order_depth.buy_orders.keys())
        ba = min(order_depth.sell_orders.keys())
        bv = order_depth.buy_orders[bb]
        av = -order_depth.sell_orders[ba]
        mid = (bb + ba) / 2
        spread = ba - bb

        K = params['strike']
        iv = self._smile_iv(S, K, T)
        theo = bs_call(S, K, T, iv)
        theo_diff = mid - theo

        diff_emas = mem.setdefault('diff_ema', {})
        prev_diff_ema = diff_emas.get(product, theo_diff)
        new_diff_ema = self.DIFF_EMA_ALPHA * theo_diff + (1 - self.DIFF_EMA_ALPHA) * prev_diff_ema
        diff_emas[product] = new_diff_ema

        deviation = theo_diff - new_diff_ema

        abs_emas = mem.setdefault('abs_ema', {})
        prev_abs_ema = abs_emas.get(product, abs(deviation))
        new_abs_ema = self.ABS_EMA_ALPHA * abs(deviation) + (1 - self.ABS_EMA_ALPHA) * prev_abs_ema
        abs_emas[product] = new_abs_ema

        thr_open = params['thr_open']
        thr_activate = params['thr_activate']
        limit = params['limit']

        active = new_abs_ema >= thr_activate

        if active and deviation > thr_open:
            sell_cap = position + limit
            if sell_cap > 0:
                qty = min(sell_cap, bv)
                if qty > 0:
                    orders.append(Order(product, bb, -qty))
            return orders

        if active and deviation < -thr_open:
            buy_cap = limit - position
            if buy_cap > 0:
                qty = min(buy_cap, av)
                if qty > 0:
                    orders.append(Order(product, ba, qty))
            return orders

        buy_cap = limit - position
        if buy_cap > 0:
            bid_px = bb + 1 if spread > 1 else bb
            orders.append(Order(product, bid_px, buy_cap))

        return orders

    def market_make(self, state: TradingState, product: str, params: dict,
                    underlying_mid, lt_means: dict, now_ts: int = 0, mem: dict = None) -> List[Order]:
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        orders: List[Order] = []

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        bid_vol_l1 = order_depth.buy_orders[best_bid]
        ask_vol_l1 = -order_depth.sell_orders[best_ask]

        if params.get('simple_micro'):
            fair_value = (best_bid * ask_vol_l1 + best_ask * bid_vol_l1) / (bid_vol_l1 + ask_vol_l1)
        else:
            total_bid_vol = sum(order_depth.buy_orders.values())
            total_ask_vol = sum(-v for v in order_depth.sell_orders.values())
            avg_bid = sum(p * v for p, v in order_depth.buy_orders.items()) / total_bid_vol
            avg_ask = sum(p * (-v) for p, v in order_depth.sell_orders.items()) / total_ask_vol
            fair_value = (avg_bid * total_ask_vol + avg_ask * total_bid_vol) / (total_bid_vol + total_ask_vol)

        if product in self.LT_MEAN_PRIOR:
            prev_mean = lt_means.get(product, self.LT_MEAN_PRIOR[product])
            new_mean = self.LT_MEAN_ALPHA * mid + (1 - self.LT_MEAN_ALPHA) * prev_mean
            lt_means[product] = new_mean
            w = self.LT_MEAN_BLEND_PER_PRODUCT.get(product, 0.05)
            fair_value = (1 - w) * fair_value + w * new_mean

        # Counterparty FV bias (VE only). Direction validated against R4 data:
        # all three signals predict price UP, so all three bias FV upward.
        m67_active = False
        if product == 'VELVETFRUIT_EXTRACT' and mem is not None:
            if now_ts < mem.get('m67_buy_until', 0):
                fair_value += self.MARK67_BUY_BIAS
                m67_active = True
            if now_ts < mem.get('m49_sell_until', 0):
                fair_value += self.MARK49_SELL_BIAS  # was -=, data says price UP
            if now_ts < mem.get('m22_sell_until', 0):
                fair_value += self.MARK22_SELL_BIAS  # was -=, data says price UP

        # M14 directional bias — applied to VE AND options. The signal is
        # per-product (M14 may buy HP without buying options, etc.).
        if mem is not None and product in self.MARK14_BIAS_PER_PRODUCT:
            bias = self.MARK14_BIAS_PER_PRODUCT[product]
            m14_buy = mem.get('m14_buy_until', {}).get(product, 0)
            m14_sell = mem.get('m14_sell_until', {}).get(product, 0)
            if now_ts < m14_buy:
                fair_value += bias
            if now_ts < m14_sell:
                fair_value -= bias

        imb_shift = self.IMBALANCE_FV_SHIFT.get(product)
        if imb_shift is not None and ask_vol_l1 > 0 and bid_vol_l1 > 0:
            if bid_vol_l1 >= self.IMBALANCE_RATIO_THR * ask_vol_l1:
                fair_value += imb_shift
            elif ask_vol_l1 >= self.IMBALANCE_RATIO_THR * bid_vol_l1:
                fair_value -= imb_shift

        if 'strike' in params and underlying_mid is not None:
            K = params['strike']
            if params.get('use_bs_fv'):
                # Smile-implied IV → Black-Scholes fair value. Captures time
                # value the intrinsic miss (significant for ATM / OTM strikes).
                T = self._track_time(mem, state) if mem is not None else 1.0/365.0
                # Apply OU pull also under BS — same reasoning as for intrinsic.
                ve_lt_mean = lt_means.get('VELVETFRUIT_EXTRACT',
                                          self.LT_MEAN_PRIOR.get('VELVETFRUIT_EXTRACT',
                                                                  underlying_mid))
                w_pull = self.LT_MEAN_BLEND_PER_PRODUCT.get('VELVETFRUIT_EXTRACT', 0.05)
                S_for_bs = (1 - w_pull) * underlying_mid + w_pull * ve_lt_mean

                # Per-tick smile fit if requested; fall back to static SMILE_*.
                if params.get('use_fitted_smile'):
                    coef = self._get_tick_smile(state, S_for_bs, T, mem)
                    if coef is not None:
                        a, b, c = coef
                        m_t = math.log(S_for_bs / K) / math.sqrt(T)
                        sigma = max(0.05, a * m_t * m_t + b * m_t + c)
                    else:
                        sigma = self._smile_iv(S_for_bs, K, T)
                else:
                    sigma = self._smile_iv(S_for_bs, K, T)
                fair_value = bs_call(S_for_bs, K, T, sigma)

                # Re-apply M14 bias AFTER BS overwrites fair_value.
                if mem is not None and product in self.MARK14_BIAS_PER_PRODUCT:
                    bias_re = self.MARK14_BIAS_PER_PRODUCT[product]
                    if now_ts < mem.get('m14_buy_until', {}).get(product, 0):
                        fair_value += bias_re
                    if now_ts < mem.get('m14_sell_until', {}).get(product, 0):
                        fair_value -= bias_re
            else:
                # OU-forecast S: use blend toward VE long-term mean instead of
                # current S. Internally consistent with VE OU strategy — using
                # current S would contradict our own mean-reversion thesis.
                ve_lt_mean = lt_means.get('VELVETFRUIT_EXTRACT',
                                          self.LT_MEAN_PRIOR.get('VELVETFRUIT_EXTRACT',
                                                                  underlying_mid))
                w = self.LT_MEAN_BLEND_PER_PRODUCT.get('VELVETFRUIT_EXTRACT', 0.05)
                forecast_S = (1 - w) * underlying_mid + w * ve_lt_mean
                intrinsic = forecast_S - params['strike']
                if intrinsic > 0:
                    fair_value = intrinsic

        limit = params['limit']
        edge = params['edge']
        skew = params['skew']

        if params.get('nl_skew'):
            pos_ratio = position / limit
            fv_shift = position * skew + pos_ratio * abs(pos_ratio) * limit * skew * self.NL_COEFF
        else:
            fv_shift = position * skew
        # Cap fv_shift so it can never invert quotes across the edge. With a
        # cap of (edge - 1), the take-side condition (bid >= fv_int when long)
        # remains conservative even at max position.
        max_shift = max(0, edge - 1)
        if fv_shift > max_shift:
            fv_shift = max_shift
        elif fv_shift < -max_shift:
            fv_shift = -max_shift
        fv = fair_value - fv_shift
        fv_int = int(round(fv))

        buy_cap = limit - position
        sell_cap = position + limit

        # Book-spread gate: when book is too tight relative to our edge, our
        # take-or-post operations would land at the existing book and get
        # cross-filled by every market trade. Skip ALL activity (take + post)
        # under that regime. Critical for OTM strikes with 1-2 tick books.
        min_book_spread = params.get('min_book_spread')
        cur_book_spread = best_ask - best_bid
        spread_too_tight = (min_book_spread is not None
                            and cur_book_spread < min_book_spread)

        # Strict take: only fire when our FV is genuinely past the book, not at
        # it. ask < fv_int (not <=) means we require real mispricing. Without
        # this, fv_int oscillating with skew across the book causes round-trip
        # losses of edge*0 = 0 to negative on every cycle.
        take_strict = bool(params.get('take_strict'))

        # M14 take-only aggression — boost cross thresholds when M14 signal
        # active for this product. Posts stay at fv_int±edge (clean), only
        # takes are aggressive. Captures M14 alpha without poisoning posts.
        buy_take_thr = fv_int
        sell_take_thr = fv_int
        m14_agg = self.MARK14_TAKE_AGGRESSION_PER_PRODUCT.get(product, 0)
        if m14_agg > 0 and mem is not None:
            if now_ts < mem.get('m14_buy_until', {}).get(product, 0):
                buy_take_thr = fv_int + int(round(m14_agg))
            if now_ts < mem.get('m14_sell_until', {}).get(product, 0):
                sell_take_thr = fv_int - int(round(m14_agg))

        for ask_price in sorted(order_depth.sell_orders.keys()):
            cond = (ask_price < buy_take_thr) if take_strict else (ask_price <= buy_take_thr)
            if cond and buy_cap > 0 and not spread_too_tight:
                vol = -order_depth.sell_orders[ask_price]
                qty = min(vol, buy_cap)
                orders.append(Order(product, ask_price, qty))
                buy_cap -= qty

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            cond = (bid_price > sell_take_thr) if take_strict else (bid_price >= sell_take_thr)
            if cond and sell_cap > 0 and not spread_too_tight:
                vol = order_depth.buy_orders[bid_price]
                qty = min(vol, sell_cap)
                orders.append(Order(product, bid_price, -qty))
                sell_cap -= qty

        # Quote placement (asymmetric pennying for cap-bound products)
        my_bid_intended = fv_int - edge
        my_ask_intended = fv_int + edge

        asym_thr = params.get('asym_pos_thr')
        asym_short = (asym_thr is not None and position <= -asym_thr)
        asym_long  = (asym_thr is not None and position >= asym_thr)

        my_bid = min(my_bid_intended, int(mid) - 1)
        if not asym_long:
            my_bid = max(my_bid, best_bid + 1)
        my_bid = min(my_bid, best_ask - 1)

        my_ask = max(my_ask_intended, int(mid) + 1)
        if not asym_short:
            my_ask = min(my_ask, best_ask - 1)
        my_ask = max(my_ask, best_bid + 1)

        if my_bid >= my_ask:
            my_bid = int(mid) - 1
            my_ask = int(mid) + 1

        skip_ask = (product == 'VELVETFRUIT_EXTRACT' and m67_active)
        skip_bid = False

        # M38 regime defense (deep ITM only — VEV_4000/4500): when book is
        # balanced (|imbal|<0.05) AND spread is at full NPC width, this is
        # uniquely Mark 38 territory (verified: M22 trades these at compressed
        # spread + high imbalance; the two CPs do not overlap on these products).
        m38_skip_spread = params.get('m38_skip_spread')
        if m38_skip_spread is not None:
            cur_spread = best_ask - best_bid
            tot = bid_vol_l1 + ask_vol_l1
            cur_imbal = (bid_vol_l1 - ask_vol_l1) / tot if tot > 0 else 0
            if cur_spread >= m38_skip_spread and abs(cur_imbal) < params.get('m38_skip_imbal', 0.05):
                skip_ask = True
                skip_bid = True

        # Book-spread gate already computed above (spread_too_tight) — applies
        # to both take and post. Cascade to skip flags here.
        if spread_too_tight:
            skip_ask = True
            skip_bid = True

        if buy_cap > 0 and not skip_bid:
            orders.append(Order(product, my_bid, buy_cap))
        if sell_cap > 0 and not skip_ask:
            orders.append(Order(product, my_ask, -sell_cap))

        return orders

    # ====================================================================
    # ===== UNIFIED RUN ==================================================
    # ====================================================================

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        mem = self._load_state(state)

        # ----- HP path (teammate's strategy) -----
        hp_orders = self._hp_run(state, mem)
        if hp_orders:
            result[self.HP_PRODUCT] = hp_orders

        # ----- VE + Options path (krishi's strategy) -----
        T = self._track_time(mem, state)
        S = self._underlying_mid(state)
        lt_means = mem.setdefault('lt_mean', {})

        # Per-option cash ledger for take-profit avg_cost computation.
        # Updates from own_trades since last call.
        opt_cash = mem.setdefault('opt_cash', {})
        for opt in self.OPT_TP_PRICE_THR:
            for tr in state.own_trades.get(opt, []):
                qty = abs(getattr(tr, 'quantity', 0))
                if getattr(tr, 'buyer', None) == 'SUBMISSION':
                    opt_cash[opt] = opt_cash.get(opt, 0) - tr.price * qty
                elif getattr(tr, 'seller', None) == 'SUBMISSION':
                    opt_cash[opt] = opt_cash.get(opt, 0) + tr.price * qty

        # VE counterparty detection
        signal_horizon_ts = self.INFORMED_SIGNAL_TICKS * 100
        for src in (state.market_trades, state.own_trades):
            for tr in src.get('VELVETFRUIT_EXTRACT', []):
                buyer = getattr(tr, 'buyer', None)
                seller = getattr(tr, 'seller', None)
                if buyer == 'Mark 67':
                    mem['m67_buy_until'] = max(mem.get('m67_buy_until', 0), tr.timestamp + signal_horizon_ts)
                if seller == 'Mark 49':
                    mem['m49_sell_until'] = max(mem.get('m49_sell_until', 0), tr.timestamp + signal_horizon_ts)
                if seller == 'Mark 22':
                    mem['m22_sell_until'] = max(mem.get('m22_sell_until', 0), tr.timestamp + signal_horizon_ts)

        # M14 detection — per-product because M14 trades VE AND options.
        m14_horizon_ts = self.MARK14_SIGNAL_TICKS * 100
        m14_buy = mem.setdefault('m14_buy_until', {})
        m14_sell = mem.setdefault('m14_sell_until', {})
        for src in (state.market_trades, state.own_trades):
            for product in (set(self.MARK14_BIAS_PER_PRODUCT)
                            | set(self.MARK14_TAKE_AGGRESSION_PER_PRODUCT)):
                for tr in src.get(product, []):
                    buyer = getattr(tr, 'buyer', None)
                    seller = getattr(tr, 'seller', None)
                    if buyer == 'Mark 14':
                        m14_buy[product] = max(m14_buy.get(product, 0),
                                                tr.timestamp + m14_horizon_ts)
                    if seller == 'Mark 14':
                        m14_sell[product] = max(m14_sell.get(product, 0),
                                                 tr.timestamp + m14_horizon_ts)

        for product, params in self.MM_PARAMS.items():
            if product in state.order_depths:
                result[product] = self.market_make(state, product, params, S, lt_means, state.timestamp, mem)

        for product, params in self.SCALP_PARAMS.items():
            if product in state.order_depths and S is not None:
                result[product] = self.iv_scalp(state, product, params, S, T, mem)

        for product, params in self.BUY_PARAMS.items():
            if product in state.order_depths:
                result[product] = self.buy_flow(state, product, params)

        return result, conversions, json.dumps(mem)