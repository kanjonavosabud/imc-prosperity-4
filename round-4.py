"""Round-4 VE + options strategy.

Combines:
  - VE + options strategy from logs/532834/532834.py: BOT_PROFILES with
    mirror_of dedup, BS-based ITM FV (use_bs_fv), asymmetric pennying
    (asym_pos_thr), book-spread gate + take_strict for OTM, M38 regime
    defense for VEV_4500, OU-pulled forecast S for non-BS strikes.
  - Adaptive counterparty learning (preserved): live-observed lead-PnL
    EMAs that correct BOT_PROFILES priors on the fly. Returns a DELTA on
    top of _active_bot_bias so total = prior + w*(learned - prior).
    Discovery path for CPs without a BOT_PROFILES entry, capped tighter.

HP is handled separately in round-4-hydrogel.py.
"""
import json
import math
from datamodel import TradingState, Order
from typing import List


# ============================================================================
# Module helpers (BS option pricing + smile fitting)
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

    # === IV smile (532834.py — fitted on R4 historical) ===
    SMILE_A = 0.142979
    SMILE_B = 0.002469
    SMILE_C = 0.235495

    TTE_INITIAL_DAYS = 7.0
    DIFF_EMA_ALPHA = 0.05
    ABS_EMA_ALPHA = 0.01

    # === Counterparty signal-window timer constants ===
    INFORMED_SIGNAL_TICKS = 50
    # Legacy bias constants (BOT_PROFILES drives actual biases below; these
    # are kept for documentation / parity with 532834.py).
    MARK67_BUY_BIAS = 1.5
    MARK49_SELL_BIAS = 0.0   # disabled in 532834.py: 85% redundant with M67 BUY
    MARK22_SELL_BIAS = 0.0   # disabled in 532834.py: 74% redundant with M67 BUY

    MARK14_SIGNAL_TICKS = 10
    # Legacy per-product M14 bias structure. Drives the iteration set in run()
    # for M14 trade detection. Bias values themselves are vestigial — actual
    # biases come from BOT_PROFILES. Kept for parity with 532834.py.
    MARK14_BIAS_PER_PRODUCT = {
        'VELVETFRUIT_EXTRACT': 1.0,
        'VEV_4000': 4.0,
    }
    MARK14_TAKE_AGGRESSION_PER_PRODUCT = {}

    # ====================================================================
    # ===== BOT_PROFILES (verified t-stats from 3-day R4 EDA) ============
    # ====================================================================
    # Each entry: (bot, side, product) -> {bias, duration_ticks, mirror_of, ...}
    # 'mirror_of' indicates this bot is the counterparty side of another bot's
    # event. When both fire on the same paired trade, only the primary signal
    # applies (the mirror is skipped) — avoids double-counting.
    BOT_PROFILES = {
        # ===== Mark 14: PRIMARY informed counterparty =====
        ('Mark 14', 'buyer',  'HYDROGEL_PACK'):
            {'shift_10t': +8.21, 't': +29.03, 'n': 496, 'wr': 0.885,
             'bias': +3.0, 'duration_ticks': 10, 'mirror_of': None},
        ('Mark 14', 'seller', 'HYDROGEL_PACK'):
            {'shift_10t': -8.05, 't': -29.84, 'n': 507, 'wr': 0.919,
             'bias': -3.0, 'duration_ticks': 10, 'mirror_of': None},
        ('Mark 14', 'buyer',  'VELVETFRUIT_EXTRACT'):
            {'shift_10t': +2.18, 't': +14.06, 'n': 316, 'wr': 0.772,
             'bias': +1.0, 'duration_ticks': 10, 'mirror_of': None},
        ('Mark 14', 'seller', 'VELVETFRUIT_EXTRACT'):
            {'shift_10t': -2.29, 't': -13.91, 'n': 331, 'wr': 0.782,
             'bias': -1.0, 'duration_ticks': 10, 'mirror_of': None},
        ('Mark 14', 'buyer',  'VEV_4000'):
            {'shift_10t': +10.54, 't': +49.02, 'n': 232, 'wr': 1.000,
             'bias': +4.0, 'duration_ticks': 10, 'mirror_of': None},
        ('Mark 14', 'seller', 'VEV_4000'):
            {'shift_10t': -10.13, 't': -51.06, 'n': 207, 'wr': 1.000,
             'bias': -4.0, 'duration_ticks': 10, 'mirror_of': None},

        # ===== Mark 38: ANTI-MIRROR of Mark 14 (paired ~70% of M14 events) =====
        ('Mark 38', 'buyer',  'HYDROGEL_PACK'):
            {'shift_10t': -7.93, 't': -29.17, 'n': 515, 'wr': 0.915,
             'bias': -3.0, 'duration_ticks': 10,
             'mirror_of': ('Mark 14', 'seller', 'HYDROGEL_PACK')},
        ('Mark 38', 'seller', 'HYDROGEL_PACK'):
            {'shift_10t': +8.01, 't': +28.05, 'n': 507, 'wr': 0.876,
             'bias': +3.0, 'duration_ticks': 10,
             'mirror_of': ('Mark 14', 'buyer', 'HYDROGEL_PACK')},
        ('Mark 38', 'buyer',  'VEV_4000'):
            {'shift_10t': -10.05, 't': -49.22, 'n': 209, 'wr': 1.000,
             'bias': -4.0, 'duration_ticks': 10,
             'mirror_of': ('Mark 14', 'seller', 'VEV_4000')},
        ('Mark 38', 'seller', 'VEV_4000'):
            {'shift_10t': +10.51, 't': +48.76, 'n': 233, 'wr': 1.000,
             'bias': +4.0, 'duration_ticks': 10,
             'mirror_of': ('Mark 14', 'buyer', 'VEV_4000')},

        # ===== Mark 67: PRIMARY informed buyer on VE =====
        ('Mark 67', 'buyer',  'VELVETFRUIT_EXTRACT'):
            {'shift_10t': +1.45, 't': +6.49, 'n': 165, 'wr': 0.667,
             'bias': +1.5, 'duration_ticks': 50, 'mirror_of': None},

        # ===== Mark 22: profile-only entries (bias=0) =====
        ('Mark 22', 'seller', 'VEV_5400'):
            {'shift_10t': +0.56, 't': +16.5, 'n': 276, 'wr': 0.801,
             'bias': 0.0, 'duration_ticks': 10, 'mirror_of': None},
        ('Mark 22', 'seller', 'VEV_5500'):
            {'shift_10t': +0.51, 't': +29.3, 'n': 306, 'wr': 0.935,
             'bias': 0.0, 'duration_ticks': 10, 'mirror_of': None},

        # ===== M49 / M22 SELL VE: redundant with M67 BUY (85% / 74% same trade) =====
        ('Mark 49', 'seller', 'VELVETFRUIT_EXTRACT'):
            {'shift_10t': +1.47, 't': +5.26, 'n': 105, 'wr': 0.657,
             'bias': 0.0, 'duration_ticks': 10,
             'mirror_of': ('Mark 67', 'buyer', 'VELVETFRUIT_EXTRACT')},
        ('Mark 22', 'seller', 'VELVETFRUIT_EXTRACT'):
            {'shift_10t': +0.85, 't': +2.89, 'n': 101, 'wr': 0.614,
             'bias': 0.0, 'duration_ticks': 10,
             'mirror_of': ('Mark 67', 'buyer', 'VELVETFRUIT_EXTRACT')},
    }

    # ====================================================================
    # ===== ADAPTIVE COUNTERPARTY LEARNING (live correction layer) =======
    # ====================================================================
    # Per (cp, product, side) we observe expected mid-change (m_now - m0)
    # over ADAPTIVE_HORIZON_TICKS after CP fires. Stored as EMA + cumulative
    # qty in mem['cp_stats']. _adaptive_bias returns a DELTA on top of the
    # static BOT_PROFILES priors, so the total per-CP contribution is:
    #     prior + w * (learned - prior)
    # which converges to the prior at qty=0 and to the learned mean as
    # qty → ∞. CPs without a BOT_PROFILES entry can still emit a learned
    # bias once ADAPTIVE_DISCOVER_MIN_QTY observed (capped tighter).
    ADAPTIVE_HORIZON_TICKS = 50
    ADAPTIVE_MIN_QTY = 30
    ADAPTIVE_QTY_SCALE = 80
    ADAPTIVE_EMA_ALPHA = 0.10
    ADAPTIVE_MAX_BIAS = 4.0
    ADAPTIVE_MIN_ABS = 0.4
    ADAPTIVE_DISCOVER_MIN_QTY = 60
    ADAPTIVE_DISCOVER_CAP = 2.0
    ADAPTIVE_HARD_CAP_BUCKETS = 60
    ADAPTIVE_PRODUCTS = ('VELVETFRUIT_EXTRACT',)

    # ====================================================================
    # ===== MM PARAMS ====================================================
    # ====================================================================
    MM_PARAMS = {
        # Deep ITM: BS-based FV with smile IV. M38 regime defense on V4500.
        'VEV_4000': {'limit': 300, 'edge': 12, 'skew': 0.0, 'strike': 4000,
                     'use_bs_fv': True},
        'VEV_4500': {'limit': 300, 'edge': 4,  'skew': 0.0, 'strike': 4500,
                     'm38_skip_spread': 16, 'm38_skip_imbal': 0.05,
                     'use_bs_fv': True},
        # ATM / near-ATM: microprice MM with asymmetric pennying.
        'VEV_5000': {'limit': 50, 'edge': 2, 'skew': 0.05, 'asym_pos_thr': 15},
        'VEV_5100': {'limit': 50, 'edge': 2, 'skew': 0.05, 'asym_pos_thr': 15},
        # OTM: book-spread gate + strict take. Quote only when spread wide enough
        # that we land inside, never at/inside the existing book.
        'VEV_5200': {'limit': 25, 'edge': 2, 'skew': 0.06, 'asym_pos_thr': 10,
                     'min_book_spread': 5, 'take_strict': True},
        'VEV_5300': {'limit': 20, 'edge': 2, 'skew': 0.08, 'asym_pos_thr': 8,
                     'min_book_spread': 5, 'take_strict': True},
        'VEV_5400': {'limit': 15, 'edge': 2, 'skew': 0.10, 'asym_pos_thr': 6,
                     'min_book_spread': 5, 'take_strict': True},
        'VEV_5500': {'limit': 15, 'edge': 2, 'skew': 0.10, 'asym_pos_thr': 6,
                     'min_book_spread': 5, 'take_strict': True},
        # Underlying
        'VELVETFRUIT_EXTRACT': {'limit': 200, 'edge': 4, 'skew': 0.003},
    }

    NL_COEFF = 0.8

    # OU mean-reversion for VE only (HP OU lives in round-4-hydrogel.py).
    LT_MEAN_PRIOR = {'VELVETFRUIT_EXTRACT': 5250.0}
    LT_MEAN_BLEND_PER_PRODUCT = {
        'VELVETFRUIT_EXTRACT': 0.10,  # 532834.py tuning (was 0.07 in our prior)
    }
    LT_MEAN_ALPHA = 0.001

    IMBALANCE_FV_SHIFT = {}
    IMBALANCE_RATIO_THR = 999.0

    SCALP_PARAMS = {}
    BUY_PARAMS = {}

    # === Take-profit ledger (computed per-tick; firing not currently wired) ===
    OPT_TP_PRICE_THR = {
        'VEV_4000': 8, 'VEV_4500': 5, 'VEV_5000': 3, 'VEV_5100': 3,
    }
    OPT_TP_POS_THR = 5

    _OPT_STRIKE_LIST = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500)

    # ====================================================================
    # ===== STATE / TIME =================================================
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

    def _get_tick_smile(self, state: TradingState, S: float, T: float, mem: dict):
        """Per-tick smile fit. Cached by timestamp so each option call uses the
        same fit. Fall-back to static SMILE_* coefs when fit is degenerate."""
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
                continue
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

    def _book_mid(self, state: TradingState, product: str):
        od = state.order_depths.get(product)
        if not od or not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0

    # ====================================================================
    # ===== BOT_PROFILES INFRASTRUCTURE (static priors) ==================
    # ====================================================================

    def _update_bot_signals(self, mem: dict, state: TradingState) -> None:
        """Set per-bot signal-until timestamps from this tick's trades.
        JSON-safe string keys so traderData round-trips."""
        until = mem.setdefault('bot_signal_until', {})
        for src in (state.market_trades, state.own_trades):
            for product, trades in src.items():
                for tr in trades:
                    ts = tr.timestamp
                    buyer = getattr(tr, 'buyer', None)
                    seller = getattr(tr, 'seller', None)
                    for side, bot in (('buyer', buyer), ('seller', seller)):
                        if not bot or bot == 'SUBMISSION':
                            continue
                        key = (bot, side, product)
                        prof = self.BOT_PROFILES.get(key)
                        if prof is None:
                            continue
                        duration_ts = prof.get('duration_ticks', 10) * 100
                        new_until = ts + duration_ts
                        skey = f"{bot}|{side}|{product}"
                        if new_until > until.get(skey, 0):
                            until[skey] = new_until

    def _active_bot_bias(self, mem: dict, product: str, now_ts: int) -> float:
        """Sum of FV biases from active BOT_PROFILES entries on `product`,
        with mirror_of dedup. If a 'secondary' signal's mirror primary is also
        active, skip the secondary."""
        until = mem.get('bot_signal_until', {})
        active = set()
        for skey, t in until.items():
            if now_ts >= t:
                continue
            try:
                bot, side, prod = skey.split('|', 2)
            except ValueError:
                continue
            if prod != product:
                continue
            active.add((bot, side, prod))
        total = 0.0
        for key in active:
            prof = self.BOT_PROFILES.get(key, {})
            mirror = prof.get('mirror_of')
            if mirror is not None and mirror in active:
                continue  # primary is firing → skip the mirror
            total += prof.get('bias', 0.0)
        return total

    # ====================================================================
    # ===== ADAPTIVE LEARNING LAYER (live correction on top of priors) ===
    # ====================================================================

    def _record_and_drain_adaptive(self, mem: dict, state: TradingState) -> None:
        """Queue new SUBMISSION-vs-Mark trades for ADAPTIVE_PRODUCTS into
        pending_obs; drain matured ones (horizon elapsed) into cp_stats.
        Dedupes via seen_keys (TTL = horizon). Bounded memory: pending_obs
        and seen_keys live ≤ horizon; cp_stats hard-capped at
        ADAPTIVE_HARD_CAP_BUCKETS."""
        H_ts = self.ADAPTIVE_HORIZON_TICKS * 100
        pending = mem.setdefault('pending_obs', [])
        seen = mem.setdefault('seen_keys', {})
        stats = mem.setdefault('cp_stats', {})

        # 1. Record new observations
        for src in (state.market_trades, state.own_trades):
            for prod, trades in src.items():
                if prod not in self.ADAPTIVE_PRODUCTS:
                    continue
                for tr in trades:
                    buyer = getattr(tr, 'buyer', '') or ''
                    seller = getattr(tr, 'seller', '') or ''
                    if buyer == 'SUBMISSION':
                        cp, side = seller, -1
                    elif seller == 'SUBMISSION':
                        cp, side = buyer, +1
                    else:
                        continue
                    if not cp.startswith('Mark'):
                        continue
                    px = getattr(tr, 'price', 0)
                    qty = abs(getattr(tr, 'quantity', 0))
                    if qty == 0:
                        continue
                    key = f"{prod[:4]}|{tr.timestamp}|{cp[-2:]}|{side}|{int(px)}|{qty}"
                    if key in seen:
                        continue
                    mid0 = self._book_mid(state, prod)
                    if mid0 is None:
                        continue
                    pending.append([cp, prod, side, tr.timestamp, mid0, qty])
                    seen[key] = tr.timestamp + H_ts + 1000

        # 2. Drain matured observations
        keep = []
        for entry in pending:
            cp, prod, side, ts0, mid0, qty = entry
            if state.timestamp - ts0 < H_ts:
                keep.append(entry)
                continue
            mid_now = self._book_mid(state, prod)
            if mid_now is None:
                continue
            # Stats are bucketed per (cp, prod, side); store raw expected Δmid
            # so units match BOT_PROFILES['bias'] (also raw Δmid).
            realized = mid_now - mid0
            skey = f"{cp}|{prod}|{side:+d}"
            st = stats.get(skey)
            if st is None:
                stats[skey] = {'lead': realized, 'qty': qty}
            else:
                a = self.ADAPTIVE_EMA_ALPHA
                st['lead'] = (1 - a) * st['lead'] + a * realized
                st['qty'] += qty
        mem['pending_obs'] = keep

        # 3. Prune expired seen keys
        mem['seen_keys'] = {k: v for k, v in seen.items() if v > state.timestamp}

        # 4. Cap stats size (keep highest-qty buckets)
        if len(stats) > self.ADAPTIVE_HARD_CAP_BUCKETS:
            top = sorted(stats.items(), key=lambda kv: kv[1]['qty'], reverse=True)
            mem['cp_stats'] = dict(top[:self.ADAPTIVE_HARD_CAP_BUCKETS])

    def _adaptive_bias(self, mem: dict, product: str, now_ts: int) -> float:
        """Returns the learned DELTA correction on top of _active_bot_bias's
        static priors (so callers add both: prior + delta = blended bias).

        For each active (cp, product, side):
          - prior = BOT_PROFILES[(cp, side_word, product)]['bias'] (or 0 if absent)
          - learned mean = cp_stats[key]['lead']
          - blended bias = prior + w * (learned - prior), with w ramping with qty
          - this method returns ONLY the delta = w * (learned - prior).

        Mirror_of dedup is honored: if a (secondary, primary) mirror pair is
        both active, only the primary contributes (matches _active_bot_bias).
        Discovery: when prior == 0 (no BOT_PROFILES entry or bias=0), require
        ADAPTIVE_DISCOVER_MIN_QTY before contributing, with tighter cap."""
        if product not in self.ADAPTIVE_PRODUCTS:
            return 0.0
        active_map = mem.get('cp_active', {})
        if not active_map:
            return 0.0
        stats = mem.get('cp_stats', {})

        # Build set of active (cp, side_word, prod) triples for mirror_of dedup.
        active = set()
        for key, until in active_map.items():
            if until <= now_ts:
                continue
            try:
                cp, prod, side_str = key.split('|')
                side = int(side_str)
            except (ValueError, AttributeError):
                continue
            side_word = 'buyer' if side == +1 else 'seller'
            active.add((cp, side_word, prod))

        total = 0.0
        for triple in active:
            cp, side_word, prod = triple
            if prod != product:
                continue
            profile = self.BOT_PROFILES.get(triple, {})
            mirror = profile.get('mirror_of')
            if mirror is not None and mirror in active:
                continue  # primary is firing → skip the mirror's delta
            prior = profile.get('bias', 0.0)
            side = +1 if side_word == 'buyer' else -1
            skey = f"{cp}|{prod}|{side:+d}"
            st = stats.get(skey)
            if st is None:
                continue
            if prior == 0.0:
                # Pure discovery (no prior). Stricter qty bar + tighter cap.
                if st['qty'] < self.ADAPTIVE_DISCOVER_MIN_QTY:
                    continue
                delta = max(-self.ADAPTIVE_DISCOVER_CAP,
                            min(self.ADAPTIVE_DISCOVER_CAP, st['lead']))
            else:
                if st['qty'] < self.ADAPTIVE_MIN_QTY:
                    continue
                w = min(1.0, (st['qty'] - self.ADAPTIVE_MIN_QTY) / self.ADAPTIVE_QTY_SCALE)
                delta = w * (st['lead'] - prior)
            if abs(delta) < self.ADAPTIVE_MIN_ABS:
                continue
            delta = max(-self.ADAPTIVE_MAX_BIAS, min(self.ADAPTIVE_MAX_BIAS, delta))
            total += delta
        return total

    # ====================================================================
    # ===== MAIN ENTRY ====================================================
    # ====================================================================

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        mem = self._load_state(state)
        T = self._track_time(mem, state)
        S = self._underlying_mid(state)
        lt_means = mem.setdefault('lt_mean', {})

        # Update BOT_PROFILES signal-until timestamps from this tick's trades.
        self._update_bot_signals(mem, state)

        # Per-option cash ledger for take-profit avg_cost computation.
        opt_cash = mem.setdefault('opt_cash', {})
        for opt in self.OPT_TP_PRICE_THR:
            for tr in state.own_trades.get(opt, []):
                qty = abs(getattr(tr, 'quantity', 0))
                if getattr(tr, 'buyer', None) == 'SUBMISSION':
                    opt_cash[opt] = opt_cash.get(opt, 0) - tr.price * qty
                elif getattr(tr, 'seller', None) == 'SUBMISSION':
                    opt_cash[opt] = opt_cash.get(opt, 0) + tr.price * qty

        # Legacy VE counterparty timers + adaptive cp_active map. The legacy
        # m67_buy_until still drives the binary skip_ask gate in market_make
        # (separate decision dimension from FV bias magnitude).
        signal_horizon_ts = self.INFORMED_SIGNAL_TICKS * 100
        cp_active = mem.setdefault('cp_active', {})
        for src in (state.market_trades, state.own_trades):
            for tr in src.get('VELVETFRUIT_EXTRACT', []):
                buyer = getattr(tr, 'buyer', None)
                seller = getattr(tr, 'seller', None)
                if buyer == 'Mark 67':
                    mem['m67_buy_until'] = max(mem.get('m67_buy_until', 0),
                                               tr.timestamp + signal_horizon_ts)
                # Adaptive: any Mark trade flags (cp, product, side) active.
                if buyer == 'SUBMISSION' and seller and seller.startswith('Mark'):
                    k = f"{seller}|VELVETFRUIT_EXTRACT|-1"
                    cp_active[k] = max(cp_active.get(k, 0),
                                       tr.timestamp + signal_horizon_ts)
                elif seller == 'SUBMISSION' and buyer and buyer.startswith('Mark'):
                    k = f"{buyer}|VELVETFRUIT_EXTRACT|+1"
                    cp_active[k] = max(cp_active.get(k, 0),
                                       tr.timestamp + signal_horizon_ts)
        mem['cp_active'] = {k: v for k, v in cp_active.items() if v > state.timestamp}

        # Per-product M14 detection (legacy take-aggression structure).
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

        # Adaptive lead-PnL stats from observed VE trades.
        self._record_and_drain_adaptive(mem, state)

        for product, params in self.MM_PARAMS.items():
            if product in state.order_depths:
                result[product] = self.market_make(state, product, params, S,
                                                   lt_means, state.timestamp, mem)

        for product, params in self.SCALP_PARAMS.items():
            if product in state.order_depths and S is not None:
                result[product] = self.iv_scalp(state, product, params, S, T, mem)

        for product, params in self.BUY_PARAMS.items():
            if product in state.order_depths:
                result[product] = self.buy_flow(state, product, params)

        return result, conversions, json.dumps(mem)

    # ====================================================================
    # ===== MARKET MAKE / SCALP / BUY_FLOW ===============================
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

        # m67_active still tracked for the binary skip_ask gate (separate from
        # the magnitude-driving FV bias path).
        m67_active = (product == 'VELVETFRUIT_EXTRACT' and mem is not None
                      and now_ts < mem.get('m67_buy_until', 0))

        # Counterparty FV bias = static BOT_PROFILES priors + adaptive learned
        # delta. With mirror_of dedup honored by both functions. Adaptive only
        # fires for ADAPTIVE_PRODUCTS (currently VE).
        if mem is not None:
            fair_value += self._active_bot_bias(mem, product, now_ts)
            fair_value += self._adaptive_bias(mem, product, now_ts)

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
                ve_lt_mean = lt_means.get('VELVETFRUIT_EXTRACT',
                                          self.LT_MEAN_PRIOR.get('VELVETFRUIT_EXTRACT',
                                                                  underlying_mid))
                w_pull = self.LT_MEAN_BLEND_PER_PRODUCT.get('VELVETFRUIT_EXTRACT', 0.05)
                S_for_bs = (1 - w_pull) * underlying_mid + w_pull * ve_lt_mean
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
                # Re-apply ALL active CP biases AFTER BS overwrites fair_value.
                if mem is not None:
                    fair_value += self._active_bot_bias(mem, product, now_ts)
                    fair_value += self._adaptive_bias(mem, product, now_ts)
            else:
                # OU-pulled forecast S for non-BS strikes: blend toward VE long-
                # term mean instead of using current S directly. Internally
                # consistent with VE OU strategy.
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
        # Cap fv_shift so it can never invert quotes across the edge.
        max_shift = max(0, edge - 1)
        if fv_shift > max_shift:
            fv_shift = max_shift
        elif fv_shift < -max_shift:
            fv_shift = -max_shift
        fv = fair_value - fv_shift
        fv_int = int(round(fv))

        buy_cap = limit - position
        sell_cap = position + limit

        # Book-spread gate: skip ALL activity when book is too tight relative
        # to our edge (would just cross-fill at the existing book).
        min_book_spread = params.get('min_book_spread')
        cur_book_spread = best_ask - best_bid
        spread_too_tight = (min_book_spread is not None
                            and cur_book_spread < min_book_spread)
        # Strict take: only fire when our FV is genuinely past the book.
        take_strict = bool(params.get('take_strict'))

        # M14 take-only aggression (post-clean; takes are aggressive).
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

        # Asymmetric pennying for cap-bound products
        my_bid_intended = fv_int - edge
        my_ask_intended = fv_int + edge
        asym_thr = params.get('asym_pos_thr')
        asym_short = (asym_thr is not None and position <= -asym_thr)
        asym_long = (asym_thr is not None and position >= asym_thr)

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

        # M38 regime defense (deep ITM only — VEV_4500 by default): when book
        # is balanced (|imbal|<thr) AND spread at full NPC width, this is M38
        # territory. Skip both quotes to avoid being adverse-selected.
        m38_skip_spread = params.get('m38_skip_spread')
        if m38_skip_spread is not None:
            cur_spread = best_ask - best_bid
            tot = bid_vol_l1 + ask_vol_l1
            cur_imbal = (bid_vol_l1 - ask_vol_l1) / tot if tot > 0 else 0
            if cur_spread >= m38_skip_spread and abs(cur_imbal) < params.get('m38_skip_imbal', 0.05):
                skip_ask = True
                skip_bid = True

        # Cascade book-spread gate to skip flags too.
        if spread_too_tight:
            skip_ask = True
            skip_bid = True

        if buy_cap > 0 and not skip_bid:
            orders.append(Order(product, my_bid, buy_cap))
        if sell_cap > 0 and not skip_ask:
            orders.append(Order(product, my_ask, -sell_cap))

        return orders
