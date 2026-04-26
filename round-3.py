from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math

# ============================================================
#  CONFIGURABLE PARAMETERS
# ============================================================

# --- HYDROGEL_PACK (z-score mean reversion, conservative) ----
# Note: "aggressive" = (500, 1.5, 0.0) had highest backtest PnL but proved
# fragile on day 2 live (lost ~$200 in 1000 ticks). Switching to conservative.
HP                 = "HYDROGEL_PACK"
HP_LIMIT           = 60
HP_WINDOW          = 2000
HP_Z_ENTRY         = 2.0
HP_Z_EXIT          = 0.5      # 0.5 = real flat zone; safer
HP_TAKE_EDGE       = 1
HP_PASSIVE_OFFSET  = 1
HP_ENTRY_CHUNK     = 20       # cap per-tick aggression to avoid walking the book

# --- VELVETFRUIT_EXTRACT (underlying — MM + delta hedge) -----
UND                = "VELVETFRUIT_EXTRACT"
UND_LIMIT          = 200
UND_TAKE_EDGE      = 1
UND_QUOTE_OFFSET   = 1
UND_QUOTE_SIZE     = 30
UND_L2_OFFSET      = 3
UND_L2_SIZE        = 20
UND_SKEW_COEFF     = 0.05
UND_HEDGE_GAIN     = 1.0      # fraction of net option delta to hedge (1.0 = full)

# --- VEV vouchers (smile-residual + asymmetric sizing) -------
# Rationale: cheap-side BUYS proved fragile on day 2 (K=5400 residual was largest
# but didn't converge in short windows; we paid spread × 200 lots and lost $269).
# Rich-side SELLS captured edge on K=5300. So: trust SELLS more than BUYS, scale
# accordingly. Per Orin's hint: "Conviction is not certainty."
# Per Orin's hint: "Conviction is not certainty. Markets remain misaligned far longer
# than anyone expects." Keep option exposure small. The smile residual is real but
# doesn't reliably mean-revert at our holding horizon — the underlying drift dominates.
VEV_LIMIT_SELL     = 30      # max short on rich strike (small per Orin)
VEV_LIMIT_BUY      = 20      # max long on cheap strike (smaller — proved fragile)
                              # Set both to 0 to disable options entirely if needed.
# Reference strikes: used to FIT the smile (no trading on these)
VEV_REF_STRIKES    = [5000, 5100, 5200, 5500]
# Candidate strikes: evaluated AGAINST the reference fit
VEV_TRADE_STRIKES  = [5300, 5400]
# All strikes we attempt to invert IV for
VEV_LIQUID_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
VEV_SKIP           = {"VEV_4000", "VEV_4500", "VEV_6000", "VEV_6500"}

# Trade economics — much stricter than before
VEV_RES_FLOOR      = 0.005    # |residual| below this → no signal
VEV_RES_FULL       = 0.015    # |residual| at-or-above this → full target size
VEV_PRICE_EDGE     = 2        # ticks past theo to AGGRESS (was 1 — too easy)
VEV_MAX_TAKE       = 2        # contracts to TAKE per tick (was 5; still walked book)
VEV_PASSIVE_QUOTE  = True
VEV_PASSIVE_OFFSET = 0

# TTE bookkeeping (CRITICAL: set this correctly for the day you submit)
# Day 0 of round 3: 7. Day 1: 6. Day 2: 5.
# The algo auto-detects day rollovers, so set this for the FIRST day of the run.
VEV_TTE_DAYS_AT_START = 7
VEV_TICKS_PER_DAY     = 1_000_000


# ============================================================
#  TRADER
# ============================================================

class Trader:

    def bid(self) -> int:
        return 2500

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

    def _microprice(self, od: OrderDepth) -> Optional[float]:
        if od and od.buy_orders and od.sell_orders:
            bb = max(od.buy_orders); ba = min(od.sell_orders)
            bv = od.buy_orders[bb]; av = abs(od.sell_orders[ba])
            tot = bv + av
            if tot > 0:
                return (bb * av + ba * bv) / tot
        return self._mid(od)

    def _best_bid_ask(self, od: OrderDepth):
        bb = max(od.buy_orders) if od and od.buy_orders else None
        ba = min(od.sell_orders) if od and od.sell_orders else None
        return bb, ba

    # ---- Black-Scholes ---------------------------------------

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @classmethod
    def _bs_call(cls, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        sqT = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
        d2 = d1 - sigma * sqT
        return S * cls._norm_cdf(d1) - K * cls._norm_cdf(d2)

    @classmethod
    def _bs_delta(cls, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        sqT = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
        return cls._norm_cdf(d1)

    @classmethod
    def _bs_iv(cls, C: float, S: float, K: float, T: float,
               max_iter: int = 50, tol: float = 1e-4) -> Optional[float]:
        intrinsic = max(S - K, 0.0)
        if C <= intrinsic + 1e-6 or T <= 0:
            return None
        lo, hi = 1e-4, 5.0
        for _ in range(max_iter):
            m = 0.5 * (lo + hi)
            p = cls._bs_call(S, K, T, m)
            if abs(p - C) < tol:
                return m
            if p < C:
                lo = m
            else:
                hi = m
        return 0.5 * (lo + hi)

    # ---- linear-algebra: solve 3x3 linear system -------------

    @staticmethod
    def _solve_3x3(A, b):
        def det3(m):
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                  - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                  + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        D = det3(A)
        if abs(D) < 1e-12:
            return None
        out = []
        for col in range(3):
            M = [row[:] for row in A]
            for r in range(3):
                M[r][col] = b[r]
            out.append(det3(M) / D)
        return tuple(out)

    @classmethod
    def _fit_quadratic(cls, ms, ivs):
        """iv = a + b*m + c*m^2. Needs >= 3 points."""
        n = len(ms)
        if n < 3:
            return None
        S0 = float(n)
        S1 = sum(ms); S2 = sum(m*m for m in ms)
        S3 = sum(m**3 for m in ms); S4 = sum(m**4 for m in ms)
        T0 = sum(ivs); T1 = sum(m*v for m,v in zip(ms,ivs)); T2 = sum(m*m*v for m,v in zip(ms,ivs))
        return cls._solve_3x3([[S0,S1,S2],[S1,S2,S3],[S2,S3,S4]], [T0,T1,T2])

    # ---- TTE (handles day rollover) --------------------------

    def _tte_years(self, state: TradingState, data: dict) -> float:
        prev = data.get("vev_prev_ts", -1)
        if state.timestamp < prev:
            data["vev_day"] = data.get("vev_day", 0) + 1
        data["vev_prev_ts"] = state.timestamp
        elapsed = data.get("vev_day", 0) + state.timestamp / VEV_TICKS_PER_DAY
        days_remaining = max(VEV_TTE_DAYS_AT_START - elapsed, 0.01)
        return days_remaining / 252.0

    # ---- HYDROGEL_PACK z-score strategy ----------------------

    def _trade_hydrogel(self, state: TradingState, data: dict) -> List[Order]:
        od = state.order_depths.get(HP, OrderDepth())
        pos = state.position.get(HP, 0)
        mid = self._mid(od)
        if mid is None:
            return []

        hist = data.setdefault("hp_mids", [])
        hist.append(mid)
        cap = HP_WINDOW * 2
        if len(hist) > cap:
            del hist[: len(hist) - cap]
        if len(hist) < HP_WINDOW:
            return []

        recent = hist[-HP_WINDOW:]
        mu = sum(recent) / HP_WINDOW
        var = sum((x - mu) ** 2 for x in recent) / (HP_WINDOW - 1)
        sigma = math.sqrt(var) if var > 1e-9 else 1.0
        z = (mid - mu) / sigma

        target = data.get("hp_target", 0)
        if z > HP_Z_ENTRY:
            target = -HP_LIMIT
        elif z < -HP_Z_ENTRY:
            target = HP_LIMIT
        elif HP_Z_EXIT > 0 and abs(z) < HP_Z_EXIT:
            target = 0
        data["hp_target"] = target

        orders: List[Order] = []
        delta = target - pos

        if delta > 0:
            # Buy toward target — but cap per-tick aggression to avoid walking
            need = delta
            chunk_remaining = HP_ENTRY_CHUNK
            for px in sorted(od.sell_orders.keys()):
                if need <= 0 or chunk_remaining <= 0 or px > mu + HP_TAKE_EDGE:
                    break
                avail = -od.sell_orders[px]
                qty = min(avail, need, chunk_remaining)
                if qty > 0:
                    orders.append(Order(HP, px, qty))
                    need -= qty
                    chunk_remaining -= qty
            # Passive bid for the rest
            if need > 0:
                bb, _ = self._best_bid_ask(od)
                if bb is not None:
                    px = min(int(math.floor(mu - HP_PASSIVE_OFFSET)), bb + 1)
                    orders.append(Order(HP, px, need))
        elif delta < 0:
            need = -delta
            chunk_remaining = HP_ENTRY_CHUNK
            for px in sorted(od.buy_orders.keys(), reverse=True):
                if need <= 0 or chunk_remaining <= 0 or px < mu - HP_TAKE_EDGE:
                    break
                avail = od.buy_orders[px]
                qty = min(avail, need, chunk_remaining)
                if qty > 0:
                    orders.append(Order(HP, px, -qty))
                    need -= qty
                    chunk_remaining -= qty
            if need > 0:
                _, ba = self._best_bid_ask(od)
                if ba is not None:
                    px = max(int(math.ceil(mu + HP_PASSIVE_OFFSET)), ba - 1)
                    orders.append(Order(HP, px, -need))
        else:
            # At target — top up via passive layer
            buy_cap = HP_LIMIT - pos
            sell_cap = HP_LIMIT + pos
            bb, ba = self._best_bid_ask(od)
            if target > 0 and bb is not None and buy_cap > 0:
                px = min(bb + 1, int(math.floor(mu - HP_PASSIVE_OFFSET)))
                orders.append(Order(HP, px, min(10, buy_cap)))
            if target < 0 and ba is not None and sell_cap > 0:
                px = max(ba - 1, int(math.ceil(mu + HP_PASSIVE_OFFSET)))
                orders.append(Order(HP, px, -min(10, sell_cap)))

        return orders

    # ---- Voucher smile-residual strategy ---------------------

    def _signal_strength(self, residual: float) -> float:
        """Scale 0..1 based on |residual| relative to FLOOR/FULL thresholds.
        Hint says: 'Volume should reflect strength of conviction.'"""
        a = abs(residual)
        if a < VEV_RES_FLOOR:
            return 0.0
        if a >= VEV_RES_FULL:
            return 1.0
        return (a - VEV_RES_FLOOR) / (VEV_RES_FULL - VEV_RES_FLOOR)

    def _trade_vouchers(self, state: TradingState, data: dict) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}

        und_od = state.order_depths.get(UND, OrderDepth())
        S = self._mid(und_od)
        if S is None:
            return result

        T = self._tte_years(state, data)
        if T <= 0:
            return result
        sqT = math.sqrt(T)

        # Invert IV at all liquid strikes
        ivs: Dict[int, Tuple[float, float]] = {}   # K -> (iv, opt_mid)
        for K in VEV_LIQUID_STRIKES:
            sym = f"VEV_{K}"
            opt_od = state.order_depths.get(sym)
            if opt_od is None:
                continue
            opt_mid = self._mid(opt_od)
            if opt_mid is None:
                continue
            iv = self._bs_iv(opt_mid, S, float(K), T)
            if iv is not None:
                ivs[K] = (iv, opt_mid)

        if len(ivs) < 4:
            return result

        # Fit smile using ONLY reference strikes (NOT the candidates).
        # This avoids the candidate's IV from absorbing into the fit and
        # underestimating the residual.
        ref_pts = [(K, math.log(K / S) / sqT, ivs[K][0]) for K in VEV_REF_STRIKES if K in ivs]
        if len(ref_pts) < 3:
            return result
        ms_ref = [p[1] for p in ref_pts]
        iv_ref = [p[2] for p in ref_pts]
        coef = self._fit_quadratic(ms_ref, iv_ref)
        if coef is None:
            return result
        a, b, c = coef

        # For each candidate strike: theo from REFERENCE smile, then trade
        net_option_delta = 0.0
        for K in VEV_TRADE_STRIKES:
            sym = f"VEV_{K}"
            if sym in VEV_SKIP or K not in ivs:
                # still capture delta from existing position
                continue
            opt_od = state.order_depths.get(sym)
            if opt_od is None:
                continue

            m = math.log(K / S) / sqT
            smile_iv = a + b * m + c * m * m
            if smile_iv <= 0:
                continue
            theo = self._bs_call(S, float(K), T, smile_iv)
            delta = self._bs_delta(S, float(K), T, smile_iv)
            market_iv = ivs[K][0]
            residual = market_iv - smile_iv

            pos = state.position.get(sym, 0)

            # Magnitude-scaled target with ASYMMETRIC limits (sell > buy conviction)
            strength = self._signal_strength(residual)
            target_pos = 0
            if strength > 0:
                if residual > 0:    # market_iv > smile_iv → option RICH → SELL
                    target_pos = -int(VEV_LIMIT_SELL * strength)
                else:               # market_iv < smile_iv → option CHEAP → BUY (smaller conviction)
                    target_pos = int(VEV_LIMIT_BUY * strength)

            # Per-strike soft limit for cap calc (use absolute max of either)
            strike_limit = max(VEV_LIMIT_SELL, VEV_LIMIT_BUY)
            orders: List[Order] = []
            buy_cap = strike_limit - pos
            sell_cap = strike_limit + pos
            need = target_pos - pos

            if need > 0 and buy_cap > 0:
                # Want more long. Take ONLY where ask < theo - edge, max VEV_MAX_TAKE.
                take_remaining = min(VEV_MAX_TAKE, need, buy_cap)
                for px in sorted(opt_od.sell_orders.keys()):
                    if take_remaining <= 0 or px > theo - VEV_PRICE_EDGE:
                        break
                    avail = -opt_od.sell_orders[px]
                    qty = min(avail, take_remaining)
                    if qty > 0:
                        orders.append(Order(sym, px, qty))
                        take_remaining -= qty
                        pos += qty
                        buy_cap -= qty
                # Passive bid AT theo for the residual capacity (size = remaining `need`)
                if VEV_PASSIVE_QUOTE and need - (target_pos - pos) > 0:
                    bid_px = int(math.floor(theo - VEV_PASSIVE_OFFSET))
                    bb, ba = self._best_bid_ask(opt_od)
                    if bb is not None and ba is not None:
                        # Only post inside the spread (don't cross)
                        if bid_px >= ba:
                            bid_px = ba - 1
                        if bid_px > bb:
                            sz = min(target_pos - pos, buy_cap)
                            if sz > 0:
                                orders.append(Order(sym, bid_px, sz))

            elif need < 0 and sell_cap > 0:
                take_remaining = min(VEV_MAX_TAKE, -need, sell_cap)
                for px in sorted(opt_od.buy_orders.keys(), reverse=True):
                    if take_remaining <= 0 or px < theo + VEV_PRICE_EDGE:
                        break
                    avail = opt_od.buy_orders[px]
                    qty = min(avail, take_remaining)
                    if qty > 0:
                        orders.append(Order(sym, px, -qty))
                        take_remaining -= qty
                        pos -= qty
                        sell_cap -= qty
                if VEV_PASSIVE_QUOTE and pos > target_pos:
                    ask_px = int(math.ceil(theo + VEV_PASSIVE_OFFSET))
                    bb, ba = self._best_bid_ask(opt_od)
                    if bb is not None and ba is not None:
                        if ask_px <= bb:
                            ask_px = bb + 1
                        if ask_px < ba:
                            sz = min(pos - target_pos, sell_cap)
                            if sz > 0:
                                orders.append(Order(sym, ask_px, -sz))

            net_option_delta += pos * delta
            result[sym] = orders

        # Also include delta from non-traded option positions (carryover)
        for K in VEV_LIQUID_STRIKES:
            if K in VEV_TRADE_STRIKES:
                continue
            sym = f"VEV_{K}"
            existing = state.position.get(sym, 0)
            if existing != 0 and K in ivs:
                m = math.log(K / S) / sqT
                smile_iv = a + b * m + c * m * m
                if smile_iv > 0:
                    delta_existing = self._bs_delta(S, float(K), T, smile_iv)
                    net_option_delta += existing * delta_existing

        # Hedge underlying
        und_pos = state.position.get(UND, 0)
        target_und = -int(round(net_option_delta * UND_HEDGE_GAIN))
        if target_und > UND_LIMIT: target_und = UND_LIMIT
        if target_und < -UND_LIMIT: target_und = -UND_LIMIT
        result[UND] = self._trade_underlying(und_od, und_pos, S, target_und)

        return result

    # ---- VELVETFRUIT_EXTRACT MM with hedge-target skew -------

    def _trade_underlying(self, od: OrderDepth, pos: int, mid: float, target: int) -> List[Order]:
        if mid is None:
            return []
        gap = pos - target
        sign = -1 if gap > 0 else (1 if gap < 0 else 0)
        skew = UND_SKEW_COEFF * sign * math.sqrt(abs(gap)) if gap != 0 else 0.0
        fair = mid - skew

        orders: List[Order] = []
        buy_cap = UND_LIMIT - pos
        sell_cap = UND_LIMIT + pos

        # TAKE
        for px in sorted(od.sell_orders.keys()):
            if px > fair - UND_TAKE_EDGE or buy_cap <= 0:
                break
            qty = min(-od.sell_orders[px], buy_cap)
            if qty > 0:
                orders.append(Order(UND, px, qty))
                buy_cap -= qty
        for px in sorted(od.buy_orders.keys(), reverse=True):
            if px < fair + UND_TAKE_EDGE or sell_cap <= 0:
                break
            qty = min(od.buy_orders[px], sell_cap)
            if qty > 0:
                orders.append(Order(UND, px, -qty))
                sell_cap -= qty

        # QUOTE L1 — bias size toward hedge target
        bid_px = int(math.floor(fair - UND_QUOTE_OFFSET))
        ask_px = int(math.ceil(fair + UND_QUOTE_OFFSET))
        if target > pos:
            bid_eff, ask_eff = UND_QUOTE_SIZE, max(5, UND_QUOTE_SIZE // 2)
        elif target < pos:
            bid_eff, ask_eff = max(5, UND_QUOTE_SIZE // 2), UND_QUOTE_SIZE
        else:
            bid_eff = ask_eff = UND_QUOTE_SIZE
        b_sz = min(bid_eff, buy_cap)
        a_sz = min(ask_eff, sell_cap)
        if b_sz > 0 and bid_px < ask_px:
            orders.append(Order(UND, bid_px, b_sz)); buy_cap -= b_sz
        if a_sz > 0 and ask_px > bid_px:
            orders.append(Order(UND, ask_px, -a_sz)); sell_cap -= a_sz

        # QUOTE L2
        if UND_L2_OFFSET > UND_QUOTE_OFFSET and UND_L2_SIZE > 0:
            b2 = min(UND_L2_SIZE, buy_cap)
            a2 = min(UND_L2_SIZE, sell_cap)
            if b2 > 0:
                orders.append(Order(UND, int(math.floor(fair - UND_L2_OFFSET)), b2))
            if a2 > 0:
                orders.append(Order(UND, int(math.ceil(fair + UND_L2_OFFSET)), -a2))

        return orders

    # ---- main entry point ------------------------------------

    def run(self, state: TradingState):
        data = self._load(state.traderData)
        result: Dict[str, List[Order]] = {}

        result[HP] = self._trade_hydrogel(state, data)

        vev_result = self._trade_vouchers(state, data)
        for sym, orders in vev_result.items():
            result[sym] = orders

        return result, 0, self._save(data)
