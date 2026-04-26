from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math

# ============================================================
#  CONFIGURABLE PARAMETERS — tweak between runs
# ============================================================

# --- HYDROGEL_PACK (z-score mean reversion) ------------------
# OPTIMUM (highest raw PnL, fragile): WINDOW=500, Z_ENTRY=1.5, Z_EXIT=0.0
# CONSERVATIVE (lower PnL, lower DD):  WINDOW=2000, Z_ENTRY=2.0, Z_EXIT=0.5
HP                = "HYDROGEL_PACK"
HP_LIMIT          = 60
HP_WINDOW         = 500
HP_Z_ENTRY        = 1.5
HP_Z_EXIT         = 0.0      # 0.0 = always-on; >0 = flat zone (set 0.5 with WINDOW=2000 for conservative)
HP_TAKE_EDGE      = 1        # ticks past fair to aggress on
HP_PASSIVE_OFFSET = 1        # passive quote distance from rolling mean

# --- VELVETFRUIT_EXTRACT (underlying — MM + delta hedge) -----
UND               = "VELVETFRUIT_EXTRACT"
UND_LIMIT         = 200
UND_TAKE_EDGE     = 1
UND_QUOTE_OFFSET  = 1
UND_QUOTE_SIZE    = 30
UND_L2_OFFSET     = 3
UND_L2_SIZE       = 20
UND_SKEW_COEFF    = 0.05     # inventory skew

# --- VEV_xxxx vouchers (smile-residual stat-arb) -------------
VEV_LIMIT         = 200
VEV_STRIKES       = [5000, 5100, 5200, 5300, 5400, 5500]
VEV_TRADE_STRIKES = [5300, 5400, 5500]   # actively trade these (smile dislocation)
VEV_RES_ENTER     = 0.003    # |IV - smile_fit| threshold to take liquidity (vol points)
VEV_PRICE_EDGE    = 1        # min ticks past theo to take
VEV_MAX_LIFT      = 30       # max contracts to lift in one tick per strike
VEV_TTE_DAYS_AT_START = 7    # days remaining at the start of the FIRST day of submission
VEV_TICKS_PER_DAY = 1_000_000

# Skip these strikes (deep ITM noisy or stuck-at-floor)
VEV_SKIP          = {"VEV_4000", "VEV_4500", "VEV_6000", "VEV_6500"}


# ============================================================
#  TRADER
# ============================================================

class Trader:

    # ---- MAF Auction Bid (carryover from R2) -----------------

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

    def _best_bid_ask(self, od: OrderDepth):
        bb = max(od.buy_orders) if od and od.buy_orders else None
        ba = min(od.sell_orders) if od and od.sell_orders else None
        return bb, ba

    # ---- Black-Scholes utilities -----------------------------

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
        # Bisection: vol in [1e-4, 5.0]
        lo, hi = 1e-4, 5.0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            price = cls._bs_call(S, K, T, mid)
            if abs(price - C) < tol:
                return mid
            if price < C:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def _fit_quadratic_smile(ms: List[float], ivs: List[float]):
        """Least squares fit iv = a + b*m + c*m^2. Returns (a, b, c)."""
        n = len(ms)
        if n < 3:
            return None
        # Normal equations
        S0 = float(n)
        S1 = sum(ms); S2 = sum(m*m for m in ms)
        S3 = sum(m*m*m for m in ms); S4 = sum(m*m*m*m for m in ms)
        T0 = sum(ivs)
        T1 = sum(m*v for m, v in zip(ms, ivs))
        T2 = sum(m*m*v for m, v in zip(ms, ivs))
        # Solve 3x3
        # [S0 S1 S2] [a]   [T0]
        # [S1 S2 S3] [b] = [T1]
        # [S2 S3 S4] [c]   [T2]
        A = [[S0, S1, S2], [S1, S2, S3], [S2, S3, S4]]
        b = [T0, T1, T2]
        return Trader._solve_3x3(A, b)

    @staticmethod
    def _solve_3x3(A, b):
        # Cramer's rule
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

    # ---- TTE bookkeeping (handles day rollover) --------------

    def _tte_years(self, state: TradingState, data: dict) -> float:
        prev = data.get("vev_prev_ts", -1)
        if state.timestamp < prev:
            data["vev_day"] = data.get("vev_day", 0) + 1
        data["vev_prev_ts"] = state.timestamp
        elapsed_days = data.get("vev_day", 0) + state.timestamp / VEV_TICKS_PER_DAY
        days_remaining = max(VEV_TTE_DAYS_AT_START - elapsed_days, 0.01)
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
        # cap at 2x window so the rolling mean is stable
        cap = HP_WINDOW * 2
        if len(hist) > cap:
            del hist[: len(hist) - cap]

        if len(hist) < HP_WINDOW:
            return []

        recent = hist[-HP_WINDOW:]
        mu = sum(recent) / HP_WINDOW
        var = sum((x - mu) * (x - mu) for x in recent) / (HP_WINDOW - 1)
        sigma = math.sqrt(var) if var > 1e-9 else 1.0
        z = (mid - mu) / sigma

        # Determine target position (z_exit=0 keeps prior target)
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

        # Aggressive take to flip toward target
        if delta > 0:
            remaining = delta
            for px in sorted(od.sell_orders.keys()):
                if remaining <= 0 or px > mu + HP_TAKE_EDGE:
                    break
                avail = -od.sell_orders[px]
                qty = min(avail, remaining)
                if qty > 0:
                    orders.append(Order(HP, px, qty))
                    remaining -= qty
            # Passive bid at fair - offset for the rest (catch incoming sells)
            if remaining > 0:
                bb, _ = self._best_bid_ask(od)
                if bb is not None:
                    px = min(int(math.floor(mu - HP_PASSIVE_OFFSET)), bb + 1)
                    orders.append(Order(HP, px, remaining))
        elif delta < 0:
            remaining = -delta
            for px in sorted(od.buy_orders.keys(), reverse=True):
                if remaining <= 0 or px < mu - HP_TAKE_EDGE:
                    break
                avail = od.buy_orders[px]
                qty = min(avail, remaining)
                if qty > 0:
                    orders.append(Order(HP, px, -qty))
                    remaining -= qty
            if remaining > 0:
                _, ba = self._best_bid_ask(od)
                if ba is not None:
                    px = max(int(math.ceil(mu + HP_PASSIVE_OFFSET)), ba - 1)
                    orders.append(Order(HP, px, -remaining))
        else:
            # At target: post passive layer to top-up against drift / capture spread
            buy_cap = HP_LIMIT - pos
            sell_cap = HP_LIMIT + pos
            bb, ba = self._best_bid_ask(od)
            if bb is not None and buy_cap > 0 and target > 0:
                # we want to be long; post passive bid one tick inside
                px = min(bb + 1, int(math.floor(mu - HP_PASSIVE_OFFSET)))
                orders.append(Order(HP, px, min(10, buy_cap)))
            if ba is not None and sell_cap > 0 and target < 0:
                px = max(ba - 1, int(math.ceil(mu + HP_PASSIVE_OFFSET)))
                orders.append(Order(HP, px, -min(10, sell_cap)))

        return orders

    # ---- VEV smile-residual strategy + delta hedge ----------

    def _trade_vouchers(self, state: TradingState, data: dict) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}

        # 1. Underlying mid
        und_od = state.order_depths.get(UND, OrderDepth())
        S = self._mid(und_od)
        if S is None:
            return result

        # 2. TTE
        T = self._tte_years(state, data)
        if T <= 0:
            return result
        sqT = math.sqrt(T)

        # 3. Per-strike IV from option mid
        ivs: Dict[int, float] = {}
        for K in VEV_STRIKES:
            sym = f"VEV_{K}"
            opt_od = state.order_depths.get(sym)
            if opt_od is None:
                continue
            opt_mid = self._mid(opt_od)
            if opt_mid is None:
                continue
            iv = self._bs_iv(opt_mid, S, float(K), T)
            if iv is not None:
                ivs[K] = iv

        if len(ivs) < 4:
            return result

        # 4. Fit quadratic smile across all liquid strikes
        ks_sorted = sorted(ivs.keys())
        ms = [math.log(K / S) / sqT for K in ks_sorted]
        iv_vec = [ivs[K] for K in ks_sorted]
        coef = self._fit_quadratic_smile(ms, iv_vec)
        if coef is None:
            return result
        a, b, c = coef

        # 5. For each tradeable strike, take liquidity vs theo
        net_option_delta = 0.0
        for K in VEV_TRADE_STRIKES:
            if K not in ivs:
                continue
            sym = f"VEV_{K}"
            if sym in VEV_SKIP:
                continue
            opt_od = state.order_depths.get(sym)
            if opt_od is None:
                continue

            m = math.log(K / S) / sqT
            smile_iv = a + b * m + c * m * m
            theo = self._bs_call(S, float(K), T, max(smile_iv, 1e-4))
            delta = self._bs_delta(S, float(K), T, max(smile_iv, 1e-4))
            residual = ivs[K] - smile_iv

            pos = state.position.get(sym, 0)
            buy_cap = VEV_LIMIT - pos
            sell_cap = VEV_LIMIT + pos
            orders: List[Order] = []

            # Only trade if smile residual exceeds threshold
            if abs(residual) >= VEV_RES_ENTER:
                if residual < 0:
                    # market IV < smile IV → option is CHEAP → BUY by lifting offers below theo
                    if buy_cap > 0:
                        remaining = min(buy_cap, VEV_MAX_LIFT)
                        for px in sorted(opt_od.sell_orders.keys()):
                            if remaining <= 0 or px > theo - VEV_PRICE_EDGE:
                                break
                            avail = -opt_od.sell_orders[px]
                            qty = min(avail, remaining)
                            if qty > 0:
                                orders.append(Order(sym, px, qty))
                                remaining -= qty
                                pos += qty       # local mark for net delta
                else:
                    # residual > 0: option RICH → SELL by hitting bids above theo
                    if sell_cap > 0:
                        remaining = min(sell_cap, VEV_MAX_LIFT)
                        for px in sorted(opt_od.buy_orders.keys(), reverse=True):
                            if remaining <= 0 or px < theo + VEV_PRICE_EDGE:
                                break
                            avail = opt_od.buy_orders[px]
                            qty = min(avail, remaining)
                            if qty > 0:
                                orders.append(Order(sym, px, -qty))
                                remaining -= qty
                                pos -= qty

            net_option_delta += pos * delta
            result[sym] = orders

        # 6. Delta-hedge with underlying (also acts as a market-making layer)
        und_pos = state.position.get(UND, 0)
        target_und = -net_option_delta  # neutralize option delta
        # Clamp to limit
        if target_und > UND_LIMIT: target_und = UND_LIMIT
        if target_und < -UND_LIMIT: target_und = -UND_LIMIT

        und_orders = self._trade_underlying(und_od, und_pos, S, target_und)
        result[UND] = und_orders

        # ensure all option symbols have an entry (even empty) — not strictly required,
        # but keeps the orders dict clean
        return result

    # ---- VELVETFRUIT_EXTRACT MM + hedge target ---------------

    def _trade_underlying(self, od: OrderDepth, pos: int, mid: float, target: float) -> List[Order]:
        if mid is None:
            return []
        # Skew fair toward target (encourages flow toward delta-hedge target)
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

        # QUOTE L1
        bid_px = int(math.floor(fair - UND_QUOTE_OFFSET))
        ask_px = int(math.ceil(fair + UND_QUOTE_OFFSET))
        # Bias size toward the hedge target side
        bias = 1.0
        if target > pos:
            bid_sz_eff = UND_QUOTE_SIZE
            ask_sz_eff = max(5, UND_QUOTE_SIZE // 2)
        elif target < pos:
            bid_sz_eff = max(5, UND_QUOTE_SIZE // 2)
            ask_sz_eff = UND_QUOTE_SIZE
        else:
            bid_sz_eff = ask_sz_eff = UND_QUOTE_SIZE

        b_sz = min(bid_sz_eff, buy_cap)
        a_sz = min(ask_sz_eff, sell_cap)
        if b_sz > 0 and bid_px < ask_px:
            orders.append(Order(UND, bid_px, b_sz))
            buy_cap -= b_sz
        if a_sz > 0 and ask_px > bid_px:
            orders.append(Order(UND, ask_px, -a_sz))
            sell_cap -= a_sz

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

        # Strategy B: HYDROGEL_PACK mean reversion
        result[HP] = self._trade_hydrogel(state, data)

        # Strategy A: VEV smile-residual + UND delta hedge
        vev_result = self._trade_vouchers(state, data)
        for sym, orders in vev_result.items():
            result[sym] = orders

        conversions = 0
        return result, conversions, self._save(data)