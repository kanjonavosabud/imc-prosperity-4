from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math

# ============================================================
#  HYDROGEL_PACK — z-score mean reversion (proven baseline)
# ============================================================
# In offline 3-day backtest: +16,104 PnL with these params.
# Strategy: classic MM with z-score-driven position target.
#   - Rolling mean μ, std σ over WINDOW ticks
#   - z = (mid - μ) / σ
#   - z > +Z_ENTRY: target = -LIMIT (price too high → sell)
#   - z < -Z_ENTRY: target = +LIMIT (price too low → buy)
#   - |z| < Z_EXIT:  target = 0 (close out)
#   - In between: HOLD prior target (hysteresis)
#
# Execution: chunk-capped aggressive take when price is past μ ± edge,
# then passive limit posting inside the market for the residual.
# ============================================================

HP                 = "HYDROGEL_PACK"
HP_LIMIT           = 200
HP_WINDOW          = 2000     # max rolling window
HP_MIN_WINDOW      = 700      # CRITICAL: anything below 500 → unstable mean → adverse takes
HP_Z_ENTRY         = 2.0
HP_Z_EXIT          = 0.5
HP_TAKE_EDGE       = 1        # ticks past μ to allow aggressive takes
HP_PASSIVE_OFFSET  = 1        # passive quotes posted at μ ± this (capped to inside spread)
HP_ENTRY_CHUNK     = 20       # max contracts to TAKE per tick (anti book-walk)


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

    def _save(self, d: dict) -> str:
        return json.dumps(d)

    def _mid(self, od: OrderDepth) -> Optional[float]:
        if od and od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    def _best_bid_ask(self, od: OrderDepth):
        bb = max(od.buy_orders) if od and od.buy_orders else None
        ba = min(od.sell_orders) if od and od.sell_orders else None
        return bb, ba

    def _trade_hydrogel(self, state: TradingState, data: dict) -> List[Order]:
        od = state.order_depths.get(HP, OrderDepth())
        pos = state.position.get(HP, 0)
        mid = self._mid(od)
        if mid is None:
            return []

        # Rolling mid history
        hist = data.setdefault("hp_mids", [])
        hist.append(mid)
        cap = HP_WINDOW * 2
        if len(hist) > cap:
            del hist[: len(hist) - cap]
        if len(hist) < HP_MIN_WINDOW:
            return []

        # Compute μ, σ, z over expanding window (capped at HP_WINDOW)
        eff_n = min(len(hist), HP_WINDOW)
        recent = hist[-eff_n:]
        mu = sum(recent) / eff_n
        var = sum((x - mu) ** 2 for x in recent) / (eff_n - 1)
        sigma = math.sqrt(var) if var > 1e-9 else 1.0
        z = (mid - mu) / sigma

        # Z-score state machine (hysteresis: HOLD prior target in gray zone)
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
            # BUY toward target
            need = delta
            chunk = HP_ENTRY_CHUNK
            for px in sorted(od.sell_orders.keys()):
                if need <= 0 or chunk <= 0 or px > mu + HP_TAKE_EDGE:
                    break
                avail = -od.sell_orders[px]
                qty = min(avail, need, chunk)
                if qty > 0:
                    orders.append(Order(HP, px, qty))
                    need -= qty; chunk -= qty
            if need > 0:
                bb, _ = self._best_bid_ask(od)
                if bb is not None:
                    px = min(int(math.floor(mu - HP_PASSIVE_OFFSET)), bb + 1)
                    orders.append(Order(HP, px, need))
        elif delta < 0:
            # SELL toward target
            need = -delta
            chunk = HP_ENTRY_CHUNK
            for px in sorted(od.buy_orders.keys(), reverse=True):
                if need <= 0 or chunk <= 0 or px < mu - HP_TAKE_EDGE:
                    break
                avail = od.buy_orders[px]
                qty = min(avail, need, chunk)
                if qty > 0:
                    orders.append(Order(HP, px, -qty))
                    need -= qty; chunk -= qty
            if need > 0:
                _, ba = self._best_bid_ask(od)
                if ba is not None:
                    px = max(int(math.ceil(mu + HP_PASSIVE_OFFSET)), ba - 1)
                    orders.append(Order(HP, px, -need))
        else:
            # At target: passive top-up to keep inventory at limit
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

    def run(self, state: TradingState):
        data = self._load(state.traderData)
        result: Dict[str, List[Order]] = {HP: self._trade_hydrogel(state, data)}
        return result, 0, self._save(data)
