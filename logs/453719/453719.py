from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math

# ============================================================
#  CONFIGURABLE PARAMETERS
# ============================================================

# --- HYDROGEL_PACK (z-score mean reversion, conservative) ----
# Backtest grid: (window=2000, z_entry=2.0, z_exit=0.5) — best risk-adj PnL.
# Aggressive (500, 1.5, 0.0) had higher raw PnL but proved fragile in live test.
HP                 = "HYDROGEL_PACK"
HP_LIMIT           = 60
HP_WINDOW          = 2000
HP_Z_ENTRY         = 2.0
HP_Z_EXIT          = 0.5
HP_TAKE_EDGE       = 1
HP_PASSIVE_OFFSET  = 1
HP_ENTRY_CHUNK     = 20       # cap per-tick aggression to avoid walking the book


# ============================================================
#  TRADER
# ============================================================

class Trader:

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

    # ---- main entry point ------------------------------------

    def run(self, state: TradingState):
        data = self._load(state.traderData)
        result: Dict[str, List[Order]] = {HP: self._trade_hydrogel(state, data)}
        return result, 0, self._save(data)