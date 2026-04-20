from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math


# ============================================================
#  CONFIGURABLE PARAMETERS — tweak these between runs
# ============================================================

# --- ASH_COATED_OSMIUM (stable, mean-reverting ~10000) -------
ACO              = "ASH_COATED_OSMIUM"
ACO_LIMIT        = 80

ACO_EWMA_SPAN    = 21      # higher = smoother fair estimate
ACO_TAKE_EDGE    = 0       # min ticks of edge to aggress
ACO_QUOTE_OFFSET = 1       # primary quote distance from fair
ACO_QUOTE_SIZE   = 25      # primary quote size per side
ACO_L2_OFFSET    = 3       # 2nd-layer quote distance
ACO_L2_SIZE      = 15      # 2nd-layer quote size per side
ACO_WALL_SIZE    = 0       # 3rd-layer: quote 1 inside detected wall
ACO_SKEW         = 0.10    # fair shifts by  -SKEW * position

# --- INTARIAN_PEPPER_ROOT (parameters for new algo) ----------
IPR              = "INTARIAN_PEPPER_ROOT"
IPR_LIMIT        = 80

IPR_TREND_SPAN   = 16      # ewma span for trend signal
IPR_QUOTE_OFFSET = 2       # quote distance from fair
IPR_QUOTE_SIZE   = 28      # quote size per side
IPR_L2_OFFSET    = 6       # 2nd-layer quote distance
IPR_L2_SIZE      = 18      # 2nd-layer quote size per side
IPR_SKEW         = 0.20    # inventory flattening coefficient

IPR_IDLE_EDGE    = 1       # minimum edge to sweep
IPR_IDLE_FLAT    = 0.0     # target position when untrended


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

    def _mid(self, od: OrderDepth):
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    def _detect_day_offset(self, mid: float) -> float:
        return round((mid - 10000) / 1000) * 1000

    # ---- generic order builder -------------------------------

    def _build_orders(
        self, symbol: str, od: OrderDepth, pos: int,
        fair: float, limit: int,
        take_edge: float, q_off: float, q_sz: int,
        l2_off: float, l2_sz: int,
    ):

        orders: List[Order] = []
        buy_cap  = limit - pos
        sell_cap = limit + pos

        # --- TAKE: sweep mispriced resting orders ---------------
        for px in sorted(od.sell_orders.keys()):
            if px > fair - take_edge or buy_cap <= 0:
                break
            vol = -od.sell_orders[px]
            qty = min(vol, buy_cap)
            if qty > 0:
                orders.append(Order(symbol, px, qty))
                buy_cap -= qty

        for px in sorted(od.buy_orders.keys(), reverse=True):
            if px < fair + take_edge or sell_cap <= 0:
                break
            vol = od.buy_orders[px]
            qty = min(vol, sell_cap)
            if qty > 0:
                orders.append(Order(symbol, px, -qty))
                sell_cap -= qty

        # --- QUOTE: passive layer 1 ----------------------------
        bid_px = math.floor(fair - q_off)
        ask_px = math.ceil(fair + q_off)
        bid_sz = min(q_sz, buy_cap)
        ask_sz = min(q_sz, sell_cap)
        if bid_sz > 0:
            orders.append(Order(symbol, bid_px, bid_sz))
            buy_cap -= bid_sz
        if ask_sz > 0:
            orders.append(Order(symbol, ask_px, -ask_sz))
            sell_cap -= ask_sz

        # --- QUOTE: passive layer 2 ----------------------------
        if l2_off > 0 and l2_sz > 0:
            bid2_px = math.floor(fair - l2_off)
            ask2_px = math.ceil(fair + l2_off)
            bid2_sz = min(l2_sz, buy_cap)
            ask2_sz = min(l2_sz, sell_cap)
            if bid2_sz > 0:
                orders.append(Order(symbol, bid2_px, bid2_sz))
                buy_cap -= bid2_sz
            if ask2_sz > 0:
                orders.append(Order(symbol, ask2_px, -ask2_sz))
                sell_cap -= ask2_sz

        return orders, buy_cap, sell_cap

    # ---- ACO strategy ----------------------------------------

    def _trade_aco(self, state: TradingState, data: dict) -> List[Order]:
        od  = state.order_depths.get(ACO, OrderDepth())
        pos = state.position.get(ACO, 0)
        mid = self._mid(od)

        alpha = 2.0 / (ACO_EWMA_SPAN + 1)
        prev  = data.get("aco_ewma")

        if mid is not None:
            ewma = mid if prev is None else alpha * mid + (1 - alpha) * prev
            data["aco_ewma"] = ewma
        else:
            ewma = prev
            if ewma is None:
                return []

        fair = ewma - ACO_SKEW * pos

        orders, buy_cap, sell_cap = self._build_orders(
            ACO, od, pos, fair, ACO_LIMIT,
            ACO_TAKE_EDGE, ACO_QUOTE_OFFSET, ACO_QUOTE_SIZE,
            ACO_L2_OFFSET, ACO_L2_SIZE,
        )

        # --- LAYER 3: wall-aware quotes -----------------------
        if ACO_WALL_SIZE > 0 and od.buy_orders and od.sell_orders:
            # Wall = largest-volume level on each side
            wall_bid = max(od.buy_orders.keys(), key=lambda p: od.buy_orders[p])
            wall_ask = min(od.sell_orders.keys(), key=lambda p: abs(od.sell_orders[p]))

            wall_bid_px = wall_bid + 1
            wall_ask_px = wall_ask - 1

            # Only place if outside our L2 quotes (no collision)
            if wall_bid_px < math.floor(fair - ACO_L2_OFFSET):
                sz = min(ACO_WALL_SIZE, buy_cap)
                if sz > 0:
                    orders.append(Order(ACO, wall_bid_px, sz))
                    buy_cap -= sz

            if wall_ask_px > math.ceil(fair + ACO_L2_OFFSET):
                sz = min(ACO_WALL_SIZE, sell_cap)
                if sz > 0:
                    orders.append(Order(ACO, wall_ask_px, -sz))
                    sell_cap -= sz

        return orders

    # ---- IPR new strategy (momentum/ewma-trend-following) ----

    def _trade_ipr(self, state: TradingState, data: dict) -> List[Order]:
        od  = state.order_depths.get(IPR, OrderDepth())
        pos = state.position.get(IPR, 0)
        ts  = state.timestamp
        mid = self._mid(od)

        # --- compute ewma of mid to extract trend --------------
        if "ipr_ewma" not in data:  # slow signal
            data["ipr_ewma"] = mid if mid is not None else 10000.0
        if mid is not None:
            alpha = 2.0 / (IPR_TREND_SPAN + 1)
            prev = data["ipr_ewma"]
            data["ipr_ewma"] = prev if mid is None else alpha * mid + (1 - alpha) * prev

        ewma = data["ipr_ewma"]

        # --- compute a linear fair using latest mid/ewma/trend ---
        trend = 0.0 if mid is None else mid - ewma
        base_fair = ewma + trend * 0.85  # amplify trend
        base_fair = max(base_fair, 9000)  # no silly prices

        # --- simple trend-following position targeting ----------
        # If trend is positive, want to bias long, else short, else flat
        if trend > 1:
            target_pos = IPR_LIMIT
        elif trend < -1:
            target_pos = -IPR_LIMIT
        else:
            target_pos = IPR_IDLE_FLAT

        # Risk control: don't overreach inventory limits
        target_pos = max(-IPR_LIMIT, min(IPR_LIMIT, target_pos))

        # --- Skew fair for inventory flattening -----------------
        adj_pos = pos - target_pos
        fair = base_fair - IPR_SKEW * adj_pos

        # --- Build orders, more aggressive sweeping in strong trend
        edge = IPR_IDLE_EDGE
        orders, _, _ = self._build_orders(
            IPR, od, pos, fair, IPR_LIMIT,
            edge, IPR_QUOTE_OFFSET, IPR_QUOTE_SIZE,
            IPR_L2_OFFSET, IPR_L2_SIZE
        )
        return orders

    # ---- main entry point ------------------------------------

    def run(self, state: TradingState):
        data   = self._load(state.traderData)
        result = {}

        result[ACO] = self._trade_aco(state, data)
        result[IPR] = self._trade_ipr(state, data)

        conversions = 0
        return result, conversions, self._save(data)