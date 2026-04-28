from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math

# ============================================================
#  CONFIGURABLE PARAMETERS
# ============================================================

# --- ASH_COATED_OSMIUM (stable, mean-reverting ~10000) -------
ACO              = "ASH_COATED_OSMIUM"
ACO_LIMIT        = 80

ACO_EWMA_SPAN    = 21      
ACO_TAKE_EDGE    = 1       # increased to 1 to guarantee edge on takes
ACO_QUOTE_OFFSET = 2       # default offset if book is empty
ACO_QUOTE_SIZE   = 25      
ACO_L2_OFFSET    = 4       
ACO_L2_SIZE      = 15      
ACO_WALL_SIZE    = 0       
ACO_SKEW         = 0.10    

# --- INTARIAN_PEPPER_ROOT (linear ramp) ----------------------
IPR              = "INTARIAN_PEPPER_ROOT"
IPR_LIMIT        = 80

IPR_SLOPE        = 0.001   
IPR_CORR_ALPHA   = 0.20    
IPR_TAKE_EDGE    = 1       
IPR_QUOTE_OFFSET = 2       
IPR_QUOTE_SIZE   = 28      
IPR_L2_OFFSET    = 4       
IPR_L2_SIZE      = 16      
IPR_SKEW         = 0.25    

# --- ADAPTIVE TARGET PARAMS ----------------------------------
IPR_MAX_TARGET   = 65      
IPR_SLOPE_WINDOW = 30      
IPR_SLOPE_THRESH = 0.0003  # Below this empirical slope, assume trend is dead


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
            return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
        return None

    def _detect_day_offset(self, mid: float) -> float:
        return round((mid - 10000) / 1000) * 1000

    # ---- generic order builder (UPGRADED WITH DYNAMIC SPREADS)

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
            if px <= fair - take_edge and buy_cap > 0:
                vol = -od.sell_orders[px]
                qty = min(vol, buy_cap)
                if qty > 0:
                    orders.append(Order(symbol, px, qty))
                    buy_cap -= qty

        for px in sorted(od.buy_orders.keys(), reverse=True):
            if px >= fair + take_edge and sell_cap > 0:
                vol = od.buy_orders[px]
                qty = min(vol, sell_cap)
                if qty > 0:
                    orders.append(Order(symbol, px, -qty))
                    sell_cap -= qty

        # --- QUOTE: passive layer 1 (DYNAMIC WIDENING) ---------
        # Quote just inside the market best bid/ask, constrained by our theoretical fair
        if od.buy_orders:
            market_bid = max(od.buy_orders.keys())
            bid_px = min(market_bid + 1, math.floor(fair - take_edge))
        else:
            bid_px = math.floor(fair - q_off)

        if od.sell_orders:
            market_ask = min(od.sell_orders.keys())
            ask_px = max(market_ask - 1, math.ceil(fair + take_edge))
        else:
            ask_px = math.ceil(fair + q_off)

        bid_sz = min(q_sz, buy_cap)
        ask_sz = min(q_sz, sell_cap)
        
        if bid_sz > 0 and bid_px < ask_px:
            orders.append(Order(symbol, bid_px, bid_sz))
            buy_cap -= bid_sz
        if ask_sz > 0 and ask_px > bid_px:
            orders.append(Order(symbol, ask_px, -ask_sz))
            sell_cap -= ask_sz

        # --- QUOTE: passive layer 2 ----------------------------
        if l2_off > 0 and l2_sz > 0:
            l2_dist = max(1, math.floor(l2_off - q_off))
            bid2_px = bid_px - l2_dist
            ask2_px = ask_px + l2_dist
            
            bid2_sz = min(l2_sz, buy_cap)
            ask2_sz = min(l2_sz, sell_cap)
            if bid2_sz > 0:
                orders.append(Order(symbol, bid2_px, bid2_sz))
                buy_cap -= bid2_sz
            if ask2_sz > 0:
                orders.append(Order(symbol, ask2_px, -ask2_sz))
                sell_cap -= ask2_sz

        return orders, buy_cap, sell_cap

    # ---- ACO strategy (UPGRADED WITH NON-LINEAR SKEW) --------

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

        # Quadratic penalty as inventory nears limit
        skew_penalty = ACO_SKEW * pos * (1 + abs(pos) / ACO_LIMIT)
        fair = ewma - skew_penalty

        orders, _, _ = self._build_orders(
            ACO, od, pos, fair, ACO_LIMIT,
            ACO_TAKE_EDGE, ACO_QUOTE_OFFSET, ACO_QUOTE_SIZE,
            ACO_L2_OFFSET, ACO_L2_SIZE,
        )
        return orders

    # ---- IPR strategy (classic round 1) ----------------------

    def _trade_ipr(self, state: TradingState, data: dict) -> List[Order]:
        od  = state.order_depths.get(IPR, OrderDepth())
        pos = state.position.get(IPR, 0)
        ts  = state.timestamp
        mid = self._mid(od)

        # --- initialize per-day offset and history --------------
        if "ipr_doff" not in data:
            if mid is not None:
                data["ipr_doff"] = self._detect_day_offset(mid)
            else:
                data["ipr_doff"] = 3000

        if "ipr_mids" not in data:
            data["ipr_mids"] = []
        if mid is not None:
            data["ipr_mids"].append(mid)
            max_hist = IPR_SLOPE_WINDOW * 2
            if len(data["ipr_mids"]) > max_hist:
                data["ipr_mids"] = data["ipr_mids"][-max_hist:]

        # --- estimate slope and adapt target -------------------
        inv_target = self._adaptive_target(data)
        data["ipr_target"] = inv_target

        # --- analytical fair value + correction ----------------
        base_fair = 10000 + data["ipr_doff"] + IPR_SLOPE * ts

        corr = data.get("ipr_corr", 0.0)
        if mid is not None:
            residual = mid - base_fair
            corr = IPR_CORR_ALPHA * residual + (1 - IPR_CORR_ALPHA) * corr
            data["ipr_corr"] = corr

        fair_raw = base_fair + corr

        # --- inventory-target skew -----------------------------
        adj_pos = pos - inv_target
        fair    = fair_raw - IPR_SKEW * adj_pos

        orders, _, _ = self._build_orders(
            IPR, od, pos, fair, IPR_LIMIT,
            IPR_TAKE_EDGE, IPR_QUOTE_OFFSET, IPR_QUOTE_SIZE,
            IPR_L2_OFFSET, IPR_L2_SIZE,
        )

        return orders

    # ---- adaptive target (used for IPR only) -----------------

    def _adaptive_target(self, data: dict) -> float:
        corr = data.get("ipr_corr", 0.0)
        if corr > -10:
            return float(IPR_MAX_TARGET)
        elif corr < -20:
            return 0.0
        else:
            return IPR_MAX_TARGET * (20 + corr) / 10.0

    # ---- Generic Strategy (FALLBACK FOR NEW ASSETS) ------------

    def _trade_generic(self, symbol: str, state: TradingState, data: dict) -> List[Order]:
        """Provides a safe, mean-reverting fallback for new, unknown symbols."""
        od  = state.order_depths.get(symbol, OrderDepth())
        pos = state.position.get(symbol, 0)
        mid = self._mid(od)
        
        limit = 50 # Conservative default limit
        
        key_ewma = f"{symbol}_ewma"
        prev = data.get(key_ewma)
        
        if mid is not None:
            ewma = mid if prev is None else 0.1 * mid + 0.9 * prev
            data[key_ewma] = ewma
        else:
            ewma = prev
            if ewma is None: return []
            
        fair = ewma - (0.1 * pos * (1 + abs(pos) / limit))

        orders, _, _ = self._build_orders(
            symbol, od, pos, fair, limit,
            take_edge=1, q_off=2, q_sz=15,
            l2_off=4, l2_sz=10,
        )
        return orders

    # ---- main entry point ------------------------------------

    def run(self, state: TradingState):
        data   = self._load(state.traderData)
        result = {}

        # Dynamically loop through all assets to catch new Round 2 additions
        for symbol in state.order_depths.keys():
            if symbol == ACO:
                result[ACO] = self._trade_aco(state, data)
            elif symbol == IPR:
                result[IPR] = self._trade_ipr(state, data)
            else:
                result[symbol] = self._trade_generic(symbol, state, data)

        conversions = 0
        return result, conversions, self._save(data)
