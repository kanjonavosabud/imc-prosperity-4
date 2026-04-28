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

ACO_EWMA_SPAN    = 21
ACO_TAKE_EDGE    = 1
ACO_QUOTE_OFFSET = 2
ACO_QUOTE_SIZE   = 25
ACO_L2_OFFSET    = 4
ACO_L2_SIZE      = 15
ACO_SKEW         = 0.10

# --- INTARIAN_PEPPER_ROOT (linear ramp) ----------------------
IPR              = "INTARIAN_PEPPER_ROOT"
IPR_LIMIT        = 80

IPR_SLOPE        = 0.001   # expected price increase per ts unit
IPR_CORR_ALPHA   = 0.80    # fast snap: residuals are iid (AR1=0.017), no persistence
IPR_TAKE_EDGE    = 1       # min ticks of edge to aggress
IPR_QUOTE_OFFSET = 1       # primary quote distance from fair
IPR_BID_SIZE     = 35      # bid-side quote size (accumulate faster to max position)
IPR_ASK_SIZE     = 2       # minimal ask: avoid giving back upside in strong trend
IPR_ASK_OFFSET   = 8       # very wide ask; only fill on spikes above fair
IPR_L2_OFFSET    = 4       # 2nd-layer distance
IPR_L2_BID_SIZE  = 16      # 2nd-layer bid size
IPR_L2_ASK_SIZE  = 0       # 2nd-layer ask size (off in strong trend)
IPR_SKEW_COEFF   = 0.6     # sublinear skew: coeff * sqrt(|gap|)

# --- ADAPTIVE TARGET PARAMS ----------------------------------
IPR_MAX_TARGET   = 80      # hold full position limit in strong trend
IPR_RAMP_TS      = 1500    # timestamps to ramp target from 0 → max
IPR_SLOPE_WINDOW = 30      # ticks of mid history to estimate slope
IPR_SLOPE_THRESH = 0.0003  # below this slope, target trends toward 0


# ============================================================
#  TRADER
# ============================================================

class Trader:

    # ---- MAF Auction Bid -------------------------------------

    def bid(self) -> int:
        """
        Bids for 25% extra market access volume in Round 2.
        2,500 is a Level-2 strategic bid designed to beat the median
        without severely impacting net PnL.
        """
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

    # ---- counterparty tracking -------------------------------

    def _track_counterparties(self, state: TradingState, data: dict):
        """
        Parses the trade tape to see who is taking our liquidity and
        maintains a persistent tally in the traderData state.
        """
        if "counterparty_volume" not in data:
            data["counterparty_volume"] = {}

        for product, trades in state.own_trades.items():
            for trade in trades:
                # Identify the counterparty. Usually, your bot is labeled "SUBMISSION"
                if trade.buyer == "SUBMISSION":
                    cp = trade.seller
                else:
                    cp = trade.buyer

                # Keep a running tally of absolute volume traded against each ID
                if cp and cp != "SUBMISSION":
                    current_vol = data["counterparty_volume"].get(cp, 0)
                    data["counterparty_volume"][cp] = current_vol + abs(trade.quantity)

        # Print to standard out so it appears in the sandbox logs
        print(f"[{state.timestamp}] Counterparty Volumes: {data['counterparty_volume']}")

    # ---- order-book helpers ----------------------------------

    def _mid(self, od: OrderDepth):
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    def _microprice(self, od: OrderDepth):
        if od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            bid_vol = od.buy_orders[best_bid]
            ask_vol = abs(od.sell_orders[best_ask])
            total = bid_vol + ask_vol
            if total > 0:
                return (best_bid * ask_vol + best_ask * bid_vol) / total
        return self._mid(od)

    def _book_imbalance(self, od: OrderDepth):
        total_bid = sum(od.buy_orders.values()) if od.buy_orders else 0
        total_ask = sum(abs(v) for v in od.sell_orders.values()) if od.sell_orders else 0
        total = total_bid + total_ask
        if total > 0:
            return (total_bid - total_ask) / total
        return 0.0

    def _detect_day_offset(self, mid: float) -> float:
        return round((mid - 10000) / 1000) * 1000

    # ---- generic order builder -------------------------------

    def _build_orders(
        self, symbol: str, od: OrderDepth, pos: int,
        fair: float, limit: int,
        take_edge: float, q_off: float,
        bid_sz: int, ask_sz: int,
        l2_off: float, l2_bid_sz: int, l2_ask_sz: int,
        q_off_ask: float = None,
    ):

        orders: List[Order] = []
        buy_cap  = limit - pos
        sell_cap = limit + pos
        if q_off_ask is None:
            q_off_ask = q_off

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
        ask_px = math.ceil(fair + q_off_ask)
        b_sz = min(bid_sz, buy_cap)
        a_sz = min(ask_sz, sell_cap)
        if b_sz > 0:
            orders.append(Order(symbol, bid_px, b_sz))
            buy_cap -= b_sz
        if a_sz > 0:
            orders.append(Order(symbol, ask_px, -a_sz))
            sell_cap -= a_sz

        # --- QUOTE: passive layer 2 ----------------------------
        if l2_off > 0:
            bid2_px = math.floor(fair - l2_off)
            ask2_px = math.ceil(fair + l2_off)
            b2_sz = min(l2_bid_sz, buy_cap)
            a2_sz = min(l2_ask_sz, sell_cap)
            if b2_sz > 0:
                orders.append(Order(symbol, bid2_px, b2_sz))
                buy_cap -= b2_sz
            if a2_sz > 0:
                orders.append(Order(symbol, ask2_px, -a2_sz))
                sell_cap -= a2_sz

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

        skew_penalty = ACO_SKEW * pos * (1 + abs(pos) / ACO_LIMIT)
        fair = ewma - skew_penalty

        orders: List[Order] = []
        buy_cap  = ACO_LIMIT - pos
        sell_cap = ACO_LIMIT + pos

        # --- TAKE: sweep mispriced resting orders ---------------
        for px in sorted(od.sell_orders.keys()):
            if px <= fair - ACO_TAKE_EDGE and buy_cap > 0:
                vol = -od.sell_orders[px]
                qty = min(vol, buy_cap)
                if qty > 0:
                    orders.append(Order(ACO, px, qty))
                    buy_cap -= qty

        for px in sorted(od.buy_orders.keys(), reverse=True):
            if px >= fair + ACO_TAKE_EDGE and sell_cap > 0:
                vol = od.buy_orders[px]
                qty = min(vol, sell_cap)
                if qty > 0:
                    orders.append(Order(ACO, px, -qty))
                    sell_cap -= qty

        # --- QUOTE: layer 1 (dynamic: inside market) -----------
        if od.buy_orders:
            market_bid = max(od.buy_orders.keys())
            bid_px = min(market_bid + 1, math.floor(fair - ACO_TAKE_EDGE))
        else:
            bid_px = math.floor(fair - ACO_QUOTE_OFFSET)

        if od.sell_orders:
            market_ask = min(od.sell_orders.keys())
            ask_px = max(market_ask - 1, math.ceil(fair + ACO_TAKE_EDGE))
        else:
            ask_px = math.ceil(fair + ACO_QUOTE_OFFSET)

        bid_sz = min(ACO_QUOTE_SIZE, buy_cap)
        ask_sz = min(ACO_QUOTE_SIZE, sell_cap)

        if bid_sz > 0 and bid_px < ask_px:
            orders.append(Order(ACO, bid_px, bid_sz))
            buy_cap -= bid_sz
        if ask_sz > 0 and ask_px > bid_px:
            orders.append(Order(ACO, ask_px, -ask_sz))
            sell_cap -= ask_sz

        # --- QUOTE: layer 2 ------------------------------------
        if ACO_L2_OFFSET > 0 and ACO_L2_SIZE > 0:
            l2_dist = max(1, math.floor(ACO_L2_OFFSET - ACO_QUOTE_OFFSET))
            bid2_px = bid_px - l2_dist
            ask2_px = ask_px + l2_dist

            bid2_sz = min(ACO_L2_SIZE, buy_cap)
            ask2_sz = min(ACO_L2_SIZE, sell_cap)
            if bid2_sz > 0:
                orders.append(Order(ACO, bid2_px, bid2_sz))
            if ask2_sz > 0:
                orders.append(Order(ACO, ask2_px, -ask2_sz))

        return orders

    # ---- IPR slope estimation --------------------------------

    def _estimate_slope(self, data: dict) -> float:
        history = data.get("ipr_mids", [])
        if len(history) < 10:
            return IPR_SLOPE

        recent = history[-IPR_SLOPE_WINDOW:]
        n = len(recent)
        if n < 10:
            return IPR_SLOPE

        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den == 0:
            return IPR_SLOPE

        slope_per_tick = num / den
        slope_per_ts = slope_per_tick / 100.0
        return slope_per_ts

    def _adaptive_target(self, data: dict) -> float:
        slope = self._estimate_slope(data)
        if slope >= IPR_SLOPE_THRESH:
            return float(IPR_MAX_TARGET)
        elif slope >= 0:
            return IPR_MAX_TARGET * (slope / IPR_SLOPE_THRESH)
        else:
            return 0.0

    # ---- IPR strategy ----------------------------------------

    def _trade_ipr(self, state: TradingState, data: dict) -> List[Order]:
        od  = state.order_depths.get(IPR, OrderDepth())
        pos = state.position.get(IPR, 0)
        ts  = state.timestamp
        mid = self._microprice(od)

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

        inv_target = self._adaptive_target(data)
        if "ipr_start_ts" not in data:
            data["ipr_start_ts"] = ts
        ramp = min(1.0, (ts - data["ipr_start_ts"]) / IPR_RAMP_TS) if IPR_RAMP_TS > 0 else 1.0
        inv_target *= ramp

        base_fair = 10000 + data["ipr_doff"] + IPR_SLOPE * ts

        corr = data.get("ipr_corr", 0.0)
        if mid is not None:
            residual = mid - base_fair
            corr = IPR_CORR_ALPHA * residual + (1 - IPR_CORR_ALPHA) * corr
            data["ipr_corr"] = corr

        # --- trend protection: adjust selling based on live slope ---
        slope = self._estimate_slope(data)

        if slope < 0:
            inv_target = 0.0
            eff_ask_sz = IPR_BID_SIZE
            eff_l2_ask_sz = IPR_L2_BID_SIZE
            eff_ask_off = IPR_QUOTE_OFFSET
        elif slope < IPR_SLOPE_THRESH:
            ratio = max(0.0, slope / IPR_SLOPE_THRESH)
            inv_target *= ratio
            eff_ask_sz = 4
            eff_l2_ask_sz = 0
            eff_ask_off = 5
        else:
            eff_ask_sz = IPR_ASK_SIZE
            eff_l2_ask_sz = IPR_L2_ASK_SIZE
            eff_ask_off = IPR_ASK_OFFSET

        data["ipr_target"] = inv_target

        fair_raw = base_fair + corr

        adj_pos = pos - inv_target
        sign = -1 if adj_pos < 0 else 1
        skew_adj = IPR_SKEW_COEFF * sign * math.sqrt(abs(adj_pos))
        fair    = fair_raw - skew_adj

        orders, _, _ = self._build_orders(
            IPR, od, pos, fair, IPR_LIMIT,
            IPR_TAKE_EDGE, IPR_QUOTE_OFFSET,
            IPR_BID_SIZE, eff_ask_sz,
            IPR_L2_OFFSET, IPR_L2_BID_SIZE, eff_l2_ask_sz,
            q_off_ask=eff_ask_off,
        )

        return orders

    # ---- main entry point ------------------------------------

    def run(self, state: TradingState):
        data   = self._load(state.traderData)
        result = {}

        self._track_counterparties(state, data)

        result[ACO] = self._trade_aco(state, data)
        result[IPR] = self._trade_ipr(state, data)

        conversions = 0
        return result, conversions, self._save(data)