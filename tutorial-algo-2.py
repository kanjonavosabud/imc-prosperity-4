from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Product, Dict, List
from typing import List
import string
import json


POSITION_LIMIT = {"EMERALDS": 80, "TOMATOES": 80}

PRODUCT        = "EMERALDS" # only trading emeralds
FAIR_VALUE_EMERALDS = 10_000
HALF_SPREAD    = 5

TOMATOES_WINDOW = 100
TOMATOES_MAKE_EDGE = 5
TOMATOES_TAKE_EDGE = 2
TOMATOES_Z_SKEW = 0.5  # shift quote centre by -Z_SKEW * zscore (fade deviations softly)

def best_bid_ask(od: OrderDepth):
    best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
    best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
    return best_bid, best_ask

class Trader:

    def bid(self):
        return 15
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        orders: Dict[Symbol, List[Order]] = {}
        memory = {}
        if state.traderData:
            try:
                memory = json.loads(state.traderData)
            except Exception:
                memory = {}
        mid_history: List[float] = memory.get("tomato_mids", [])

        for symbol, od in state.order_depths.items():
            pos = state.position.get(symbol, 0)
            limit = POSITION_LIMIT.get(symbol, 80)

            if symbol == "EMERALDS":
                orders[symbol] = self._trade_emeralds(od, pos, limit)
            elif symbol == "TOMATOES":
                orders[symbol], mid_history = self._trade_tomatoes(
                    od, pos, limit, mid_history
                )

        memory["tomato_mids"] = mid_history[-TOMATOES_WINDOW:]
        return orders, 0, json.dumps(memory)


    def _trade_emeralds(self, od: OrderDepth, pos: int, limit: int):
        orders: List[Order] = []

        acceptable_price = FAIR_VALUE_EMERALDS

        # Available capacity in each direction
        buy_capacity  = limit - pos   # how many more we can buy
        sell_capacity = limit + pos   # how many more we can sell (short side)

        # just checking for mispriced orders
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price >= acceptable_price: break                          
            if buy_capacity <= 0: break
            available_vol = -od.sell_orders[ask_price]
            fill_qty = min(available_vol, buy_capacity)
            orders.append(Order("EMERALDS", ask_price, fill_qty))
            buy_capacity -= fill_qty

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price <= acceptable_price: break
            if sell_capacity <= 0: break
            available_vol = od.buy_orders[bid_price]
            fill_qty = min(available_vol, sell_capacity)
            orders.append(Order("EMERALDS", bid_price, -fill_qty))
            sell_capacity -= fill_qty

        # Quote skew based on current inventory
        skew = -int(round((pos / limit) * 2))
        our_bid = acceptable_price - HALF_SPREAD + skew
        our_ask = acceptable_price + HALF_SPREAD + skew

        # Safety check: never quote bid >= ask
        if our_bid >= our_ask:
            our_bid = acceptable_price - 1
            our_ask = acceptable_price + 1
        
        # Post passive limit orders with remaining capacity
        if buy_capacity > 0:
            orders.append(Order("EMERALDS", our_bid, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order("EMERALDS", our_ask, -sell_capacity))
        
        return orders

    
    def _trade_tomatoes(self, od: OrderDepth, pos: int, limit: int, history: List[float]):
        orders: List[Order] = []
        best_bid, best_ask = best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return orders, history

        mid = (best_bid + best_ask) / 2
        history = history + [mid]
        history = history[-TOMATOES_WINDOW:]

        # Rolling fair value and z-score of current mid vs that mean
        n = len(history)
        fair = sum(history) / n
        if n >= 5:
            var = sum((x - fair) ** 2 for x in history) / n
            sd = var ** 0.5
        else:
            sd = 0.0
        zscore = (mid - fair) / sd if sd > 0 else 0.0

        buy_cap = limit - pos
        sell_cap = limit + pos

        # Aggressive take only when book clearly mis-quotes vs rolling fair
        if best_ask <= fair - TOMATOES_TAKE_EDGE and buy_cap > 0:
            ask_vol = -od.sell_orders[best_ask]
            qty = min(ask_vol, buy_cap)
            orders.append(Order("TOMATOES", best_ask, qty))
            buy_cap -= qty

        if best_bid >= fair + TOMATOES_TAKE_EDGE and sell_cap > 0:
            bid_vol = od.buy_orders[best_bid]
            qty = min(bid_vol, sell_cap)
            orders.append(Order("TOMATOES", best_bid, -qty))
            sell_cap -= qty

        # Inventory skew + soft z-score fade: centre shifts against position
        # and slightly against current deviation from rolling fair.
        inv_skew = -pos / limit  # in [-1, 1]
        z_skew = -TOMATOES_Z_SKEW * zscore
        centre = fair + inv_skew + z_skew
        make_bid = int(round(centre - TOMATOES_MAKE_EDGE))
        make_ask = int(round(centre + TOMATOES_MAKE_EDGE))

        # Stay inside existing book
        if make_bid >= best_ask:
            make_bid = best_ask - 1
        if make_ask <= best_bid:
            make_ask = best_bid + 1

        if buy_cap > 0:
            orders.append(Order("TOMATOES", make_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order("TOMATOES", make_ask, -sell_cap))

        return orders, history

