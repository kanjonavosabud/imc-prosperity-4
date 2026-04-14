from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

PRODUCT        = "EMERALDS" # only trading emeralds
FAIR_VALUE     = 10_000
POSITION_LIMIT = 80
HALF_SPREAD    = 7
INV_SKEW       = 0


class Trader:

    def bid(self):
        return 15
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}

        for product in state.order_depths:
            
            if product != PRODUCT:
                continue

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            acceptable_price = FAIR_VALUE  # Participant should calculate this value
            # print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            position: int = state.position.get(PRODUCT, 0)

            # Available capacity in each direction
            buy_capacity  = POSITION_LIMIT - position   # how many more we can buy
            sell_capacity = POSITION_LIMIT + position   # how many more we can sell (short side)

            # just checking for mispriced orders
            if len(order_depth.sell_orders) != 0:
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if ask_price >= acceptable_price: break                          
                    if buy_capacity <= 0: break
                    available_vol = -order_depth.sell_orders[ask_price]
                    fill_qty = min(available_vol, buy_capacity)
                    orders.append(Order(PRODUCT, ask_price, fill_qty))
                    buy_capacity -= fill_qty
    
            if len(order_depth.buy_orders) != 0:
                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bid_price <= acceptable_price: break
                    if sell_capacity <= 0: break
                    available_vol = order_depth.buy_orders[bid_price]
                    fill_qty = min(available_vol, sell_capacity)
                    orders.append(Order(PRODUCT, bid_price, -fill_qty))
                    sell_capacity -= fill_qty

            # Quote skew based on current inventory
            skew = -int(round((position / POSITION_LIMIT) * INV_SKEW))
            our_bid = acceptable_price - HALF_SPREAD + skew
            our_ask = acceptable_price + HALF_SPREAD + skew

            # Safety check: never quote bid >= ask
            if our_bid >= our_ask:
                our_bid = acceptable_price - 1
                our_ask = acceptable_price + 1
            
            # Post passive limit orders with remaining capacity
            if buy_capacity > 0:
                orders.append(Order(PRODUCT, our_bid, buy_capacity))
            if sell_capacity > 0:
                orders.append(Order(PRODUCT, our_ask, -sell_capacity))
            
            result[product] = orders
    
        # String value holding Trader state data required. 
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
        # Sample conversion request. Check more details below. 
        conversions = 0
        return result, conversions, traderData
