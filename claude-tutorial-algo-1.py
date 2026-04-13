from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle

PRODUCT        = "EMERALDS"
FAIR_VALUE     = 10_000
POSITION_LIMIT = 80
HALF_SPREAD    = 5

class Trader:

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """
        EMERALDS-only pure market-making strategy.

        Logic per iteration
        ───────────────────
        1. Take any existing orders in the book that are mispriced vs fair value
           (i.e. someone selling below FV or buying above FV).  This is "lifting"
           and locks in guaranteed edge immediately.

        2. Post passive limit orders at our desired bid/ask (FV ± HALF_SPREAD)
           with the remaining position capacity, so bots can trade against us.

        3. All order sizes are capped to never breach POSITION_LIMIT in either
           direction. The exchange cancels any order that would breach the limit,
           so we manage it ourselves to avoid wasted order slots.

        Quote skewing
        ─────────────
        When we are already long, we lower both our bid and ask slightly so we
        attract more sells and fewer buys, helping us unwind.  When short, we
        raise both quotes.  The skew is proportional to how close we are to the
        position limit, capped at ±2 ticks so we never quote on the wrong side
        of fair value.
        """

        result: Dict[str, List[Order]] = {}
        orders: List[Order] = []

        # ── Guard: product must be present this iteration ──────────────────────
        if PRODUCT not in state.order_depths:
            return result, 0, jsonpickle.encode({})

        order_depth: OrderDepth = state.order_depths[PRODUCT]
        position: int = state.position.get(PRODUCT, 0)

        # Available capacity in each direction
        buy_capacity  = POSITION_LIMIT - position   # how many more we can buy
        sell_capacity = POSITION_LIMIT + position   # how many more we can sell (short side)

        # ── Step 1: Take mispriced orders (aggressive fills) ───────────────────

        # Buy any ask strictly below fair value (guaranteed positive edge)
        if order_depth.sell_orders:
            # sell_orders: {price: -volume}  (volumes stored as negative ints)
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price >= FAIR_VALUE:
                    break                            # not mispriced, stop
                if buy_capacity <= 0:
                    break
                available_vol = -order_depth.sell_orders[ask_price]  # make positive
                fill_qty = min(available_vol, buy_capacity)
                orders.append(Order(PRODUCT, ask_price, fill_qty))   # positive = buy
                buy_capacity -= fill_qty

        # Sell any bid strictly above fair value (guaranteed positive edge)
        if order_depth.buy_orders:
            # buy_orders: {price: volume}  (volumes stored as positive ints)
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price <= FAIR_VALUE:
                    break
                if sell_capacity <= 0:
                    break
                available_vol = order_depth.buy_orders[bid_price]
                fill_qty = min(available_vol, sell_capacity)
                orders.append(Order(PRODUCT, bid_price, -fill_qty))  # negative = sell
                sell_capacity -= fill_qty

        # ── Step 2: Quote skew based on current inventory ─────────────────────
        # Skew proportional to inventory, capped at ±2 ticks
        skew = -int(round((position / POSITION_LIMIT) * 2))

        our_bid = FAIR_VALUE - HALF_SPREAD + skew
        our_ask = FAIR_VALUE + HALF_SPREAD + skew

        # Safety check: never quote bid >= ask
        if our_bid >= our_ask:
            our_bid = FAIR_VALUE - 1
            our_ask = FAIR_VALUE + 1

        # ── Step 3: Post passive limit orders with remaining capacity ──────────
        if buy_capacity > 0:
            orders.append(Order(PRODUCT, our_bid, buy_capacity))

        if sell_capacity > 0:
            orders.append(Order(PRODUCT, our_ask, -sell_capacity))

        result[PRODUCT] = orders
        return result, 0, jsonpickle.encode({})
