from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math

# ─────────────────────────────────────────────────────────────
#  INTARIAN_PEPPER_ROOT — Trend-Following Market Maker
#
#  Key insight from data analysis:
#    Fair Value = 10,000 + 1,000 × (day + 2) + 0.001 × timestamp
#
#  Within each day the price rises linearly by ~1 pt per 1,000 ts
#  and jumps +1,000 between days. Residual noise std ≈ ±2.2 ticks.
#  The market's own spread is ~13 ticks (ask ≈ FV+6.5, bid ≈ FV-6.5),
#  so quoting tighter than that captures the spread profitably.
#
#  Strategy:
#    1. Compute FV from the known formula every tick.
#    2. Take any resting orders that are mispriced vs FV (free alpha).
#    3. Post passive quotes at FV ± QUOTE_OFFSET to earn the spread.
#    4. Respect position limits at all times.
# ─────────────────────────────────────────────────────────────

PRODUCT = "INTARIAN_PEPPER_ROOT"

# Position limit for Pepper Root (standard Prosperity limit)
POSITION_LIMIT = 50

# How far from FV we post our passive bid/ask quotes.
# Market spread is ~13 ticks; quoting at ±4 undercuts the market
# while staying safely outside the ±2.2 noise std.
QUOTE_OFFSET = 4

# We will cross the market (take liquidity) for any order priced
# more than this many ticks away from FV — pure mispricing profit.
TAKE_THRESHOLD = 1

# We infer the day from traderData to avoid depending on hidden state.
# On the very first tick of a new session the day is unknown; we
# bootstrap it from the order book mid-price.
DAY_UNKNOWN = -999


class Trader:

    # ── helpers ──────────────────────────────────────────────

    def _fair_value(self, day: int, timestamp: int) -> float:
        """
        Core formula derived from 3-day regression:
          FV = 10,000 + 1,000*(day+2) + 0.001*timestamp
        day is the integer day index (e.g. -2, -1, 0, 1 …).
        """
        return 10_000 + 1_000 * (day + 2) + 0.001 * timestamp

    def _infer_day(self, state: TradingState) -> int:
        """
        Infer the current day from the order book mid-price.
        FV at ts=0 is 10,000 + 1,000*(day+2), so:
          day ≈ round((mid - 10,000) / 1,000) - 2
        This is only used as a fallback on the very first tick if
        we have no stored day in traderData.
        """
        od: OrderDepth = state.order_depths.get(PRODUCT)
        if od is None:
            return 0  # safe default

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid and best_ask:
            mid = (best_bid + best_ask) / 2
        elif best_bid:
            mid = best_bid
        elif best_ask:
            mid = best_ask
        else:
            return 0

        raw_day = round((mid - 10_000) / 1_000) - 2
        return int(raw_day)

    def _load_state(self, trader_data: str) -> dict:
        if trader_data:
            try:
                return json.loads(trader_data)
            except Exception:
                pass
        return {}

    def _save_state(self, data: dict) -> str:
        return json.dumps(data)

    # ── main entry point ─────────────────────────────────────

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """
        Returns (orders, conversions, traderData).
        """
        # ── load persistent state ──────────────────────────
        saved = self._load_state(state.traderData)
        day: int = saved.get("day", DAY_UNKNOWN)

        # Infer day if we haven't stored it yet
        if day == DAY_UNKNOWN:
            day = self._infer_day(state)
            saved["day"] = day

        # ── compute fair value ─────────────────────────────
        fv = self._fair_value(day, state.timestamp)

        # ── current position ───────────────────────────────
        position: int = state.position.get(PRODUCT, 0)

        # How much more we can buy / sell before hitting limits
        buy_capacity  = POSITION_LIMIT - position   # positive
        sell_capacity = POSITION_LIMIT + position   # positive (position can be negative)

        orders: List[Order] = []
        od: OrderDepth = state.order_depths.get(PRODUCT, OrderDepth())

        # ── STEP 1: Take mispriced resting orders ──────────
        #
        # If someone is offering (selling) at a price below FV - TAKE_THRESHOLD,
        # that is cheap — we buy it.
        #
        # If someone is bidding (buying) above FV + TAKE_THRESHOLD,
        # that is overpriced — we sell to them.

        # Sort asks ascending (cheapest first) and bids descending (highest first)
        sorted_asks = sorted(od.sell_orders.items())   # list of (price, volume); volume is negative in Prosperity
        sorted_bids = sorted(od.buy_orders.items(), reverse=True)

        # Take cheap asks
        for ask_price, ask_vol in sorted_asks:
            if ask_price > fv - TAKE_THRESHOLD:
                break  # no longer cheap
            if buy_capacity <= 0:
                break

            # ask_vol is negative in Prosperity datamodel; qty to buy is positive
            available = -ask_vol
            qty = min(available, buy_capacity)
            orders.append(Order(PRODUCT, ask_price, qty))
            buy_capacity -= qty

        # Take expensive bids
        for bid_price, bid_vol in sorted_bids:
            if bid_price < fv + TAKE_THRESHOLD:
                break  # no longer overpriced
            if sell_capacity <= 0:
                break

            available = bid_vol   # positive
            qty = min(available, sell_capacity)
            orders.append(Order(PRODUCT, bid_price, -qty))
            sell_capacity -= qty

        # ── STEP 2: Post passive market-making quotes ──────
        #
        # Quote bid at fv - QUOTE_OFFSET and ask at fv + QUOTE_OFFSET.
        # These undercut the existing market spread (~13 ticks) while
        # staying well outside the ±2.2 noise band.
        #
        # Size the quotes proportional to remaining capacity so we
        # never breach position limits.

        passive_bid_price = math.floor(fv - QUOTE_OFFSET)
        passive_ask_price = math.ceil(fv + QUOTE_OFFSET)

        # Quote size: use a modest fraction of remaining capacity so
        # a single fill doesn't max us out; cap at 10 per quote.
        passive_bid_qty = min(10, buy_capacity)
        passive_ask_qty = min(10, sell_capacity)

        if passive_bid_qty > 0:
            orders.append(Order(PRODUCT, passive_bid_price, passive_bid_qty))

        if passive_ask_qty > 0:
            orders.append(Order(PRODUCT, passive_ask_price, -passive_ask_qty))

        # ── pack results ───────────────────────────────────
        result: Dict[str, List[Order]] = {PRODUCT: orders}
        conversions = 0
        new_trader_data = self._save_state(saved)

        return result, conversions, new_trader_data
