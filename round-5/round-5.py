"""Round-5 strategy.

Built on findings from ROUND_5/eda_round5.ipynb. The 50 products break into 10
named families of 5, but only TWO families have actual structure:

  - PEBBLES_XS + S + M + L + XL ≡ 50,000   (std = 1.25, hard constraint)
  - SNACKPACK_CHOCOLATE + VANILLA  ≈ 19,940 (std = 76, half-life ~1100 ticks)

Edges, in priority order:

  A. PEBBLES BASKET ARBITRAGE (deterministic).
     When Σ best_asks(5 PEBBLES) < 50000, buy 1 lot of each leg simultaneously.
     The basket is pinned at 50000 so the trade is risk-free as long as all 5
     legs fill in the same tick. Dislocations last 1-2 ticks → react fast.

  B. SNACKPACK CHOC/VANILLA z-score mean reversion.
     pair = mid(CHOC) + mid(VANILLA). Trade the z-score; enter at |z|≥2, exit
     near 0. Slow half-life → position trade, not scalp.

  C. Cross-cluster cointegration on UV_VISOR_AMBER hub.
     PEBBLES_XS ≈ 1.39·UV_VISOR_AMBER − 3619    (β positive)
     SLEEP_POD_POLYESTER ≈ −0.92·UV_VISOR_AMBER + 19140  (β negative)
     Trade the residuals as stat-arb.

  D. MICROCHIP buy-pressure quoting.
     XIRECS bot is a +7% net buyer of every MICROCHIP at +0.20 above mid.
     Quote tight on the offer, wider on the bid.

  E. Default conservative MM for the other 33 products.
     Slight sell-side skew (~0.10) to compensate for the universe-wide bot
     sell tilt (-2.5% bias, -0.10 below mid).

All 5 layers run in parallel inside one Trader class.
"""
import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple


# ============================================================
#  CONSTANTS
# ============================================================

# Universal position limit. Adjust if Round 5 docs say otherwise per product.
DEFAULT_LIMIT = 50

# === Recipe A: PEBBLES basket arb =============================
PEBBLES_LEGS = ['PEBBLES_XS', 'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L', 'PEBBLES_XL']
PEBBLES_TARGET_SUM = 50000
PEBBLES_DEV_BAND = 5             # |sum-50000| < 5 = MID regime, unwind here
PEBBLES_BASKET_LIMIT = 30        # max number of "baskets" (lots per leg) on book

# === Recipe B: SNACKPACK CHOC/VANILLA pair ====================
SP_PAIR_A = 'SNACKPACK_CHOCOLATE'
SP_PAIR_B = 'SNACKPACK_VANILLA'
SP_PAIR_TARGET = 19940.67        # empirical mean across 3 days
SP_PAIR_STD = 76.20              # empirical std
SP_PAIR_Z_ENTRY = 2.0
SP_PAIR_Z_EXIT = 0.5
SP_PAIR_LIMIT = 30               # max lots per leg in the pair trade

# === Recipe C: Cross-cluster cointegration ====================
# Each row: (target, hub, beta, intercept, residual_std, max_lots)
COINT_PAIRS = [
    ('PEBBLES_XS',          'UV_VISOR_AMBER',  1.39, -3619.46, 414.0, 15),
    ('SLEEP_POD_POLYESTER', 'UV_VISOR_AMBER', -0.92, 19140.15, 331.0, 15),
]
COINT_Z_ENTRY = 2.0
COINT_Z_EXIT = 0.5

# === Recipe D: MICROCHIP buy-bias quoting =====================
MICROCHIP_PRODUCTS = [
    'MICROCHIP_CIRCLE', 'MICROCHIP_OVAL', 'MICROCHIP_RECTANGLE',
    'MICROCHIP_SQUARE', 'MICROCHIP_TRIANGLE',
]
MICROCHIP_BID_OFFSET = 2         # quote wider bid (less aggressive buying)
MICROCHIP_ASK_OFFSET = 1         # tight ask, lift the bot's buy pressure
MICROCHIP_QUOTE_SIZE = 20

# === Recipe E: Default MM for everything else =================
DEFAULT_BID_OFFSET = 1
DEFAULT_ASK_OFFSET = 1
DEFAULT_QUOTE_SIZE = 15
DEFAULT_FV_SKEW = -0.10          # slight sell tilt for non-MICROCHIP, non-structured

# All 50 products (used for default-MM iteration). Anything claimed by another
# recipe is excluded.
ALL_PRODUCTS = [
    'GALAXY_SOUNDS_BLACK_HOLES', 'GALAXY_SOUNDS_DARK_MATTER',
    'GALAXY_SOUNDS_PLANETARY_RINGS', 'GALAXY_SOUNDS_SOLAR_FLAMES',
    'GALAXY_SOUNDS_SOLAR_WINDS',
    'MICROCHIP_CIRCLE', 'MICROCHIP_OVAL', 'MICROCHIP_RECTANGLE',
    'MICROCHIP_SQUARE', 'MICROCHIP_TRIANGLE',
    'OXYGEN_SHAKE_CHOCOLATE', 'OXYGEN_SHAKE_EVENING_BREATH',
    'OXYGEN_SHAKE_GARLIC', 'OXYGEN_SHAKE_MINT', 'OXYGEN_SHAKE_MORNING_BREATH',
    'PANEL_1X2', 'PANEL_1X4', 'PANEL_2X2', 'PANEL_2X4', 'PANEL_4X4',
    'PEBBLES_L', 'PEBBLES_M', 'PEBBLES_S', 'PEBBLES_XL', 'PEBBLES_XS',
    'ROBOT_DISHES', 'ROBOT_IRONING', 'ROBOT_LAUNDRY', 'ROBOT_MOPPING',
    'ROBOT_VACUUMING',
    'SLEEP_POD_COTTON', 'SLEEP_POD_LAMB_WOOL', 'SLEEP_POD_NYLON',
    'SLEEP_POD_POLYESTER', 'SLEEP_POD_SUEDE',
    'SNACKPACK_CHOCOLATE', 'SNACKPACK_PISTACHIO', 'SNACKPACK_RASPBERRY',
    'SNACKPACK_STRAWBERRY', 'SNACKPACK_VANILLA',
    'TRANSLATOR_ASTRO_BLACK', 'TRANSLATOR_ECLIPSE_CHARCOAL',
    'TRANSLATOR_GRAPHITE_MIST', 'TRANSLATOR_SPACE_GRAY', 'TRANSLATOR_VOID_BLUE',
    'UV_VISOR_AMBER', 'UV_VISOR_MAGENTA', 'UV_VISOR_ORANGE',
    'UV_VISOR_RED', 'UV_VISOR_YELLOW',
]

# Products consumed by structured strategies. Default-MM skips these unless
# we explicitly want both layers running on the same product (we don't).
RESERVED = (set(PEBBLES_LEGS) | {SP_PAIR_A, SP_PAIR_B}
            | {p[0] for p in COINT_PAIRS} | {p[1] for p in COINT_PAIRS}
            | set(MICROCHIP_PRODUCTS))


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

    @staticmethod
    def _mid(od: OrderDepth) -> Optional[float]:
        if od and od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    @staticmethod
    def _best_bid_ask(od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        bb = max(od.buy_orders) if od and od.buy_orders else None
        ba = min(od.sell_orders) if od and od.sell_orders else None
        return bb, ba

    @staticmethod
    def _top_ask_size(od: OrderDepth) -> int:
        if od and od.sell_orders:
            ba = min(od.sell_orders)
            return -od.sell_orders[ba]
        return 0

    @staticmethod
    def _top_bid_size(od: OrderDepth) -> int:
        if od and od.buy_orders:
            bb = max(od.buy_orders)
            return od.buy_orders[bb]
        return 0

    # ====================================================================
    # ===== RECIPE A: PEBBLES BASKET ARBITRAGE ==========================
    # ====================================================================

    def _trade_pebbles_basket(self, state: TradingState, data: dict) -> Dict[str, List[Order]]:
        """Detect and execute the basket arbitrage in a single tick.

        Long arb: when sum of best asks < 50000, buy 1 of each leg.
        Short arb: never observed in EDA (bid spread too wide), but coded
                   symmetrically just in case.
        Unwind: when basket sum is back inside ±PEBBLES_DEV_BAND of target,
                we don't need to actively unwind — by construction the basket
                is pinned to 50,000 so any future fill at-mid clears the trade
                profitably. Default-MM doesn't run on PEBBLES so legs sit until
                someone else takes them.
        """
        result: Dict[str, List[Order]] = {leg: [] for leg in PEBBLES_LEGS}

        ods = {leg: state.order_depths.get(leg) for leg in PEBBLES_LEGS}
        if any(od is None for od in ods.values()):
            return result
        for od in ods.values():
            if not od.buy_orders or not od.sell_orders:
                return result

        best_bids = {leg: max(ods[leg].buy_orders) for leg in PEBBLES_LEGS}
        best_asks = {leg: min(ods[leg].sell_orders) for leg in PEBBLES_LEGS}
        ask_sizes = {leg: -ods[leg].sell_orders[best_asks[leg]] for leg in PEBBLES_LEGS}
        bid_sizes = {leg: ods[leg].buy_orders[best_bids[leg]] for leg in PEBBLES_LEGS}

        buy_basket = sum(best_asks.values())
        sell_basket = sum(best_bids.values())

        positions = {leg: state.position.get(leg, 0) for leg in PEBBLES_LEGS}

        # --- Long arb: buy basket cheap ---
        if buy_basket < PEBBLES_TARGET_SUM:
            edge = PEBBLES_TARGET_SUM - buy_basket
            # Capacity per leg = min(top-of-ask-book size, room to position limit)
            caps = []
            for leg in PEBBLES_LEGS:
                room = DEFAULT_LIMIT - positions[leg]
                caps.append(min(ask_sizes[leg], max(0, room)))
            # Also cap by basket inventory ceiling (sum of long basket positions)
            current_basket_pos = min(positions.values())  # how many full baskets we hold long
            basket_room = max(0, PEBBLES_BASKET_LIMIT - current_basket_pos)
            qty = min(min(caps), basket_room)
            if qty > 0 and edge > 0:
                for leg in PEBBLES_LEGS:
                    result[leg].append(Order(leg, best_asks[leg], qty))
                data['pebbles_arb_count'] = data.get('pebbles_arb_count', 0) + 1

        # --- Short arb: sell basket rich ---
        elif sell_basket > PEBBLES_TARGET_SUM:
            edge = sell_basket - PEBBLES_TARGET_SUM
            caps = []
            for leg in PEBBLES_LEGS:
                room = DEFAULT_LIMIT + positions[leg]
                caps.append(min(bid_sizes[leg], max(0, room)))
            current_basket_pos = max(positions.values())  # if all positions are negative, max is least-negative
            basket_room = max(0, PEBBLES_BASKET_LIMIT + current_basket_pos)
            qty = min(min(caps), basket_room)
            if qty > 0 and edge > 0:
                for leg in PEBBLES_LEGS:
                    result[leg].append(Order(leg, best_bids[leg], -qty))
                data['pebbles_arb_count'] = data.get('pebbles_arb_count', 0) + 1

        # --- Passive unwind: when basket is in MID regime and we hold inventory,
        #     post a flatten order at the inside book. The sum-50000 invariant
        #     guarantees the trade is profitable on any fill above our entry. ---
        else:
            for leg in PEBBLES_LEGS:
                pos = positions[leg]
                if pos > 0:
                    # Long this leg → post offer 1 tick above best_ask floor (inside spread)
                    px = best_asks[leg]
                    qty = min(pos, ask_sizes[leg] if ask_sizes[leg] > 0 else pos)
                    if qty > 0:
                        result[leg].append(Order(leg, px, -qty))
                elif pos < 0:
                    px = best_bids[leg]
                    qty = min(-pos, bid_sizes[leg] if bid_sizes[leg] > 0 else -pos)
                    if qty > 0:
                        result[leg].append(Order(leg, px, qty))

        return result

    # ====================================================================
    # ===== RECIPE B: SNACKPACK CHOC/VANILLA Z-SCORE PAIR ===============
    # ====================================================================

    def _trade_snackpack_pair(self, state: TradingState, data: dict) -> Dict[str, List[Order]]:
        """Z-score mean reversion on (CHOC + VANILLA).

        High z (sum above mean) → short both legs.
        Low z  (sum below mean) → long both legs.
        Exit at |z| ≤ Z_EXIT.
        """
        result: Dict[str, List[Order]] = {SP_PAIR_A: [], SP_PAIR_B: []}

        od_a = state.order_depths.get(SP_PAIR_A)
        od_b = state.order_depths.get(SP_PAIR_B)
        if od_a is None or od_b is None:
            return result
        mid_a = self._mid(od_a)
        mid_b = self._mid(od_b)
        if mid_a is None or mid_b is None:
            return result

        pair = mid_a + mid_b
        z = (pair - SP_PAIR_TARGET) / SP_PAIR_STD

        target_per_leg = data.get('sp_target', 0)
        if z >= SP_PAIR_Z_ENTRY:
            target_per_leg = -SP_PAIR_LIMIT      # short both
        elif z <= -SP_PAIR_Z_ENTRY:
            target_per_leg = +SP_PAIR_LIMIT      # long both
        elif abs(z) <= SP_PAIR_Z_EXIT:
            target_per_leg = 0
        data['sp_target'] = target_per_leg

        for sym, od in ((SP_PAIR_A, od_a), (SP_PAIR_B, od_b)):
            pos = state.position.get(sym, 0)
            delta = target_per_leg - pos
            bb, ba = self._best_bid_ask(od)
            if bb is None or ba is None:
                continue
            if delta > 0:
                # Need to buy — take asks cheaper than mid + 1, otherwise post bid
                taken = 0
                for px in sorted(od.sell_orders):
                    if taken >= delta:
                        break
                    avail = -od.sell_orders[px]
                    qty = min(avail, delta - taken)
                    if qty > 0:
                        result[sym].append(Order(sym, px, qty))
                        taken += qty
                if taken < delta:
                    # Passive bid above the inside bid, capped below the ask
                    bid_px = min(ba - 1, bb + 1)
                    result[sym].append(Order(sym, bid_px, delta - taken))
            elif delta < 0:
                taken = 0
                for px in sorted(od.buy_orders, reverse=True):
                    if taken >= -delta:
                        break
                    avail = od.buy_orders[px]
                    qty = min(avail, -delta - taken)
                    if qty > 0:
                        result[sym].append(Order(sym, px, -qty))
                        taken += qty
                if taken < -delta:
                    ask_px = max(bb + 1, ba - 1)
                    result[sym].append(Order(sym, ask_px, -(-delta - taken)))

        return result

    # ====================================================================
    # ===== RECIPE C: CROSS-CLUSTER COINTEGRATION ========================
    # ====================================================================

    def _trade_coint(self, state: TradingState, data: dict) -> Dict[str, List[Order]]:
        """Trade the residual of `target = beta·hub + intercept` for each pair.
        Long target / short hub when residual is far below 0; vice versa.

        Note: UV_VISOR_AMBER appears as the hub in two pairs with opposite β,
        so per-trade hedge requirements partially net out. We sum them per
        hub product and emit one consolidated order set.
        """
        result: Dict[str, List[Order]] = {}
        # accumulator for hub net delta-position target (we don't actively hedge
        # the hub here — the residual trade implicitly hedges both legs)
        target_positions: Dict[str, int] = {}

        for target_sym, hub_sym, beta, intercept, res_std, max_lots in COINT_PAIRS:
            od_t = state.order_depths.get(target_sym)
            od_h = state.order_depths.get(hub_sym)
            if od_t is None or od_h is None:
                continue
            mid_t = self._mid(od_t)
            mid_h = self._mid(od_h)
            if mid_t is None or mid_h is None:
                continue
            theo = beta * mid_h + intercept
            residual = mid_t - theo
            z = residual / res_std

            # Per-pair state key (so each pair has its own latching target)
            key_target = f'coint_target_{target_sym}'
            key_hub    = f'coint_hub_{target_sym}'
            t_target = data.get(key_target, 0)
            h_target = data.get(key_hub, 0)

            if z >= COINT_Z_ENTRY:
                # target is rich vs hub → short target, long hub (hedged by β)
                t_target = -max_lots
                h_target = int(round(max_lots * beta))
            elif z <= -COINT_Z_ENTRY:
                t_target = +max_lots
                h_target = -int(round(max_lots * beta))
            elif abs(z) <= COINT_Z_EXIT:
                t_target = 0
                h_target = 0
            data[key_target] = t_target
            data[key_hub] = h_target

            # accumulate desired hub position (sum across all pairs sharing this hub)
            target_positions[target_sym] = target_positions.get(target_sym, 0) + t_target
            target_positions[hub_sym]    = target_positions.get(hub_sym, 0) + h_target

        # Convert net target positions into orders
        for sym, target in target_positions.items():
            od = state.order_depths.get(sym)
            if od is None:
                continue
            pos = state.position.get(sym, 0)
            delta = target - pos
            if delta == 0:
                continue
            # Cap by position limit
            if delta > 0:
                delta = min(delta, DEFAULT_LIMIT - pos)
            else:
                delta = max(delta, -DEFAULT_LIMIT - pos)
            if delta == 0:
                continue

            bb, ba = self._best_bid_ask(od)
            if bb is None or ba is None:
                continue
            orders: List[Order] = []
            if delta > 0:
                # Cross to take if a tight ask is cheap; else post inside
                if ba is not None:
                    avail = -od.sell_orders[ba]
                    qty = min(avail, delta)
                    if qty > 0:
                        orders.append(Order(sym, ba, qty))
                        delta -= qty
                if delta > 0:
                    bid_px = min(ba - 1, bb + 1)
                    orders.append(Order(sym, bid_px, delta))
            else:
                if bb is not None:
                    avail = od.buy_orders[bb]
                    qty = min(avail, -delta)
                    if qty > 0:
                        orders.append(Order(sym, bb, -qty))
                        delta += qty
                if delta < 0:
                    ask_px = max(bb + 1, ba - 1)
                    orders.append(Order(sym, ask_px, delta))
            result[sym] = result.get(sym, []) + orders

        return result

    # ====================================================================
    # ===== RECIPE D: MICROCHIP BUY-PRESSURE QUOTING =====================
    # ====================================================================

    def _trade_microchips(self, state: TradingState) -> Dict[str, List[Order]]:
        """Quote MICROCHIP family with skewed offer (tight) / bid (wider).
        XIRECS bot consistently lifts MICROCHIP offers at +0.20 above mid.
        """
        result: Dict[str, List[Order]] = {}
        for sym in MICROCHIP_PRODUCTS:
            od = state.order_depths.get(sym)
            if od is None or not od.buy_orders or not od.sell_orders:
                continue
            mid = self._mid(od)
            if mid is None:
                continue
            bb, ba = self._best_bid_ask(od)
            pos = state.position.get(sym, 0)

            bid_px = min(int(math.floor(mid - MICROCHIP_BID_OFFSET)), bb)
            ask_px = max(int(math.ceil(mid + MICROCHIP_ASK_OFFSET)), ba)
            # Don't cross
            if bid_px >= ask_px:
                bid_px = int(math.floor(mid)) - 1
                ask_px = int(math.ceil(mid)) + 1

            buy_cap = DEFAULT_LIMIT - pos
            sell_cap = DEFAULT_LIMIT + pos
            orders: List[Order] = []
            if buy_cap > 0:
                orders.append(Order(sym, bid_px, min(MICROCHIP_QUOTE_SIZE, buy_cap)))
            if sell_cap > 0:
                orders.append(Order(sym, ask_px, -min(MICROCHIP_QUOTE_SIZE, sell_cap)))
            result[sym] = orders
        return result

    # ====================================================================
    # ===== RECIPE E: DEFAULT MM (everything not claimed by A-D) =========
    # ====================================================================

    def _trade_default_mm(self, state: TradingState) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}
        for sym in ALL_PRODUCTS:
            if sym in RESERVED:
                continue
            od = state.order_depths.get(sym)
            if od is None or not od.buy_orders or not od.sell_orders:
                continue
            mid = self._mid(od)
            if mid is None:
                continue
            bb, ba = self._best_bid_ask(od)
            pos = state.position.get(sym, 0)

            # Slight sell-side skew across the universe
            fair = mid + DEFAULT_FV_SKEW
            bid_px = min(int(math.floor(fair - DEFAULT_BID_OFFSET)), bb)
            ask_px = max(int(math.ceil(fair + DEFAULT_ASK_OFFSET)), ba)
            if bid_px >= ask_px:
                bid_px = int(math.floor(mid)) - 1
                ask_px = int(math.ceil(mid)) + 1

            buy_cap = DEFAULT_LIMIT - pos
            sell_cap = DEFAULT_LIMIT + pos
            # Position-aware skew: shrink the side we don't want more of
            bid_sz = ask_sz = DEFAULT_QUOTE_SIZE
            if pos >= DEFAULT_LIMIT // 2:
                bid_sz = max(5, DEFAULT_QUOTE_SIZE // 2)
            elif pos <= -DEFAULT_LIMIT // 2:
                ask_sz = max(5, DEFAULT_QUOTE_SIZE // 2)

            orders: List[Order] = []
            if buy_cap > 0:
                orders.append(Order(sym, bid_px, min(bid_sz, buy_cap)))
            if sell_cap > 0:
                orders.append(Order(sym, ask_px, -min(ask_sz, sell_cap)))
            result[sym] = orders
        return result

    # ====================================================================
    # ===== MAIN ENTRY ====================================================
    # ====================================================================

    def run(self, state: TradingState):
        data = self._load(state.traderData)
        result: Dict[str, List[Order]] = {}

        # Recipe A — PEBBLES basket arb (highest priority, deterministic)
        for sym, orders in self._trade_pebbles_basket(state, data).items():
            if orders:
                result[sym] = orders

        # Recipe B — SNACKPACK CHOC/VANILLA z-score
        for sym, orders in self._trade_snackpack_pair(state, data).items():
            if orders:
                result[sym] = orders

        # Recipe C — Cross-cluster cointegration
        for sym, orders in self._trade_coint(state, data).items():
            if orders:
                result[sym] = orders

        # Recipe D — MICROCHIP buy-pressure quoting
        for sym, orders in self._trade_microchips(state).items():
            if orders:
                result[sym] = orders

        # Recipe E — Default MM for everything else
        for sym, orders in self._trade_default_mm(state).items():
            if orders:
                result[sym] = orders

        return result, 0, self._save(data)
