from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math

# ─────────────────────────────────────────────────────────────────────────────
# IMC Prosperity 4 – Round 2 Trader
# Products: INTARIAN_PEPPER_ROOT, ASH_COATED_OSMIUM
#
# STRATEGIES:
#   PEPPER  → FV-anchored market making. Fair value is known analytically:
#             FV(day, t) = (11 + day) * 1000 + timestamp * 0.001
#             We undercut the standing best bid/ask by posting 1 tick inside.
#             Inventory skew adjusts quotes toward neutral when position drifts.
#
#   OSMIUM  → Mean-reversion market making. True price ≈ 10,000 (stationary).
#             AR(1) beta ≈ 0.73, half-life ≈ 2 ticks – very fast reversion.
#             Order flow imbalance (OFI) tilts our quotes pre-emptively.
#             We take mispriced quotes aggressively when price deviates >5 from 10k.
# ─────────────────────────────────────────────────────────────────────────────

PEPPER  = "INTARIAN_PEPPER_ROOT"
OSMIUM  = "ASH_COATED_OSMIUM"

# Position limits (assumed – adjust if actual limits differ)
POS_LIMIT = {PEPPER: 60, OSMIUM: 50}

# Per-product half-spread we target for our passive quotes
PEPPER_HALF_SPREAD  = 6    # FV ± 6  (market quotes at ±7.6, so we undercut)
OSMIUM_HALF_SPREAD  = 7    # 10000 ± 7 (market quotes ±8, tighten slightly)

# Osmium: aggressively take if mid deviates beyond this from 10k
OSMIUM_TAKE_THRESH  = 5

# Inventory skew: per unit of position, shift both quotes by this many ticks
# towards reducing the position (prevents limit lock-up)
SKEW_PER_UNIT = 0.15   # applied to both bid and ask

# Day-of-competition value (must be updated if/when the day changes).
# The engine does not expose the day number directly, so we infer it from
# price levels each tick and cache in traderData.
PEPPER_DAY_BASES = {-1: 11000, 0: 12000, 1: 13000}


class Trader:

    # ── helpers ──────────────────────────────────────────────────────────────

    def _best_bid(self, od: OrderDepth):
        return max(od.buy_orders.keys()) if od.buy_orders else None

    def _best_ask(self, od: OrderDepth):
        return min(od.sell_orders.keys()) if od.sell_orders else None

    def _mid(self, od: OrderDepth):
        bb, ba = self._best_bid(od), self._best_ask(od)
        if bb and ba:
            return (bb + ba) / 2
        return None

    def _ofi(self, od: OrderDepth) -> float:
        """Order Flow Imbalance: (bid_vol - ask_vol) / total_vol at level 1."""
        bvol = sum(od.buy_orders.values()) if od.buy_orders else 0
        avol = sum(abs(v) for v in od.sell_orders.values()) if od.sell_orders else 0
        total = bvol + avol
        return (bvol - avol) / total if total > 0 else 0.0

    def _infer_pepper_day(self, mid: float, ts: int) -> int:
        """Infer which day we are in based on current mid price."""
        for day, base in PEPPER_DAY_BASES.items():
            expected = base + ts * 0.001
            if abs(mid - expected) < 200:
                return day
        return 0

    def _clamp_qty(self, qty: int, pos: int, limit: int, side: str) -> int:
        """Ensure orders don't breach position limit."""
        if side == "buy":
            return min(qty, limit - pos)
        else:
            return min(qty, limit + pos)

    # ── PEPPER strategy ──────────────────────────────────────────────────────

    def _trade_pepper(
        self,
        od: OrderDepth,
        pos: int,
        ts: int,
        data: dict,
    ) -> List[Order]:

        orders: List[Order] = []
        limit = POS_LIMIT[PEPPER]
        mid = self._mid(od)
        if mid is None or mid < 5000:
            return orders

        # Infer / cache current day
        day = self._infer_pepper_day(mid, ts)
        data["pepper_day"] = day

        fv = PEPPER_DAY_BASES[day] + ts * 0.001

        # Inventory skew: positive position → shift quotes down to encourage selling
        skew = int(round(pos * SKEW_PER_UNIT))

        bid_price = int(math.floor(fv - PEPPER_HALF_SPREAD - skew))
        ask_price = int(math.ceil(fv + PEPPER_HALF_SPREAD - skew))

        # ── PHASE 1: TAKE mispriced quotes ──
        # Buy anything below fv - 1 (underpriced asks)
        for ask, vol in sorted(od.sell_orders.items()):
            if ask < fv - 1:
                qty = self._clamp_qty(-vol, pos, limit, "buy")
                if qty > 0:
                    orders.append(Order(PEPPER, ask, qty))
                    pos += qty

        # Sell anything above fv + 1 (overpriced bids)
        for bid, vol in sorted(od.buy_orders.items(), reverse=True):
            if bid > fv + 1:
                qty = self._clamp_qty(vol, pos, limit, "sell")
                if qty > 0:
                    orders.append(Order(PEPPER, bid, -qty))
                    pos -= qty

        # ── PHASE 2: POST passive quotes ──
        buy_qty  = self._clamp_qty(10, pos, limit, "buy")
        sell_qty = self._clamp_qty(10, pos, limit, "sell")

        if buy_qty > 0 and bid_price > 0:
            orders.append(Order(PEPPER, bid_price, buy_qty))
        if sell_qty > 0 and ask_price > 0:
            orders.append(Order(PEPPER, ask_price, -sell_qty))

        return orders

    # ── OSMIUM strategy ──────────────────────────────────────────────────────

    def _trade_osmium(
        self,
        od: OrderDepth,
        pos: int,
        ts: int,
        data: dict,
    ) -> List[Order]:

        orders: List[Order] = []
        limit = POS_LIMIT[OSMIUM]
        FV = 10000

        mid = self._mid(od)
        if mid is None:
            return orders

        ofi = self._ofi(od)
        # OFI-based quote tilt: corr of 0.38 with next return
        # If LOB is bid-heavy (ofi > 0), price likely to tick up → raise both quotes slightly
        ofi_tilt = int(round(ofi * 2))  # max ±2 ticks

        # Inventory skew
        skew = int(round(pos * SKEW_PER_UNIT))

        bid_price = FV - OSMIUM_HALF_SPREAD + ofi_tilt - skew
        ask_price = FV + OSMIUM_HALF_SPREAD + ofi_tilt - skew

        # ── PHASE 1: TAKE aggressively when price deviates ──
        # If best ask is well below FV, buy it
        best_ask = self._best_ask(od)
        best_bid = self._best_bid(od)

        if best_ask is not None and best_ask < FV - OSMIUM_TAKE_THRESH:
            for ask, vol in sorted(od.sell_orders.items()):
                if ask < FV - OSMIUM_TAKE_THRESH:
                    qty = self._clamp_qty(-vol, pos, limit, "buy")
                    if qty > 0:
                        orders.append(Order(OSMIUM, ask, qty))
                        pos += qty

        if best_bid is not None and best_bid > FV + OSMIUM_TAKE_THRESH:
            for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                if bid > FV + OSMIUM_TAKE_THRESH:
                    qty = self._clamp_qty(vol, pos, limit, "sell")
                    if qty > 0:
                        orders.append(Order(OSMIUM, bid, -qty))
                        pos -= qty

        # ── PHASE 2: POST passive quotes ──
        buy_qty  = self._clamp_qty(8, pos, limit, "buy")
        sell_qty = self._clamp_qty(8, pos, limit, "sell")

        if buy_qty > 0:
            orders.append(Order(OSMIUM, int(bid_price), buy_qty))
        if sell_qty > 0:
            orders.append(Order(OSMIUM, int(ask_price), -sell_qty))

        return orders

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        # Load persistent state
        try:
            data: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        result: Dict[str, List[Order]] = {}
        ts = state.timestamp

        # ── PEPPER ──
        if PEPPER in state.order_depths:
            pos = state.position.get(PEPPER, 0)
            od  = state.order_depths[PEPPER]
            result[PEPPER] = self._trade_pepper(od, pos, ts, data)

        # ── OSMIUM ──
        if OSMIUM in state.order_depths:
            pos = state.position.get(OSMIUM, 0)
            od  = state.order_depths[OSMIUM]
            result[OSMIUM] = self._trade_osmium(od, pos, ts, data)

        # Persist state
        trader_data = json.dumps(data)

        # ── BID for extra market access ──
        # See reasoning below – we bid 2,500 XIRECs.
        conversions = 0
        return result, conversions, trader_data

    def bid(self) -> int:
        """
        Bid for extra market access (25% more quotes fitting within
        the existing distribution).

        REASONING:
        ──────────────────────────────────────────────────────────────
        The extra quotes are additional bot orders inserted within the
        existing LOB distribution. They are not outside the spread –
        they fill in between existing levels, giving us more counterparties.

        We estimate value as follows:

        OSMIUM:
          ~465 bot trades/day × avg 5.1 units × ~8 ticks edge × 50% fill rate
          = ~9,500 XIRECs/day baseline
          25% more counterparties → +2,376/day → +7,128 over 3 days

        PEPPER:
          ~332 trades/day × avg 5.0 units × ~6 ticks edge × 50% fill rate
          = ~5,000 XIRECs/day baseline
          25% more counterparties → +1,260/day → +3,780 over 3 days

        Total gross value ≈ 10,900 XIRECs over 3 days.

        DISCOUNTS applied:
          - Not all extra quotes are at prices we can profitably fill
            (some are at levels we already would have traded)
          - Position limit constraints reduce fill capacity at extremes
          - Other teams bidding + opportunity cost of the bid itself

        Applying ~30% utilisation discount → net value ≈ 7,600.

        GAME-THEORY ADJUSTMENT:
          - This is a sealed bid – bidding our full value is rational
            only if we expect competitors to bid less.
          - If the marginal team bids around the midpoint of the true
            value range, we want to be just above them.
          - We set bid at 2,500: this is ~23% of gross value, giving
            a healthy positive EV even under conservative assumptions,
            while avoiding overbidding if most teams underestimate the
            feature.
          - If the fee turns out to be per-day, this becomes 2,500/day,
            which we'd accept given ~3,600+/day incremental profit.

        Final bid: 2,500 XIRECs
        ──────────────────────────────────────────────────────────────
        """
        return 2500
