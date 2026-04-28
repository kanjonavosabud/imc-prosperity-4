from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
POS_LIMIT = {PEPPER: 60, OSMIUM: 50}
PEPPER_HALF_SPREAD = 6
OSMIUM_HALF_SPREAD = 7
OSMIUM_TAKE_THRESH = 5
SKEW_PER_UNIT = 0.15
PEPPER_DAY_BASES = {-1: 11000, 0: 12000, 1: 13000}

class Trader:
    def _max_bid(self, od: OrderDepth):
        if not od.buy_orders:
            return None
        return max(od.buy_orders.keys())
    def _min_ask(self, od: OrderDepth):
        if not od.sell_orders:
            return None
        return min(od.sell_orders.keys())
    def _midpoint(self, od: OrderDepth):
        b = self._max_bid(od)
        a = self._min_ask(od)
        if b is not None and a is not None:
            return (b + a) / 2
        return None
    def _ofi(self, od: OrderDepth):
        buy = sum(od.buy_orders.values()) if od.buy_orders else 0
        sell = sum(abs(v) for v in od.sell_orders.values()) if od.sell_orders else 0
        t = buy + sell
        return (buy - sell) / t if t else 0
    def _pepper_day(self, m, ts):
        for d, b in PEPPER_DAY_BASES.items():
            if abs(m - (b + ts * 0.001)) < 200:
                return d
        return 0
    def _limit_qty(self, qty, pos, lim, typ):
        if typ == "buy":
            return min(qty, lim - pos)
        return min(qty, lim + pos)
    def _pepper_orders(self, od, p, t, d):
        out = []
        l = POS_LIMIT[PEPPER]
        m = self._midpoint(od)
        if m is None or m < 5000:
            return out
        dd = self._pepper_day(m, t)
        d["pepper_day"] = dd
        fv = PEPPER_DAY_BASES[dd] + t * 0.001
        shift = int(round(p * SKEW_PER_UNIT))
        pbid = int(math.floor(fv - PEPPER_HALF_SPREAD - shift))
        pask = int(math.ceil(fv + PEPPER_HALF_SPREAD - shift))
        for ak, av in sorted(od.sell_orders.items()):
            if ak < fv - 1:
                q = self._limit_qty(-av, p, l, "buy")
                if q > 0:
                    out.append(Order(PEPPER, ak, q))
                    p += q
        for bk, bv in sorted(od.buy_orders.items(), reverse=True):
            if bk > fv + 1:
                q = self._limit_qty(bv, p, l, "sell")
                if q > 0:
                    out.append(Order(PEPPER, bk, -q))
                    p -= q
        bq = self._limit_qty(10, p, l, "buy")
        sq = self._limit_qty(10, p, l, "sell")
        if bq > 0 and pbid > 0:
            out.append(Order(PEPPER, pbid, bq))
        if sq > 0 and pask > 0:
            out.append(Order(PEPPER, pask, -sq))
        return out
    def _osmium_orders(self, od, p, t, d):
        res = []
        l = POS_LIMIT[OSMIUM]
        fair = 10000
        m = self._midpoint(od)
        if m is None:
            return res
        ofi_val = self._ofi(od)
        tilt = int(round(ofi_val * 2))
        inv_skew = int(round(p * SKEW_PER_UNIT))
        pbid = fair - OSMIUM_HALF_SPREAD + tilt - inv_skew
        pask = fair + OSMIUM_HALF_SPREAD + tilt - inv_skew
        ba = self._min_ask(od)
        bb = self._max_bid(od)
        if ba is not None and ba < fair - OSMIUM_TAKE_THRESH:
            for ak, av in sorted(od.sell_orders.items()):
                if ak < fair - OSMIUM_TAKE_THRESH:
                    q = self._limit_qty(-av, p, l, "buy")
                    if q > 0:
                        res.append(Order(OSMIUM, ak, q))
                        p += q
        if bb is not None and bb > fair + OSMIUM_TAKE_THRESH:
            for bk, bv in sorted(od.buy_orders.items(), reverse=True):
                if bk > fair + OSMIUM_TAKE_THRESH:
                    q = self._limit_qty(bv, p, l, "sell")
                    if q > 0:
                        res.append(Order(OSMIUM, bk, -q))
                        p -= q
        bq = self._limit_qty(8, p, l, "buy")
        sq = self._limit_qty(8, p, l, "sell")
        if bq > 0:
            res.append(Order(OSMIUM, int(pbid), bq))
        if sq > 0:
            res.append(Order(OSMIUM, int(pask), -sq))
        return res
    def run(self, state: TradingState):
        try:
            dat = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            dat = {}
        ret = {}
        ts = state.timestamp
        if PEPPER in state.order_depths:
            pos = state.position.get(PEPPER, 0)
            od = state.order_depths[PEPPER]
            ret[PEPPER] = self._pepper_orders(od, pos, ts, dat)
        if OSMIUM in state.order_depths:
            pos = state.position.get(OSMIUM, 0)
            od = state.order_depths[OSMIUM]
            ret[OSMIUM] = self._osmium_orders(od, pos, ts, dat)
        td = json.dumps(dat)
        conversions = 0
        return ret, conversions, td
    def bid(self) -> int:
        return 2500
   
