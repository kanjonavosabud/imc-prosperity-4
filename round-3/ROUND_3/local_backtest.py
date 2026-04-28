"""
Local backtester for round-3 algorithm against historical CSV data.
Simulates fills with realistic pessimism: passive limits cross only when next-tick
price reaches them; aggressive takes fill at the displayed ask/bid up to displayed size.
"""
import sys, os, json, csv, math
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

# Stub out the buggy datamodel.ConversionObservation by importing only what we need
import types
dm = types.ModuleType('datamodel')
class OrderDepth:
    def __init__(self):
        self.buy_orders = {}
        self.sell_orders = {}
class Order:
    def __init__(self, symbol, price, quantity):
        self.symbol = symbol
        self.price = int(price)
        self.quantity = int(quantity)
    def __repr__(self):
        return f"Order({self.symbol}, {self.price}, {self.quantity})"
class TradingState:
    def __init__(self, traderData, timestamp, listings, order_depths, own_trades, market_trades, position, observations):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
dm.OrderDepth = OrderDepth
dm.Order = Order
dm.TradingState = TradingState
sys.modules['datamodel'] = dm

# Now load the trader
trader_path = os.path.dirname(os.path.abspath(__file__)) + '/../round-3.py'
trader_globals = {'__name__': 'round3'}
exec(open(trader_path).read(), trader_globals)
Trader = trader_globals['Trader']


def load_day(day_idx):
    path = f"prices_round_3_day_{day_idx}.csv"
    rows_by_ts = defaultdict(dict)
    with open(path) as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for r in reader:
            if len(r) < 16:
                continue
            ts = int(r[1]); prod = r[2]
            rows_by_ts[ts][prod] = {
                'bid_p': [int(p) if p else None for p in [r[3], r[5], r[7]]],
                'bid_v': [int(v) if v else 0 for v in [r[4], r[6], r[8]]],
                'ask_p': [int(p) if p else None for p in [r[9], r[11], r[13]]],
                'ask_v': [int(v) if v else 0 for v in [r[10], r[12], r[14]]],
                'mid': float(r[15]) if r[15] else None,
            }
    return sorted(rows_by_ts.keys()), rows_by_ts


def make_book(snapshot):
    od = OrderDepth()
    for p, v in zip(snapshot['bid_p'], snapshot['bid_v']):
        if p is not None and v > 0:
            od.buy_orders[p] = v
    for p, v in zip(snapshot['ask_p'], snapshot['ask_v']):
        if p is not None and v > 0:
            od.sell_orders[p] = -v
    return od


def simulate_day(day_idx, n_ticks=None, position_limits=None):
    if position_limits is None:
        position_limits = {'HYDROGEL_PACK': 60, 'VELVETFRUIT_EXTRACT': 200,
                           'VEV_5000': 200, 'VEV_5100': 200, 'VEV_5200': 200,
                           'VEV_5300': 200, 'VEV_5400': 200, 'VEV_5500': 200,
                           'VEV_4000': 200, 'VEV_4500': 200, 'VEV_6000': 200, 'VEV_6500': 200}

    timestamps, rows = load_day(day_idx)
    if n_ticks is not None:
        timestamps = timestamps[:n_ticks]

    trader = Trader()
    traderData = ''
    position = defaultdict(int)
    cash = 0.0           # cumulative cash (negative = bought, positive = sold)
    trades_log = []

    for i, ts in enumerate(timestamps):
        snap = rows[ts]
        ods = {prod: make_book(s) for prod, s in snap.items()}
        state = TradingState(
            traderData=traderData, timestamp=ts, listings={}, order_depths=ods,
            own_trades={}, market_trades={}, position=dict(position), observations=None,
        )
        result, _, traderData = trader.run(state)

        # Simulate fills against the SAME-tick book (pessimistic but consistent)
        for sym, orders in result.items():
            if sym not in snap:
                continue
            book = snap[sym]
            for o in orders:
                qty = o.quantity
                if qty == 0:
                    continue
                lim = position_limits.get(sym, 200)
                # Position limit guard
                if qty > 0 and position[sym] + qty > lim:
                    qty = lim - position[sym]
                if qty < 0 and position[sym] + qty < -lim:
                    qty = -lim - position[sym]
                if qty == 0:
                    continue

                if qty > 0:
                    # Aggressive buy: fill against asks at o.price or below
                    rem = qty
                    for ap, av in zip(book['ask_p'], book['ask_v']):
                        if ap is None or av <= 0 or rem <= 0:
                            break
                        if ap <= o.price:
                            fill_qty = min(rem, av)
                            cash -= ap * fill_qty
                            position[sym] += fill_qty
                            trades_log.append((ts, sym, 'B', ap, fill_qty))
                            rem -= fill_qty
                else:
                    # Aggressive sell: fill against bids at o.price or above
                    rem = -qty
                    for bp, bv in zip(book['bid_p'], book['bid_v']):
                        if bp is None or bv <= 0 or rem <= 0:
                            break
                        if bp >= o.price:
                            fill_qty = min(rem, bv)
                            cash += bp * fill_qty
                            position[sym] -= fill_qty
                            trades_log.append((ts, sym, 'S', bp, fill_qty))
                            rem -= fill_qty

    # Mark-to-market at last mid
    last_snap = rows[timestamps[-1]]
    mtm = 0.0
    for sym, p in position.items():
        if p == 0:
            continue
        snap = last_snap.get(sym)
        if snap is None or snap['mid'] is None:
            continue
        mtm += p * snap['mid']

    total_pnl = cash + mtm

    # Per-symbol breakdown
    per_sym = defaultdict(lambda: {'cash': 0.0, 'pos': 0, 'mid': 0, 'pnl': 0.0, 'n_trades': 0})
    for ts, sym, side, px, qty in trades_log:
        if side == 'B':
            per_sym[sym]['cash'] -= px * qty
        else:
            per_sym[sym]['cash'] += px * qty
        per_sym[sym]['n_trades'] += 1
    for sym, p in position.items():
        per_sym[sym]['pos'] = p
        snap = last_snap.get(sym)
        if snap and snap['mid'] is not None:
            per_sym[sym]['mid'] = snap['mid']
            per_sym[sym]['pnl'] = per_sym[sym]['cash'] + p * snap['mid']

    return total_pnl, per_sym, trades_log


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    days = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [2]

    print(f"Backtesting {n} ticks per day on days {days}")
    grand_total = 0.0
    for d in days:
        pnl, per_sym, trades = simulate_day(d, n_ticks=n)
        print(f"\n=== Day {d} ({n} ticks) ===")
        print(f"{'symbol':25s}  {'cash':>10s}  {'pos':>5s}  {'mid':>8s}  {'pnl':>10s}  {'#trades':>7s}")
        for sym in sorted(per_sym.keys()):
            d_ = per_sym[sym]
            print(f"  {sym:25s}  {d_['cash']:10.0f}  {d_['pos']:5d}  {d_['mid']:8.2f}  {d_['pnl']:10.0f}  {d_['n_trades']:7d}")
        print(f"  {'TOTAL':25s}  {'':>10s}  {'':>5s}  {'':>8s}  {pnl:10.0f}  {len(trades):7d}")
        grand_total += pnl
    if len(days) > 1:
        print(f"\nGrand total across all days: {grand_total:.0f}")
