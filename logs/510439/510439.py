import json
import math
from datamodel import TradingState, Order
from typing import List


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)


class Trader:

    # IV smile fitted on R4 historical data (3 days, all snapshots, TTE=7d at start):
    #   IV(m_t) = a*m_t^2 + b*m_t + c,  m_t = log(S/K) / sqrt(T)
    # Refit improves smile RMSE from 0.0325 (R3 coefs on R4 data) to 0.0064 — 5x tighter.
    # ATM IV (C term) drops from 0.235 → 0.200 because R4 is genuinely less volatile.
    SMILE_A = 0.018517
    SMILE_B = 0.002661
    SMILE_C = 0.199515

    # TTE in calendar days at the start of round 4 day 1 (per spec: 7 days).
    TTE_INITIAL_DAYS = 7.0

    # IV scalping EMA params (Timo-style)
    DIFF_EMA_ALPHA = 0.05    # ~20-tick window for local equilibrium
    ABS_EMA_ALPHA = 0.01     # ~100-tick window for regime activity

    # === COUNTERPARTY SIGNALS (Round 4 — buyer/seller IDs now visible) ===
    # Empirical analysis on round 4 historical data (3 days, 4281 trades):
    #   Mark 67 BUYS VE: 165 events, +1.9 mean move in 50 ticks  (INFORMED)
    #   Mark 49 SELLS VE: 105 events, -2.0 mean move in 50 ticks (INFORMED)
    #   Mark 38 HYDRO: balanced, both sides UNINFORMED — already captured by MM
    #   Mark 22 sells OTM vouchers: weak signal (-0.18) — skip
    # Strategy: bias VE fair value when an informed Mark recently traded.
    INFORMED_SIGNAL_TICKS = 50            # primary decay window for M67 (100 ts/tick)
    # Mark 67 BUYS VE → +1.97 ticks @ h=100, 96% wr-up. Reactive only: state.market_trades
    # does NOT include bot-bot trades, so we only see M67 when he hits SUBMISSION's offer.
    MARK67_BUY_BIAS = 1.5
    # M49/M22 SELLS are paired with M67 BUYS 99% of the time (same trade record).
    # Theory says price RISES after M49/M22 SELLS (paired insider flow), but empirically
    # in submission 497762 we tested clean removal (=0) and lost $941 on VE vs the
    # DOWN-bias baseline. Without these, our bid stayed too aggressive during paired
    # insider events → flipped position from short→long → captured less spread.
    # Keeping the empirically-tuned DOWN biases despite theoretical mismatch.
    MARK49_SELL_BIAS = 1.5
    MARK22_SELL_BIAS = 1.0
    # HYDROGEL_PACK overlay biases — tiny effects at h=100; not used.
    MARK14H_SELL_BIAS = 0.0
    MARK14H_BUY_BIAS = 0.0
    MARK38H_BUY_BIAS = 0.0

    # === SIZE-BUCKET DEFENSIVE SIGNALS (REACTIVE: only when CP hits SUBMISSION) ===
    # M14 SELL HP @ qty≥6 → -0.47 avg @ h=100, 47% wr-down. When M14 sells to us with
    # large size, we just bought into a forecasted drop → pull BID for 5000 ts so we
    # don't accumulate more bad longs. Submission 497762: fired 1× → +$533 on HP.
    SIZE_DEFENSE_TICKS = 50
    M14_HP_SELL_QTY_THR = 6
    # M01 SELL VE qty 6-7 signal: dropped — fired 0× in the official Day-3 window.

    MM_PARAMS = {
        # HYDROGEL_PACK: wide stable spread (16 ticks), mean-reverting around ~9990,
        # uncorrelated with options. Penny the NPCs to capture the spread.
        # simple_micro=True because full-book microprice has shown HIGH variance officially
        # for HYDROGEL — it has 100% L2 presence so adding more data adds more noise here.
        'HYDROGEL_PACK':       {'limit': 200, 'edge': 12, 'skew': 0.02, 'nl_skew': True, 'simple_micro': True},
        # Deep ITM: intrinsic FV (vega ≈ 0)
        'VEV_4000':            {'limit': 300, 'edge': 12, 'skew': 0.0,   'strike': 4000},
        'VEV_4500':            {'limit': 300, 'edge': 4,  'skew': 0.0,   'strike': 4500},
        # ATM/near-ATM: microprice MM (works well, scalping noise too small)
        'VEV_5000':            {'limit': 300, 'edge': 2,  'skew': 0.0},
        'VEV_5100':            {'limit': 300, 'edge': 2,  'skew': 0.0},
        # Underlying
        'VELVETFRUIT_EXTRACT': {'limit': 200, 'edge': 4,  'skew': 0.003},
    }

    NL_COEFF = 0.8

    # Ornstein-Uhlenbeck mean-reversion for HYDROGEL and VE.
    # Estimated from regression of (X(t+τ) - X(t)) on -(X(t) - μ):
    #   slope = 1 - exp(-θ·τ)  ⇒  θ ≈ 0.00127 (half-life ~550 ticks for HYDROGEL)
    # OU expected fair at horizon τ:  E[X(t+τ)] = μ + (X(t)-μ)·exp(-θτ)
    # Equivalent to blending current FV with μ at weight w = 1 - exp(-θτ).
    # τ=200 gives w ≈ 0.22 — middle ground between robustness and edge.
    # Empirical means from R4 historical data (3 days, 30k snapshots each):
    #   VE: 5247.65 (median 5247.5)  →  prior 5248
    #   HP: 9994.65 (median 9999)    →  prior 9995
    LT_MEAN_PRIOR = {'HYDROGEL_PACK': 9995.0, 'VELVETFRUIT_EXTRACT': 5248.0}
    LT_MEAN_BLEND_PER_PRODUCT = {
        'HYDROGEL_PACK': 0.30,        # τ ≈ 280 ticks (peak via local sweep)
        'VELVETFRUIT_EXTRACT': 0.07,  # τ ≈ 45 ticks for VE — balanced day 2 win
    }
    LT_MEAN_ALPHA = 0.001  # slow adaptive adjustment to μ if regime shifts

    # (Imbalance signal removed — fires only ~107 times in 30k ticks for HYDROGEL,
    # and the OU mean-reversion blend already captures most of the value. Keeping
    # it added noise without measurable benefit in local sweeps.)
    IMBALANCE_FV_SHIFT = {}
    IMBALANCE_RATIO_THR = 999.0

    # IV scalping products: OTM with meaningful vega.
    # `mispricing_buy_thr`: when BS theo > market mid by this much, lift the offer
    # (theoretically positive EV: option converges to fair OR appreciates with VE).
    # `mispricing_sell_thr`: when market mid > BS theo by this much, hit the bid.
    # Negative means disabled (we don't sell because market drift in OTM options
    # historically goes UP with VE, making short positions a losing bet).
    # IV scalping kept dormant (thresholds wide; deviations in our data don't reach them).
    # The mispricing-buy variant we tried was a loser: market doesn't converge to BS fair,
    # it converges to consensus mid, so paying ask to "capture mispricing" just bleeds spread.
    SCALP_PARAMS = {
        'VEV_5200': {'limit': 300, 'strike': 5200, 'thr_open': 1.5, 'thr_close': 0.3, 'thr_activate': 0.6},
        'VEV_5300': {'limit': 300, 'strike': 5300, 'thr_open': 1.2, 'thr_close': 0.3, 'thr_activate': 0.5},
        'VEV_5400': {'limit': 300, 'strike': 5400, 'thr_open': 0.6, 'thr_close': 0.15, 'thr_activate': 0.3},
        'VEV_5500': {'limit': 300, 'strike': 5500, 'thr_open': 0.6, 'thr_close': 0.15, 'thr_activate': 0.3},
    }

    # Pure buy_flow for very far OTM (vega ≈ 0, scalping won't help)
    BUY_PARAMS = {
        'VEV_6000': {'limit': 300},
        'VEV_6500': {'limit': 300},
    }

    def _load_state(self, state: TradingState) -> dict:
        if state.traderData:
            try:
                return json.loads(state.traderData)
            except Exception:
                return {}
        return {}

    def _track_time(self, mem: dict, state: TradingState) -> float:
        last_ts = mem.get('last_ts', None)
        day = mem.get('day', 0)
        if last_ts is not None and state.timestamp < last_ts:
            day += 1
        mem['day'] = day
        mem['last_ts'] = state.timestamp
        elapsed_within_day = state.timestamp / 1_000_000.0
        days_remaining = max(0.5, self.TTE_INITIAL_DAYS - day - elapsed_within_day)
        return days_remaining / 365.0

    def _smile_iv(self, S: float, K: float, T: float) -> float:
        if S <= 0 or K <= 0 or T <= 0:
            return 0.20
        m_t = math.log(S / K) / math.sqrt(T)
        iv = self.SMILE_A * m_t * m_t + self.SMILE_B * m_t + self.SMILE_C
        return max(0.05, iv)

    def _underlying_mid(self, state: TradingState):
        if 'VELVETFRUIT_EXTRACT' not in state.order_depths:
            return None
        od = state.order_depths['VELVETFRUIT_EXTRACT']
        if not od.buy_orders or not od.sell_orders:
            return None
        bb = max(od.buy_orders.keys())
        ba = min(od.sell_orders.keys())
        bv = od.buy_orders[bb]
        av = -od.sell_orders[ba]
        return (bb * av + ba * bv) / (bv + av)

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        mem = self._load_state(state)
        T = self._track_time(mem, state)
        S = self._underlying_mid(state)

        lt_means = mem.setdefault('lt_mean', {})

        # ===== Counterparty detection =====
        # state.market_trades only includes trades where SUBMISSION participated, so
        # detection is REACTIVE: we see a Mark only after they've already hit us.
        # Pull-side defenses then prevent further fills in the same direction.
        signal_horizon_ts = self.INFORMED_SIGNAL_TICKS * 100
        size_defense_ts = self.SIZE_DEFENSE_TICKS * 100
        for src in (state.market_trades, state.own_trades):
            for tr in src.get('VELVETFRUIT_EXTRACT', []):
                buyer = getattr(tr, 'buyer', None)
                seller = getattr(tr, 'seller', None)
                # Mark 67 BUY VE: insider — pull ask defensively (in market_make below).
                if buyer == 'Mark 67':
                    mem['m67_buy_until'] = max(mem.get('m67_buy_until', 0), tr.timestamp + signal_horizon_ts)
                # M49/M22 SELL VE: paired insider flow — apply DOWN bias to FV.
                if seller == 'Mark 49':
                    mem['m49_sell_until'] = max(mem.get('m49_sell_until', 0), tr.timestamp + signal_horizon_ts)
                if seller == 'Mark 22':
                    mem['m22_sell_until'] = max(mem.get('m22_sell_until', 0), tr.timestamp + signal_horizon_ts)
            for tr in src.get('HYDROGEL_PACK', []):
                qty = abs(getattr(tr, 'quantity', 0))
                seller = getattr(tr, 'seller', None)
                # M14 SELL HP @ qty≥6: pull our BID (price drops, don't accumulate longs).
                if seller == 'Mark 14' and qty >= self.M14_HP_SELL_QTY_THR:
                    mem['m14h_high_sell_until'] = max(mem.get('m14h_high_sell_until', 0), tr.timestamp + size_defense_ts)

        for product, params in self.MM_PARAMS.items():
            if product in state.order_depths:
                result[product] = self.market_make(state, product, params, S, lt_means, state.timestamp, mem)

        for product, params in self.SCALP_PARAMS.items():
            if product in state.order_depths and S is not None:
                result[product] = self.iv_scalp(state, product, params, S, T, mem)

        for product, params in self.BUY_PARAMS.items():
            if product in state.order_depths:
                result[product] = self.buy_flow(state, product, params)

        return result, conversions, json.dumps(mem)

    def buy_flow(self, state: TradingState, product: str, params: dict) -> List[Order]:
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        buy_cap = params['limit'] - position
        if buy_cap <= 0:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        # Always place an order. When spread > 1 we can post inside (bid+1) to win
        # FIFO under "worse" matching; when spread == 1 we queue at best_bid as a
        # fallback (occasionally fills via order-depth crossings).
        bid_px = best_bid + 1 if (best_ask - best_bid) > 1 else best_bid
        orders.append(Order(product, bid_px, buy_cap))
        return orders

    def iv_scalp(self, state: TradingState, product: str, params: dict,
                 S: float, T: float, mem: dict) -> List[Order]:
        """Hybrid: passive buy_flow (bid+1) by default; aggressive scalping on extreme deviations.

        EMA-detrended mispricing measures how far the option mid is from its local equilibrium
        (Timo-style). Wide thresholds because round-tripping costs the spread (~1-2 PnL) on
        these tight markets, so we only fire on multi-sigma events.
        """
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        bb = max(order_depth.buy_orders.keys())
        ba = min(order_depth.sell_orders.keys())
        bv = order_depth.buy_orders[bb]
        av = -order_depth.sell_orders[ba]
        mid = (bb + ba) / 2
        spread = ba - bb

        K = params['strike']
        iv = self._smile_iv(S, K, T)
        theo = bs_call(S, K, T, iv)
        theo_diff = mid - theo

        diff_emas = mem.setdefault('diff_ema', {})
        prev_diff_ema = diff_emas.get(product, theo_diff)
        new_diff_ema = self.DIFF_EMA_ALPHA * theo_diff + (1 - self.DIFF_EMA_ALPHA) * prev_diff_ema
        diff_emas[product] = new_diff_ema

        deviation = theo_diff - new_diff_ema

        abs_emas = mem.setdefault('abs_ema', {})
        prev_abs_ema = abs_emas.get(product, abs(deviation))
        new_abs_ema = self.ABS_EMA_ALPHA * abs(deviation) + (1 - self.ABS_EMA_ALPHA) * prev_abs_ema
        abs_emas[product] = new_abs_ema

        thr_open = params['thr_open']
        thr_activate = params['thr_activate']
        limit = params['limit']

        active = new_abs_ema >= thr_activate

        # Aggressive scalping when extreme deviation in active regime
        if active and deviation > thr_open:
            # Market mid above local equilibrium → SHORT (hit bid)
            sell_cap = position + limit
            if sell_cap > 0:
                qty = min(sell_cap, bv)
                if qty > 0:
                    orders.append(Order(product, bb, -qty))
            return orders

        if active and deviation < -thr_open:
            # Market mid below local equilibrium → LONG (lift offer)
            buy_cap = limit - position
            if buy_cap > 0:
                qty = min(buy_cap, av)
                if qty > 0:
                    orders.append(Order(product, ba, qty))
            return orders

        # Default: passive buy_flow inside the spread when we have room, else queue at best_bid.
        buy_cap = limit - position
        if buy_cap > 0:
            bid_px = bb + 1 if spread > 1 else bb
            orders.append(Order(product, bid_px, buy_cap))

        return orders

    def market_make(self, state: TradingState, product: str, params: dict,
                    underlying_mid, lt_means: dict, now_ts: int = 0, mem: dict = None) -> List[Order]:
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        orders: List[Order] = []

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        bid_vol_l1 = order_depth.buy_orders[best_bid]
        ask_vol_l1 = -order_depth.sell_orders[best_ask]

        if params.get('simple_micro'):
            # Top-of-book microprice — lower variance for products with stable spread.
            fair_value = (best_bid * ask_vol_l1 + best_ask * bid_vol_l1) / (bid_vol_l1 + ask_vol_l1)
        else:
            # Volume-weighted microprice using the full order book.
            total_bid_vol = sum(order_depth.buy_orders.values())
            total_ask_vol = sum(-v for v in order_depth.sell_orders.values())
            avg_bid = sum(p * v for p, v in order_depth.buy_orders.items()) / total_bid_vol
            avg_ask = sum(p * (-v) for p, v in order_depth.sell_orders.items()) / total_ask_vol
            fair_value = (avg_bid * total_ask_vol + avg_ask * total_bid_vol) / (total_bid_vol + total_ask_vol)

        # OU-derived mean-reversion blend. Per-product weight reflects estimated
        # mean-reversion speed at the chosen holding horizon (~τ ticks).
        if product in self.LT_MEAN_PRIOR:
            prev_mean = lt_means.get(product, self.LT_MEAN_PRIOR[product])
            new_mean = self.LT_MEAN_ALPHA * mid + (1 - self.LT_MEAN_ALPHA) * prev_mean
            lt_means[product] = new_mean
            w = self.LT_MEAN_BLEND_PER_PRODUCT.get(product, 0.05)
            fair_value = (1 - w) * fair_value + w * new_mean

        # ===== Counterparty FV bias (VE only) =====
        # M67 BUY VE: insider, FV +1.5. M49/M22 SELL VE: empirically DOWN biases work
        # better despite theoretical mismatch (see class constant comments).
        m67_active = False
        if product == 'VELVETFRUIT_EXTRACT' and mem is not None:
            if now_ts < mem.get('m67_buy_until', 0):
                fair_value += self.MARK67_BUY_BIAS
                m67_active = True
            if now_ts < mem.get('m49_sell_until', 0):
                fair_value -= self.MARK49_SELL_BIAS
            if now_ts < mem.get('m22_sell_until', 0):
                fair_value -= self.MARK22_SELL_BIAS

        # Imbalance signal: extreme top-of-book imbalance predicts direction with
        # 70-77% win rate. Shift FV in the direction of the heavier side.
        imb_shift = self.IMBALANCE_FV_SHIFT.get(product)
        if imb_shift is not None and ask_vol_l1 > 0 and bid_vol_l1 > 0:
            if bid_vol_l1 >= self.IMBALANCE_RATIO_THR * ask_vol_l1:
                fair_value += imb_shift
            elif ask_vol_l1 >= self.IMBALANCE_RATIO_THR * bid_vol_l1:
                fair_value -= imb_shift

        if 'strike' in params and underlying_mid is not None:
            intrinsic = underlying_mid - params['strike']
            if intrinsic > 0:
                fair_value = intrinsic
                # Deep ITM has delta ≈ 1 — propagate VE counterparty biases through
                # to the option's intrinsic FV. Without this, when M67 BUY is active
                # we update VE's FV but VEV_4000/VEV_4500 still quote off stale S.
                if mem is not None:
                    if now_ts < mem.get('m67_buy_until', 0):
                        fair_value += self.MARK67_BUY_BIAS
                    if now_ts < mem.get('m49_sell_until', 0):
                        fair_value -= self.MARK49_SELL_BIAS
                    if now_ts < mem.get('m22_sell_until', 0):
                        fair_value -= self.MARK22_SELL_BIAS

        limit = params['limit']
        edge = params['edge']
        skew = params['skew']

        if params.get('nl_skew'):
            pos_ratio = position / limit
            fv_shift = position * skew + pos_ratio * abs(pos_ratio) * limit * skew * self.NL_COEFF
        else:
            fv_shift = position * skew
        fv = fair_value - fv_shift
        fv_int = int(round(fv))

        buy_cap = limit - position
        sell_cap = position + limit

        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price <= fv_int and buy_cap > 0:
                vol = -order_depth.sell_orders[ask_price]
                qty = min(vol, buy_cap)
                orders.append(Order(product, ask_price, qty))
                buy_cap -= qty

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_price >= fv_int and sell_cap > 0:
                vol = order_depth.buy_orders[bid_price]
                qty = min(vol, sell_cap)
                orders.append(Order(product, bid_price, -qty))
                sell_cap -= qty

        my_bid = fv_int - edge
        my_ask = fv_int + edge

        my_bid = min(my_bid, int(mid) - 1)
        my_ask = max(my_ask, int(mid) + 1)

        my_bid = max(my_bid, best_bid + 1)
        my_ask = min(my_ask, best_ask - 1)

        if my_bid >= my_ask:
            my_bid = int(mid) - 1
            my_ask = int(mid) + 1

        # ===== Side-skip defenses =====
        # Pull ASK on M67 BUY VE (insider — price will rise, our ask is stale).
        skip_ask = (product == 'VELVETFRUIT_EXTRACT' and m67_active)
        # Pull BID on M14 high-qty SELL HP (price will drop ~0.47 ticks → don't keep
        # buying). +$533 in submission 497762 with 1 firing.
        skip_bid = (mem is not None and product == 'HYDROGEL_PACK'
                    and now_ts < mem.get('m14h_high_sell_until', 0))

        if buy_cap > 0 and not skip_bid:
            orders.append(Order(product, my_bid, buy_cap))
        if sell_cap > 0 and not skip_ask:
            orders.append(Order(product, my_ask, -sell_cap))

        return orders