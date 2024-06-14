# Predicting stocks using technical indicators (trix, ema)

This trading strategy is designed for the [Quantiacs](https://quantiacs.com/contest) platform, which hosts competitions
for trading algorithms. Detailed information about the competitions is available on
the [official Quantiacs website](https://quantiacs.com/contest).

## How to Run the Strategy

### In an Online Environment

The strategy can be executed in an online environment using Jupiter or JupiterLab on
the [Quantiacs personal dashboard](https://quantiacs.com/personalpage/homepage). To do this, clone the template in your
personal account.

### In a Local Environment

To run the strategy locally, you need to install the [Quantiacs Toolbox](https://github.com/quantiacs/toolbox).

## Strategy Overview

This notebook provides a multi-factor trading strategy for the NASDAQ-100 index, leveraging technical indicators such as
TRIX and EMA. It defines three functions (`multi_trix_v3`, `multi_ema_v3`, and `multi_ema_v4`) to calculate weights
based on combinations of TRIX and EMA signals, applying these to determine trading positions. The strategy computes
aggregate weights from multiple sets of parameters, ensuring liquidity constraints are met. It then calculates
performance statistics and plots the equity curve. Finally, the weights are validated and saved for participation in a
competition.

Key components:

1. **Technical Indicators**: Uses TRIX and EMA to derive trading signals.
2. **Weight Calculation**: Combines signals from multiple parameter sets.
3. **Liquidity Check**: Ensures sufficient liquidity before executing trades.
4. **Performance Analysis**: Computes and visualizes performance metrics.
5. **Validation and Output**: Checks and writes the final weights for competition submission.

```python
from IPython.display import display
import xarray as xr
import qnt.data as qndata
import qnt.output as qnout
import qnt.ta as qnta
import qnt.stats as qns


def multi_trix_v3(data, params):
    s_ = qnta.trix(data.sel(field='high'), params[0])
    w_1 = s_.shift(time=params[1]) > s_.shift(time=params[2])
    w_2 = s_.shift(time=params[3]) > s_.shift(time=params[4])
    weights = (w_1 * w_2) * data.sel(field="is_liquid")
    return weights.fillna(0)


def multi_ema_v3(data, params):
    s_ = qnta.ema(data.sel(field='high'), params[0])
    w_1 = s_.shift(time=params[1]) > s_.shift(time=params[2])
    w_2 = s_.shift(time=params[3]) > s_.shift(time=params[4])
    weights = (w_1 * w_2) * data.sel(field="is_liquid")
    return weights.fillna(0)


def multi_ema_v4(data, params):
    s_ = qnta.trix(data.sel(field='high'), 30)
    w_1 = s_.shift(time=params[0]) > s_.shift(time=params[1])
    s_ = qnta.ema(data.sel(field='high'), params[2])
    w_2 = s_.shift(time=params[3]) > s_.shift(time=params[4])
    weights = (w_1 * w_2) * data.sel(field="is_liquid")
    return weights.fillna(0)


data = qndata.stocks.load_ndx_data(min_date="2005-01-01")

weights_1 = multi_trix_v3(data, [87, 135, 108, 13, 114])
weights_2 = multi_trix_v3(data, [89, 8, 101, 148, 36])
weights_3 = multi_trix_v3(data, [196, 125, 76, 12, 192])
weights_4 = multi_ema_v3(data, [69, 47, 57, 7, 41])

weights_f = (weights_1 + weights_2) * weights_3 * weights_4

weights_5 = multi_trix_v3(data, [89, 139, 22, 8, 112])
weights_6 = multi_trix_v3(data, [92, 139, 20, 10, 110])
weights_7 = multi_ema_v4(data, [13, 134, 42, 66, 133])

weights_t = (weights_5 + weights_6) * weights_7 + weights_3

weights_all = 4 * weights_f + weights_t


def get_enough_bid_for(weights_):
    time_traded = weights_.time[abs(weights_).fillna(0).sum('asset') > 0]
    is_strategy_traded = len(time_traded)
    if is_strategy_traded:
        return xr.where(weights_.time < time_traded.min(), data.sel(field="is_liquid"), weights_)
    return weights_


weights_new = get_enough_bid_for(weights_all)
weights_new = weights_new.sel(time=slice("2006-01-01", None))

weights = qnout.clean(output=weights_new, data=data, kind="stocks_nasdaq100")


def print_statistic(data, weights_all):
    import qnt.stats as qnstats

    stats = qnstats.calc_stat(data, weights_all)
    display(stats.to_pandas().tail(5))
    # graph
    performance = stats.to_pandas()["equity"]
    import qnt.graph as qngraph

    qngraph.make_plot_filled(performance.index, performance, name="PnL (Equity)", type="log")


print_statistic(data, weights)
weights = weights.sel(time=slice("2006-01-01", None))

qnout.check(weights, data, "stocks_nasdaq100")
qnout.write(weights)  # To participate in the competition, save this code in a separate cell.

```
