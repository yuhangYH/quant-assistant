# Research Notes (Sample)

> These notes are synthetic and written only to demonstrate retrieval. They describe
> a fictional asset, ACME, and do not represent real securities or investment advice.

## Momentum vs mean reversion

Over the sample window, ACME shows a persistent upward drift with month-to-month
noise. Simple momentum signals (holding after a positive month) capture most of the
drift, but the single-month reversals are frequent enough that a naive mean-reversion
overlay tends to fight the trend and give back gains. The working view in these notes
is that momentum dominates at the monthly horizon while any mean-reversion edge lives
at a shorter, intramonth horizon that this dataset cannot see.

## Volatility regime

Realized volatility clusters. Drawdown months (for example the roughly two percent
pullbacks) tend to arrive after the strongest up months, consistent with volatility
clustering rather than a stable level. Position sizing that scales inversely with
recent realized volatility would have reduced the depth of the interim drawdowns
without materially changing the cumulative path.

## Liquidity and volume

Volume rises on up months and thins on pullbacks. That pattern makes execution easier
when adding to a winning position and harder when trying to exit into weakness, which
argues for scaling out gradually rather than in a single block.

## Evaluation discipline

Any backtest on this series should be judged out of sample. The cumulative return looks
attractive in hindsight, but the number that matters is whether a rule set survives on
data it was not tuned on. Report the Sharpe ratio and the maximum drawdown alongside the
cumulative return, never the cumulative return alone.
