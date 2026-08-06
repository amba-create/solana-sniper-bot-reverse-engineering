# Solana Sniper Bot Reverse-Engineering

Entry for the Kaggle competition **Solana Sniper Bot Reverse-Engineering**. The goal is to reconstruct the deployer-selection strategy of an anonymous, high-performing Solana memecoin sniper bot from raw on-chain deploy and wallet-activity data, then build a replica strategy and evaluate whether it could match the bot's real performance.

The full pipeline lives in [`solana_sniper_bot_analysis.ipynb`](./solana_sniper_bot_analysis.ipynb) and is organised into three parts.

## Part 1 — Behavioural analysis

Descriptive analysis of the bot's actual trading behaviour, using its real buy/sell/burn transactions:

- 16,192 buys, 70,812 sells, 2 burns across 16,168 unique tokens
- Entry size (USD): mean 263.25, median 183.72, std 232.51
- Median hold time: 14 blocks/slots
- Hit rate on realized trades: 78.74%, avg win 122.63 USD, avg loss -21.09 USD
- The large majority of exited positions use multi-tranche (staged) selling rather than a single exit

## Part 2 — t_decision-safe feature engineering + interpretable classifier

Builds a classifier that predicts, at the moment a token is deployed, whether the bot will buy it — using **only information available strictly before that decision**.

- Negative class is subsampled (seed=42, N=80,000) from the much larger population of tokens the bot did not buy, to keep the dataset tractable
- All cumulative wallet-history features (`cum_n_events`, `cum_n_buys`, `cum_cost_usd_sum`, `cum_gas_usd_sum`, `cum_n_distinct_tokens`, `wallet_first_seen`, dev-buy signals) are attached via `pandas.merge_asof(..., direction="backward", allow_exact_matches=False)`. `allow_exact_matches=False` mechanically guarantees a wallet-history row timestamped at or after the deploy event can never leak into a feature — enforcing the t_decision-safe constraint by construction, not by convention
- Classifier: LightGBM (`class_weight="balanced"`), trained on a **time-based** 80/20 split (76,741 train / 19,186 test rows) — never a random split, to avoid look-ahead bias
- Test-set performance: PR-AUC 0.7362, precision 0.7290, recall 0.5826, F1 0.6476
- Interpretability via SHAP (`TreeExplainer`); top features are translated into plain-language rules in the notebook

## Part 3 — Replica strategy backtest

Simulates "buy whatever the model would buy" and compares it against the bot's real trades:

| | Bot actual (all test-period buys) | Replica strategy (model true positives) |
|---|---|---|
| n trades | 4,792 | 2,792 |
| Hit rate | 79.3% | 81.7% |
| Avg win (USD) | 99.71 | 116.17 |
| Avg loss (USD) | -17.80 | -19.30 |
| Total P&L (USD) | 361,338.9 | 255,253.3 |
| Max drawdown (USD) | -361.79 | -422.90 |

- Coverage: the replica strategy captures 58.3% of the bot's actual buy volume (recall)
- 1,038 of 3,830 replica-strategy buys are false positives relative to the bot's real behaviour — for these tokens the bot never traded, so ground-truth P&L is unknowable. This is an explicit, honest limitation of the backtest, not something papered over
- An entry-delay sensitivity check (proxying a 1–2 slot delayed entry using 1st vs 2nd buy price on multi-tranche tokens) found only 8 of 2,792 true-positive tokens had a usable second buy transaction — too small a sample to draw a firm conclusion, so the result (delayed entry appearing to hurt hit rate and P&L in this thin sample) is reported as a directional signal only, not a finding

## Repository contents

- `solana_sniper_bot_analysis.ipynb` — the full, current notebook (all three parts), as run on Kaggle
- `README.md` — this file

## Reproducing this notebook

This notebook is built to run inside a **Kaggle Notebook environment** attached to the competition, because:

- It reads directly from the competition's mounted input at `/kaggle/input/competitions/solana-sniper-bot-reverse-engineering/`
- It downloads a large (~39GB) supplementary dataset and a small (~106MB) wallet-activity dataset from external hosts at runtime (URLs are in the first cell) — these are not part of the Kaggle-mirrored competition data
- It relies on Kaggle's disk layout (`/kaggle/temp`, `/kaggle/working`) for intermediate and output files

To reproduce:

1. Open the notebook on Kaggle attached to the "Solana Sniper Bot Reverse-Engineering" competition (or import this `.ipynb` into a new Kaggle notebook and attach the competition as input).
2. Run all cells top to bottom. The first cell downloads and extracts the required data; later cells depend on it being present.
3. Expect the full run to take a meaningful amount of wall-clock time and RAM — Part 2's initial sampling step streams and filters roughly 25GB of raw activity data. A high-RAM Kaggle session is recommended.
4. Chart outputs (SHAP feature importance, P&L distribution, equity curve, entry-size/latency histograms) and the feature-importance CSV are written to `/kaggle/working/` as the notebook runs.

### Dependencies

Everything used is preinstalled in Kaggle's standard Python notebook image: `pandas`, `numpy`, `pyarrow`, `lightgbm`, `shap`, `scikit-learn`, `matplotlib`.

## Author

Amba Sharma ([@amba-create](https://github.com/amba-create))
