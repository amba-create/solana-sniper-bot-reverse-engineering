# Solana Sniper Bot Reverse-Engineering

Entry for the Kaggle competition **Solana Sniper Bot Reverse-Engineering**. The goal is to reconstruct the deployer-selection strategy of an anonymous, high-performing Solana memecoin sniper bot from raw on-chain deploy and wallet-activity data, then build a replica strategy and evaluate whether it could match the bot's real performance.

The full pipeline lives in [`solana_sniper_bot_analysis.ipynb`](./solana_sniper_bot_analysis.ipynb) and is organised into three parts. This README is a technical summary; the full narrative, including the hypothesized deployer-selection rules, methodology justification, and stated limitations, is in the [Kaggle Writeup](https://www.kaggle.com/competitions/solana-sniper-bot-reverse-engineering/writeups).

## Part 1 — Behavioural analysis

Descriptive analysis of the bot's actual trading behaviour, using its real buy/sell/burn transactions:

- 16,192 buys, 70,812 sells, 2 burns across 16,168 unique tokens
- Entry size (USD): mean 263.25, median 183.72, std 232.51
- Median hold time: 14 slots (a Solana slot targets ~400ms under normal network conditions [1])
- Hit rate on realized trades: 78.74%, avg win 122.63 USD, avg loss -21.09 USD (~6:1 win/loss ratio)
- The large majority of exited positions use multi-tranche (staged) selling rather than a single exit

## Part 2 — t_decision-safe feature engineering + interpretable classifier

Builds a classifier that predicts, at the moment a token is deployed, whether the bot will buy it — using **only information available strictly before that decision**.

- Negative class is subsampled (seed=42, N=80,000) from the much larger population of tokens the bot did not buy, to keep the dataset tractable. This is a documented methodological choice: a different sample size would shift precision/recall somewhat even with the same underlying signal
- All cumulative wallet-history features (`cum_n_events`, `cum_n_buys`, `cum_cost_usd_sum`, `cum_gas_usd_sum`, `cum_n_distinct_tokens`, `wallet_first_seen`, dev-buy signals) are attached via `pandas.merge_asof(..., direction="backward", allow_exact_matches=False)`. `allow_exact_matches=False` mechanically guarantees a wallet-history row timestamped at or after the deploy event can never leak into a feature — enforcing the t_decision-safe constraint by construction, not by convention
- Classifier: LightGBM (`class_weight="balanced"`) [2], chosen over alternatives such as logistic regression because wallet-history features (fee spend, wallet age) are heavily skewed, logistic regression assumes a roughly linear relationship in log-odds space [3] and would need those features transformed, whereas LightGBM splits on thresholds directly and captures feature interactions (e.g. wallet age combined with dev-buy behaviour) automatically. It also integrates directly with SHAP for interpretability. No empirical head-to-head comparison against logistic regression was run; this is a structural justification, not a tested result
- Trained on a **time-based** 80/20 split (76,741 train / 19,186 test rows) — never a random split, to avoid look-ahead bias. The 80/20 ratio is a standard default; the deliberate choice is the chronological ordering, which prevents the model training on events that occurred after its test events
- Test-set performance: PR-AUC 0.7362, precision 0.7290, recall 0.5826, F1 0.6476
- **Population-reweighted evaluation**: the test set's negative class is subsampled to 80,000 events (~25% positive rate), not the true population rate. Recall and false-positive rate are invariant to this subsampling, so both can be reweighted to the true population base rate (0.314%, across all 5,076,421 deploy events) without rescoring every candidate. At that base rate: precision falls to **2.48%** (recall unchanged at 58.3%, F1 4.76%). Pointed at every deploy event on Solana rather than this curated set, roughly 1 in 40 flagged tokens would be a genuine match, not roughly 3 in 4 — the expected effect of subsampling, reported so balanced-set precision is not mistaken for real-world precision
- Interpretability via SHAP (`TreeExplainer`) [4], based on the Shapley value from cooperative game theory [5]; top features are translated into plain-language rules in the notebook
- **Leakage audit**: since positives are sourced from the Kaggle-mirrored data and negatives from an externally downloaded file, the two could in principle use inconsistent timestamp encoding (a bug found in at least one other team's early attempt at this problem). Comparing raw schemas, timestamp ranges, and units between the two sources found no mismatch — both use the same encoding and cover the same period

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
- **Executability stress test**: the figures above ignore transaction costs. The bot's own recorded fees show why this matters — winning the zero-block race requires a competitive priority fee, so median buy-side gas is $7.62, versus $0.08 median on (non-time-competitive) sells. Applying this $7.71 round-trip cost to every replica-strategy trade drops hit rate from 81.7% to **67.7%** (a meaningful share of recorded wins are thin margins a realistic fee erases), while total P&L is more resilient, falling from $255,253 to **$233,734** (~8%). Aggregate profitability survives the stress test; the headline hit rate does not fully reflect how fragile individual trades are
- Next step for the remaining open items above: independently source Solana price data for the false-positive tokens and for the full true-positive set, rather than relying on the bot's own second buy as a proxy. The competition rules explicitly permit querying public Solana on-chain data as External Data, provided it is not used to alter the decision-time features in Parts 2–3 [6]

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

## Sources

1. Solana. "Terminology." [solana.com/docs/references/terminology](https://solana.com/docs/references/terminology)
2. Ke, G., et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS, 2017. [papers.nips.cc/paper/6907](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer.
4. Lundberg, S. M., and Lee, S. "A Unified Approach to Interpreting Model Predictions." NeurIPS, 2017. [arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)
5. Shapley, L. S. "A Value for n-Person Games." In *Contributions to the Theory of Games II*, Princeton University Press, 1953, pp. 307–317.
6. Kaggle. "Solana Sniper Bot Reverse-Engineering: Competition Rules," Section 2.6, External Data and Tools. [kaggle.com/competitions/solana-sniper-bot-reverse-engineering/rules](https://www.kaggle.com/competitions/solana-sniper-bot-reverse-engineering/rules)

## Author

Amba Sharma ([@amba-create](https://github.com/amba-create))
