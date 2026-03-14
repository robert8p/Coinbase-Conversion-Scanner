# S&P 500 Prob Scanner — v9.6.1

A production-ready FastAPI scanner for one target only:
- `prob_1`: probability the stock reaches at least **+1%** from scan time before the close

## Core target definition
For each stock at scan time:
- `P0` = stock price at scan time
- `H_future` = maximum future **1-minute HIGH** from scan time until the regular-session close
- `Y=1` if `H_future >= 1.01 * P0`, else `0`

Displayed `prob_1` is a calibrated 0–1 probability, not a rank.

## What changed in v9.6

### v9.6.1 operational fix
This patch fixes a live market-data coverage issue in Alpaca multi-symbol bar fetching.

Problem that was fixed:
- the client could request too many symbols in one chunk
- leave Alpaca at its default page size
- then stop after a low pagination cap
- which could silently drop the tail end of the universe and produce `pagination cap hit` / inflated `no_bars`

What v9.6.1 changes:
- forces a high per-page bar limit (`10000`) for multi-symbol requests
- raises the internal pagination ceiling materially
- recursively splits oversized symbol batches instead of truncating coverage

This is specifically intended to restore near-full live coverage before training or interpreting signals.

### 1) Single-target architecture
v9.6 removes the entire 2% path:
- no `pt2`
- no `prob_2`
- no 2% UI or endpoint fields
- storage is `MODEL_DIR/pt1`

### 2) Two-stage design
v9.6 is explicitly designed to increase the number of **trustworthy** high-confidence names.

#### Stage 1: candidate generator
A fast, deterministic filter scores the full universe using:
- liquidity / dollar-volume sufficiency
- ToD RVOL
- intraday momentum and relative strength
- trend and VWAP state
- daily-structure context
- path quality and opening-range state
- structural falling-knife blockers
- event-risk suppression
- minimum time-to-close guardrail

Stage 1 is intentionally higher-recall than the final model. It narrows the universe to a cleaner set of plausible +1% candidates.

#### Stage 2: confidence model
Only Stage 1 candidates are passed into the trained model:
- elastic-net logistic regression
- time-aware training / validation / holdout
- segmented calibration by time-to-close bucket and risk bucket
- ToD volume normalization
- strict persistence of bundle + metadata

This matters because the model is now asked a narrower question on a cleaner population, which gives it a better chance of issuing legitimate `prob_1 > 0.75` outputs without simply inflating probabilities.

### 3) Stronger protection against ugly long setups
The v9.6 guardrail stack is designed to suppress FICO-style failures:
- long-term downtrend / drawdown protection
- open weakness + no-reclaim logic
- intraday damage from session high
- event-risk proxy blocking on extreme gap / RVOL / range combinations
- downside caps
- uncertainty / out-of-distribution caps

## UI and endpoint outputs
Main table columns:
- Symbol
- Price
- VWAP
- Prob 1%
- Risk
- Risk reason
- Downside
- Uncertainty
- Sector
- Reasons

Endpoints:
- `GET /`
- `GET /health`
- `GET /api/status`
- `GET /api/scores`
- `POST /train`
- `GET /api/training/status`
- `GET /api/debug/coverage?password=...`

## How to interpret the outputs

### `prob_1`
The estimated probability that the stock touches **+1% from scan time** before today’s close.

### `risk`
A structural / microstructure classification:
- `OK`: no major live guardrail issue detected
- `HIGH`: tradability / stability concerns exist
- `BLOCKED`: setup looks structurally dangerous or falling-knife-like

In v9.6, `BLOCKED` and event-risk names are normally filtered out at Stage 1 before Stage 2 scoring.

### `downside_risk`
A 0–1 score estimating asymmetric long downside risk. Higher means more danger that the setup is weak, damaged, or structurally poor. High downside can cap probability even if the raw model score is strong.

### `uncertainty`
The model’s out-of-distribution warning:
- `LOW`
- `MED`
- `HIGH`

This is based on distance from the training distribution across key features. High uncertainty can cap probability.

## Training outputs
Training builds only `pt1` and volume profiles.

Reported metrics include:
- holdout AUC
- holdout Brier
- holdout precision at `0.60 / 0.70 / 0.75 / 0.80`
- acceptable-only precision at the same thresholds
- counts at those thresholds
- challenge-set diagnostics for structurally dangerous names
- path diagnostics for `p >= 0.75` acceptable setups
- holdout feature-group ablation diagnostics

## Environment variables

### Required for live mode
- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `ALPACA_DATA_FEED=sip`
- `TIMEZONE=America/New_York`

### Core scanner
- `SCAN_INTERVAL_MINUTES` default `5`
- `MIN_BARS_5M` default `7`
- `MODEL_DIR` default `./runtime/model`
- `DEMO_MODE` default `false`
- `DISABLE_SCHEDULER` default `false`
- `ADMIN_PASSWORD`
- `DEBUG_PASSWORD` optional; falls back to `ADMIN_PASSWORD`

### Training
- `TRAIN_LOOKBACK_DAYS` default `60`
- `TRAIN_MAX_SYMBOLS` default `0` (= all)
- `CALIB_MIN_BUCKET_SAMPLES` default `200`
- `ENET_C_VALUES` default `0.25,0.5,1.0,2.0`
- `ENET_L1_VALUES` default `0.0,0.25,0.5,0.75`
- `PRIOR_ALPHA_VALUES` default `0.35,0.45,0.55,0.65,0.75,0.85`
- `SELECTION_MIN_COUNT_75` default `8`

### ToD RVOL
- `TOD_RVOL_LOOKBACK_DAYS` default `20`
- `TOD_RVOL_MIN_DAYS` default `8`

### Liquidity / tradability guardrails
- `LIQ_ROLLING_BARS`
- `LIQ_DVOL_MIN_USD`
- `LIQ_RANGE_PCT_MAX`
- `LIQ_WICK_ATR_MAX`

### Structural blockers
- `BLOCKED_RET20D_MAX`
- `BLOCKED_RET60D_MAX`
- `BLOCKED_DIST50DMA_MAX`
- `BLOCKED_RET_SINCE_OPEN_MAX`
- `BLOCKED_DAMAGE_FROM_HIGH_ATR_MIN`
- `BLOCKED_BELOW_VWAP_FRAC_MIN`
- `BLOCKED_PROB_CAP`

### Event / uncertainty / downside caps
- `EVENT_GAP_ABS_MIN`
- `EVENT_RVOL_MIN`
- `EVENT_RANGE_PCT_MIN`
- `EVENT_PROB_CAP`
- `UNCERTAINTY_Z_THRESH`
- `UNCERTAINTY_EXTREME_FEATURES_MIN`
- `UNCERTAINTY_PROB_CAP`
- `DOWNSIDE_HIGH_THRESHOLD`
- `DOWNSIDE_MEDIUM_THRESHOLD`
- `DOWNSIDE_PROB_CAP_HIGH`
- `DOWNSIDE_PROB_CAP_MEDIUM`

### Stage 1 controls
- `STAGE1_CANDIDATE_CAP` default `120`
- `STAGE1_MIN_SCORE` default `2.0`
- `STAGE1_MIN_MINUTES_TO_CLOSE` default `35`
- `STAGE1_MIN_RVOL` default `0.85`
- `STAGE1_MIN_DOLLAR_VOLUME_MULT` default `0.7`

## Render deployment
1. Create a new Render **Web Service**.
2. Choose **Dockerfile** build.
3. Enable a **Persistent Disk** and mount it at `/var/data`.
4. Set at minimum:
   - `MODEL_DIR=/var/data/model`
   - `ALPACA_DATA_FEED=sip`
   - `TIMEZONE=America/New_York`
   - `ADMIN_PASSWORD=<your password>`
   - `DEMO_MODE=true` for a no-keys smoke test, or live Alpaca keys for live mode
5. Health check path: `/health`
6. Keep a single worker. The Dockerfile starts uvicorn with `--workers 1`.

## How to train after deployment
1. Open the dashboard.
2. Enter `ADMIN_PASSWORD` in the Training panel.
3. Click **Start**.
4. Poll `/api/training/status` or watch the dashboard until training completes.
5. After training, `/api/status` will show the pt1 bundle as trained.

## Operational notes
- Regular-session only.
- SIP is enforced.
- The app still loads if training has never run.
- Demo mode works with no Alpaca keys.
- If you upgrade from older model bundles, retrain. The v9.6 feature schema and bundle metadata changed.
