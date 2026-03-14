# Coinbase Crypto Prob Scanner

FastAPI scanner for **Coinbase-tradeable spot crypto** with a single target:

- **pt2 / prob_2**: probability an asset reaches **+2%** within the next **120 minutes**, based on future 5-minute highs.

## What this build does
- 24/7 crypto scanner (no market-open/close assumptions).
- Two-stage pipeline:
  - Stage 1 candidate generator (high recall + liquidity/quality filters).
  - Stage 2 calibrated elastic-net logistic model (`MODEL_DIR/pt2`).
- Guardrails with `risk`, `risk_reasons`, `downside_risk`, `uncertainty`, and caps for event/downside/OOD states.
- Demo mode works without keys.

## Deploy on Render (single web service)
1. Create Web Service from this repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
4. Attach persistent disk and set `MODEL_DIR=/var/data/model`.

## Required env vars
- `MODEL_DIR` (default `./runtime/model`)
- `DEMO_MODE` (`true/false`)
- `DISABLE_SCHEDULER` (`true/false`)
- `SCAN_INTERVAL_MINUTES` (default `5`)
- `ADMIN_PASSWORD` (required for `/train`)
- `TRAIN_LOOKBACK_DAYS` (default `60`)
- `TRAIN_MAX_SYMBOLS` (default `0`, all)

## Universe/config env vars
- `COINBASE_API_BASE` (default `https://api.exchange.coinbase.com`)
- `COINBASE_MAX_PRODUCTS` (default `400`)
- `COINBASE_FETCH_WORKERS` (default `12`, concurrent symbol fetch workers)
- `COINBASE_QUOTE_CURRENCIES` (default `USD,USDC`)
- `UNIVERSE_MODE` (`all`, `top_n`, `all_plus_top_n`; default `top_n`)
- `UNIVERSE_TOP_N` (default `120`)
- `MIN_LISTING_DAYS` (default `30`)
- `MIN_ROLLING_DOLLAR_VOLUME` (default `750000`)
- `MAX_WICKINESS` (default `0.75`)
- `UNIVERSE_EXCLUDE_SYMBOLS` (CSV)

## Target env vars
- `TARGET_HORIZON_MINUTES=120`
- `TARGET_MOVE_PCT=0.02`

## Train
- In UI, enter `ADMIN_PASSWORD` and click **Start training**.
- Training builds **only `pt2`** artifacts into `MODEL_DIR/pt2/bundle.joblib`.

## API
- `GET /`
- `GET /health`
- `GET /api/status`
- `GET /api/scores`
- `POST /train`
- `GET /api/training/status`
- `GET /api/debug/coverage`

## Interpretation
- `prob_2`: model-estimated probability of hitting +2% in 120m.
- `risk`: qualitative guardrail bucket.
- `downside_risk`: adverse-move proxy (0 to 1).
- `uncertainty`: LOW/MED/HIGH OOD signal.

