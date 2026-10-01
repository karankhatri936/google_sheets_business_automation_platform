# Google Sheets Business Automation Platform

> Modular Python platform that automates multi-stage data ingestion, validation, cleaning, business KPI computation, LLM-powered narrative interpretation, customer feedback categorization, and formatted multi-tab Google Sheets reporting.

---

## Highlights

- **Complete 7-Layer Architecture**:
  1. **Config & Logging**: Typed dataclass configuration with `.env` loading, validation, and structured console/file logging.
  2. **Google Sheets Integration**: Service Account + OAuth 2.0 flows, backoff retry for 429/500/503 errors, cell formatting, number format masking, auto-column resizing, and batch updating.
  3. **Data Quality Engine**: Schema validation, typed coercion, duplicate detection, category validation, outlier flagging, revenue recalculation, and comprehensive rejected-rows audit log.
  4. **Analytics & KPI Engine**: Summary metrics (gross/net revenue, order count, AOV, return rate), product performance, category aggregation, regional distribution, monthly trends, and period-over-period percentage comparisons.
  5. **AI Interpretation & Classification**:
     - Executive summary, positives, watch items, and forward recommendations.
     - Anti-hallucination verification comparing narrative numbers against deterministic KPI facts.
     - Few-shot sentiment & topic classification for free-text customer reviews.
     - Providers: `mock` (deterministic, zero-cost, offline) and `openai_compatible` (via stdlib `urllib.request`).
  6. **Multi-Sheet Report Builder & Sinks**: Emits 8 business worksheets (`Clean_Data`, `Data_Quality`, `KPI_Summary`, `Product_Analysis`, `Category_Analysis`, `Regional_Analysis`, `AI_Insights`, `Run_Log`) with professional navy/slate styling, currency masks, borders, and column autosizing. Supports live Google Sheets and offline CSV/JSON dry-run sinks.
  7. **Pipeline Orchestrator & Scheduler**: Single-command execution, graceful failure containment, APScheduler cron/interval daemon, and run summary exports.

- **Zero-Dependency Core AI**: LLM client uses Python stdlib (`urllib.request`), keeping external dependencies strictly scoped to Google APIs, APScheduler, and pandas.
- **100% Offline Capable**: Built-in synthetic demo data generator and mock AI provider allow complete end-to-end execution and testing without external credentials or internet connectivity.
- **Test Suite**: 189 unit and integration tests, fully offline and deterministic (no network, no real credentials required).


---

## Quickstart

### 1. Requirements & Setup

- Python 3.11+
- Virtual environment:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Offline Demo (Zero Google Credentials Required)

Test the complete pipeline locally using synthetic e-commerce data and deterministic mock AI:

```bash
# PowerShell:
$env:AI_PROVIDER="mock"; python main.py --dry-run --demo

# macOS/Linux:
AI_PROVIDER=mock python main.py --dry-run --demo
```

Generated reports are written to `./output/`:
- `output/Clean_Data.csv`
- `output/Data_Quality.csv`
- `output/KPI_Summary.csv`
- `output/Product_Analysis.csv`
- `output/Category_Analysis.csv`
- `output/Regional_Analysis.csv`
- `output/AI_Insights.csv`
- `output/Run_Log.csv`
- `output/run_summary.json`

---

## Google Sheets Setup (Live Mode)

### 1. Service Account (Recommended for automated/cron jobs)
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **Google Sheets API** and **Google Drive API**.
3. Create a Service Account and download the JSON key to `credentials/service_account.json`.
4. Create a Google Spreadsheet and share it with the service account email (with **Editor** permissions).
5. Set `GS_SPREADSHEET_ID=<YOUR_SPREADSHEET_ID>` in `.env`.

### 2. OAuth 2.0 (For user-delegated access)
1. In Google Cloud Console, configure OAuth Consent Screen and create **OAuth client ID** (Desktop Application).
2. Download the client JSON to `credentials/client_secret.json`.
3. Set `GS_AUTH_MODE=oauth` in `.env`.
4. On first run, a browser window opens requesting consent; token is cached to `credentials/token.json`.

---

## OpenRouter (OpenAI-compatible provider)

OpenRouter exposes the OpenAI chat-completions contract, so it needs **no separate provider class** - only configuration:

```bash
AI_ENABLED=true
AI_PROVIDER=openai_compatible
AI_BASE_URL=https://openrouter.ai/api/v1
AI_MODEL=qwen/qwen3.8-27b:free          # always <vendor>/<model>[:variant]
OPENAI_API_KEY=<your OpenRouter key>    # variable name kept for compatibility
AI_MAX_OUTPUT_TOKENS=3000               # see the reasoning note below
AI_REASONING_EFFORT=minimal             # least thinking every endpoint accepts
AI_HTTP_REFERER=                        # optional attribution
AI_APP_TITLE=Google Sheets Business Automation
```

* The URL always resolves to exactly `https://openrouter.ai/api/v1/chat/completions`: trailing slashes are tolerated and a base URL that already ends in `/chat/completions` is normalised, so the path is never doubled.
* `AI_MODEL` must carry the OpenRouter vendor prefix. A slug like `qwen3.8-27b:free` is rejected at startup with a clear configuration error instead of failing later at the API.
* **Reasoning models**: OpenRouter reports `reasoning.default_enabled: true` (default effort `xhigh`) for models such as `qwen/qwen3.8-27b:free`, and routers like `openrouter/free` may route to such a model. They spend output tokens on hidden reasoning *first*, so this project defaults to `AI_REASONING_EFFORT=minimal` (the lowest level every endpoint accepts; the KPI engine is authoritative and the model only verbalises verified numbers) and a `AI_MAX_OUTPUT_TOKENS=3000` output budget. Some endpoints reject `effort: "none"` (`Reasoning is mandatory for this endpoint`) — when a provider rejects the reasoning field the request is retried once without it. If a reply still arrives empty with `finish_reason="length"`, the error states the finish reason, the reasoning-token usage and both settings to change. Reasoning parameters are only sent to OpenRouter hosts (or when `AI_SEND_REASONING_PARAMS=true`), so non-reasoning models and other gateways are unaffected.
* Different routes may return `message.content` as a string or as a list of text parts; both are handled, while genuinely empty or malformed replies fail with a diagnostic message instead of a vague "empty message".

Check the resolved endpoint without spending a request:

```bash
python -c "from src.ai.provider import OpenAICompatibleProvider; from src.config import load_settings; print(OpenAICompatibleProvider(load_settings().ai).endpoint)"
```

* **Latency**: free routers such as `openrouter/free` queue requests and can take minutes per call. For scheduled runs, pin a concrete slug (for example `AI_MODEL=qwen/qwen3.8-27b:free`) for predictable latency and stable output quality; `AI_MODEL` remains fully configurable.

---

## Configuration Reference

Key variables available in `.env` (see `.env.example` for the complete list):

| Variable | Default | Description |
|---|---|---|
| `GS_SPREADSHEET_ID` | `""` | Target spreadsheet ID (empty triggers dry-run) |
| `GS_AUTH_MODE` | `service_account` | `service_account` or `oauth` |
| `GS_SERVICE_ACCOUNT_FILE` | `credentials/service_account.json` | Path to service account key |
| `GS_RAW_DATA_RANGE` | `Raw_Data!A1:Z10000` | Input range for source data |
| `GS_RUN_LOG_MAX_ROWS` | `500` | Max history rows in Run_Log |
| `AI_PROVIDER` | `openai_compatible` | `openai_compatible` or `mock` (offline) |
| `AI_ENABLED` | `true` | Enable/disable LLM features |
| `AI_BASE_URL` | `https://api.openai.com/v1` | Base URL; `/chat/completions` is appended (OpenRouter: `https://openrouter.ai/api/v1`) |
| `AI_MODEL` | `gpt-4o-mini` | Model id; OpenRouter needs `<vendor>/<model>[:variant]`, e.g. `qwen/qwen3.8-27b:free` |
| `AI_API_KEY_ENV` | `OPENAI_API_KEY` | Name of the variable holding the key |
| `OPENAI_API_KEY` | `""` | The secret itself (never committed) |
| `AI_MAX_OUTPUT_TOKENS` | `3000` | Output budget per request (reasoning models spend part of it on thinking) |
| `AI_REASONING_EFFORT` | `minimal` | `minimal`/`low`/`medium`/`high`/`xhigh`/`none`, or `auto` to omit the field |
| `AI_REASONING_MAX_TOKENS` | unset | Optional explicit thinking budget (`reasoning.max_tokens`) |
| `AI_SEND_REASONING_PARAMS` | auto | Force (`true`) or suppress (`false`) reasoning fields; auto = OpenRouter hosts only |
| `AI_HTTP_REFERER` / `AI_APP_TITLE` | unset | Optional OpenRouter attribution headers |
| `SCHEDULE_MODE` | `daily` | `daily`, `weekly` or `interval` |
| `SCHEDULE_TIME` | `07:30` | Time of day for daily/weekly runs |
| `SCHEDULE_INTERVAL_MINUTES` | `60` | Interval mode period |

---

## Scheduled Execution & Testing

Run as a persistent daemon with APScheduler:
```bash
python main.py --schedule
```

Run full test suite (189 tests across all 7 layers):
```bash
pytest
```

---

## Security & secrets

* `.env`, `credentials/` (service-account key, OAuth client secret, cached token), `logs/` and `output/` are listed in `.gitignore` and must never be committed.
* Only `.env.example` (placeholders, no real values) belongs in version control. Copy it to `.env` locally and fill in real values.
* Verify before/after any push:

```bash
git ls-files | grep -E "(^\.env$|credentials/)"   # must print nothing
```

* If a credential is ever exposed (committed, pasted, or shared), treat it as compromised: rotate/revoke it in the provider console (Google Cloud for service-account keys, OpenRouter for API keys) and replace it in `.env`. Deleting the file in a later commit is not sufficient because it remains in history.
* The AI layer never logs API keys: outgoing `Authorization` headers are redacted, and provider errors are stripped of credential-shaped strings before they are raised or logged.

