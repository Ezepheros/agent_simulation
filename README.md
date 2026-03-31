# Small Model Simulation

Run LLM agents powered by small local models (via LMStudio) with access to web fetching, Gmail, Google Calendar, and a Python sandbox. Every reasoning step, tool call, and result is logged in structured detail.

---

## Requirements

- Python 3.11+
- [LMStudio](https://lmstudio.ai) (free download, Windows/Mac/Linux)
- A Google account (for Gmail tool)
- 16GB+ RAM recommended (8GB minimum with a small model)

---

## 1. Install LMStudio

1. Download and install LMStudio from [lmstudio.ai](https://lmstudio.ai)
2. Open LMStudio → go to the **Discover** tab
3. Search for a model — recommended for 16GB RAM: **Qwen2.5-7B-Instruct** (Q4_K_M, ~4GB)
   - If you have 32GB RAM and a 16GB GPU: **Qwen/Qwen3.5-9B** works well
   - Avoid models larger than ~60% of your available RAM to leave room for inference
4. Click **Download** on the Q4_K_M variant
5. Go to the **Local Server** tab
6. Select your downloaded model from the dropdown
7. Configure before loading:
   - **Context Length**: set to `8192` (default 32k+ will use too much VRAM)
   - **GPU Offload**: drag to maximum (offloads model weights to VRAM)
   - **Flash Attention**: enable if available
   - **Keep model in memory**: leave ON
8. Click **Start Server** — note the URL shown (default: `http://127.0.0.1:1234`)
9. Note the exact model name shown in the dropdown (e.g. `qwen/qwen2.5-7b-instruct`)

---

## 2. Install the Python package

```bash
cd agents/small_model_simulation
pip install -e .
```

---

## 3. Configure the environment

```bash
cp .env.example .env
```

Edit `.env`:
```
LMSTUDIO_BASE_URL=http://localhost:1234/v1
GMAIL_CREDENTIALS_PATH=~/.config/small_agent/gmail_credentials.json
```

Open `configs/base.yaml` and update the model name to match what LMStudio shows:
```yaml
llm:
  model: qwen/qwen2.5-7b-instruct  # must match LMStudio exactly
```

---

## 4. Set up Google APIs (optional — needed for Gmail and Calendar tasks)

Gmail and Google Calendar both use OAuth2 credentials from the same Google Cloud project. Do the one-time project setup once, then enable whichever APIs you need.

### a) Create a Google Cloud project

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Click the project dropdown (top left) → **New Project** → name it anything → **Create**

### b) Enable APIs

In the left sidebar go to **APIs & Services → Library**, then enable whichever services you need:

| Tool | API to enable |
|---|---|
| Gmail (`email_*.yaml`) | **Gmail API** |
| Calendar (`calendar_*.yaml`) | **Google Calendar API** |

You can enable both now — it does not hurt to have both active.

### c) Configure the OAuth consent screen

This only needs to be done once, even if you enable both APIs.

1. **APIs & Services → OAuth consent screen**
2. User type: **External** → **Create**
3. Fill in:
   - App name: anything (e.g. `SmallAgent`)
   - User support email: your Gmail address
   - Developer contact: your Gmail address
4. Click **Save and Continue** through the Scopes page (no changes needed)
5. On the **Test users** page → **Add Users** → enter your Gmail address → **Save**
6. Click **Back to Dashboard**

### d) Create OAuth credentials

1. **APIs & Services → Credentials → Create Credentials → OAuth client ID**
2. Application type: **Desktop app** → name it anything → **Create**
3. Click **Download JSON** on the created credential
4. Move it to the config location:

```bash
mkdir -p ~/.config/small_agent
mv ~/Downloads/client_secret_*.json ~/.config/small_agent/gmail_credentials.json
```

The same credentials file is used by both Gmail and Calendar tools.

### e) First run authorisation

Each tool group (Gmail, Calendar) has its own token file and will open a browser window the first time it runs. After you approve, the token is saved and all future runs authenticate silently.

| Tool | Token file | Scope |
|---|---|---|
| Gmail | `~/.config/small_agent/gmail_token.json` | `gmail.readonly` |
| Calendar | `~/.config/small_agent/calendar_token.json` | `calendar` (read + write) |

> If you see **"access_denied"**: go back to the OAuth consent screen → Test users → confirm your exact Gmail address is listed.

---

## 5. Run an agent

```bash
# Web research — no API keys required, good first test
python scripts/run_agent.py --config configs/runs/web_research.yaml

# Email triage — what needs my attention from the last 2 weeks?
python scripts/run_agent.py --config configs/runs/email_triage.yaml

# Weekly digest — classify and summarise emails since the new year
python scripts/run_agent.py --config configs/runs/weekly_digest.yaml

# Calendar — create a simple event (low complexity, no search needed)
python scripts/run_agent.py --config configs/runs/calendar_low.yaml

# Calendar — find an event and reschedule it (moderate: search + modify)
python scripts/run_agent.py --config configs/runs/calendar_moderate.yaml

# Calendar — bulk create prep blocks across the week (high: multi-step reasoning)
python scripts/run_agent.py --config configs/runs/calendar_high.yaml

# Override the task from the command line
python scripts/run_agent.py --config configs/runs/web_research.yaml \
    --task "Summarise the latest news about open-source LLMs"

# Override the run ID (useful for keeping multiple result sets)
python scripts/run_agent.py --config configs/runs/calendar_moderate.yaml \
    --run-id calendar_moderate_002
```

Results are saved to `runs/{run_id}/`:
```
runs/calendar_moderate_001/
├── logs/
│   ├── calendar_moderate_001.log    ← human-readable step-by-step trace
│   └── calendar_moderate_001.jsonl  ← structured JSON Lines (one object per log line)
└── result.json                      ← final answer + full step record including reasoning
```

---

## 6. Create your own task

Create a new YAML file in `configs/runs/`, inheriting from the base config:

```yaml
base_config: ../../configs/base.yaml

run_id: my_task_001
description: "My custom task."
task: >
  Your task prompt here.

agent:
  max_steps: 8

tools:
  # Email — read-only Gmail access
  - type: tools.gmail.GmailTool
    max_results: 8
    max_body_chars: 400

  # Web — fetch and parse any URL
  - type: tools.web_fetch.WebFetchTool
    max_chars: 1500

  # Python — run code in a sandboxed subprocess
  - type: tools.python_sandbox.PythonSandboxTool
    timeout_s: 20.0

  # Calendar — list all calendars
  - type: tools.google_calendar.GetAllCalendarsTool

  # Calendar — search events by keyword or date range
  - type: tools.google_calendar.SearchCalendarEventsTool
    max_results: 10

  # Calendar — create a new event
  - type: tools.google_calendar.CreateCalendarEventTool

  # Calendar — update an existing event (requires event_id from search)
  - type: tools.google_calendar.ModifyCalendarEventTool

  # Calendar — permanently delete an event (requires event_id from search)
  - type: tools.google_calendar.RemoveCalendarEventTool
```

Only include the tools your task needs. The agent decides which ones to call and when.

---

## Performance tips

| Symptom | Fix |
|---|---|
| System freezes during inference | Reduce LMStudio context length to 8192 |
| VRAM full / slow generation | Increase GPU offload layers to maximum |
| Run stalls halfway | Reduce `max_body_chars` and `max_chars` in your config |
| Agent loops without finishing | Reduce `max_results` (fewer emails = shorter context) |
| Want to see reasoning | Set `logging.level: DEBUG` in your run config |

---

## Project structure

```
small_agent/
├── core/types.py          ← domain objects (Message, ToolCall, AgentStep, …)
├── tools/
│   ├── web_fetch.py       ← fetch and parse any URL
│   ├── gmail.py           ← read Gmail via OAuth2 (read-only)
│   ├── google_calendar.py ← read/write Google Calendar via OAuth2
│   └── python_sandbox.py  ← run Python code in a sandboxed subprocess
├── backends/              ← LMStudio (OpenAI-compatible) backend
├── agents/react.py        ← ReAct loop (Reason + Act)
├── pipeline/              ← orchestrates a full run from config to saved output
├── config/schemas.py      ← Pydantic v2 config models with YAML inheritance
├── logging.py             ← structured logging (run-scoped, JSON + human-readable)
└── registry.py            ← dynamic component factory (type path → instance)
```
