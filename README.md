# Small Model Simulation

Run LLM agents powered by small local models (via LMStudio) with access to web fetching, Gmail, and a Python sandbox. Every reasoning step, tool call, and result is logged in structured detail.

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

## 4. Set up Gmail (optional — only needed for email tasks)

### a) Create a Google Cloud project

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Click the project dropdown (top left) → **New Project** → name it anything → **Create**
3. In the left sidebar: **APIs & Services → Enable APIs and Services**
4. Search for **Gmail API** → click it → **Enable**

### b) Configure the OAuth consent screen

1. **APIs & Services → OAuth consent screen**
2. User type: **External** → **Create**
3. Fill in:
   - App name: anything (e.g. `SmallAgent`)
   - User support email: your Gmail address
   - Developer contact: your Gmail address
4. Click **Save and Continue** through the Scopes page (no changes needed)
5. On the **Test users** page → **Add Users** → enter your Gmail address → **Save**
6. Click **Back to Dashboard**

### c) Create OAuth credentials

1. **APIs & Services → Credentials → Create Credentials → OAuth client ID**
2. Application type: **Desktop app** → name it anything → **Create**
3. Click **Download JSON** on the created credential
4. Move it to the config location:

```bash
mkdir -p ~/.config/small_agent
mv ~/Downloads/client_secret_*.json ~/.config/small_agent/gmail_credentials.json
```

### d) First run authorisation

The first time a Gmail tool is used, a browser window will open asking you to sign in and grant read-only access. After approving, a token is saved to `~/.config/small_agent/gmail_token.json` and all future runs authenticate silently.

> If you see **"access_denied"**: go back to the OAuth consent screen → Test users → confirm your exact Gmail address is listed.

---

## 5. Run an agent

```bash
# Email triage — what needs my attention from the last 2 weeks?
python scripts/run_agent.py --config configs/runs/email_triage.yaml

# Weekly digest — classify and summarise emails since the new year
python scripts/run_agent.py --config configs/runs/weekly_digest.yaml

# Web research task
python scripts/run_agent.py --config configs/runs/web_research.yaml

# Override the task from the command line
python scripts/run_agent.py --config configs/runs/web_research.yaml \
    --task "Summarise the latest news about open-source LLMs"
```

Results are saved to `runs/{run_id}/`:
```
runs/weekly_digest_001/
├── logs/
│   ├── weekly_digest_001.log    ← human-readable step-by-step trace
│   └── weekly_digest_001.jsonl  ← structured JSON Lines (one object per log line)
└── result.json                  ← final answer + full step record including reasoning
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
  - type: tools.gmail.GmailTool
    max_results: 8
    max_body_chars: 400

  - type: tools.web_fetch.WebFetchTool
    max_chars: 1500

  - type: tools.python_sandbox.PythonSandboxTool
    timeout_s: 20.0
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
├── tools/                 ← WebFetch, Gmail, PythonSandbox
├── backends/              ← LMStudio (OpenAI-compatible) backend
├── agents/react.py        ← ReAct loop (Reason + Act)
├── pipeline/              ← orchestrates a full run from config to saved output
├── config/schemas.py      ← Pydantic v2 config models with YAML inheritance
├── logging.py             ← structured logging (run-scoped, JSON + human-readable)
└── registry.py            ← dynamic component factory (type path → instance)
```
