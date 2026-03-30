# Small Model Simulation — Project Plan

## Goal

Evaluate LLM agents powered by **small local models** (hosted via LMStudio) on real-world tasks. The agent has access to:
- **WebFetch** — fetch and parse web pages
- **Gmail** — read emails via the Gmail API
- **Python Sandbox** — execute code in an isolated environment

Every step, thought, and tool interaction is logged in structured detail.

---

## Architecture Overview

The design follows a **strategy + registry pattern** inspired by `benchmark_legal_rag`, adapted for agent-loop execution rather than retrieval pipelines.

```
Task (string prompt)
    ↓
AgentRunPipeline          ← orchestrates the entire run
    ├── build LLM backend (LMStudio)
    ├── build tools (WebFetch, Gmail, PythonSandbox)
    └── build agent (ReActAgent)
         ↓
         AgentLoop:
           while not done and steps < max_steps:
             1. LLM generates thought + tool call(s)
             2. Tool(s) execute → ToolResult(s)
             3. Results appended to context
             4. Log AgentStep (thought, calls, results)
         ↓
AgentRun (complete run record with all steps, final answer, metrics)
    ↓
runs/{run_id}/
    ├── run.jsonl          ← structured step-by-step log
    ├── run.log            ← human-readable log
    └── result.json        ← final answer + summary metrics
```

---

## Directory Structure

```
small_model_simulation/
├── small_agent/                   # Main Python package
│   ├── core/
│   │   └── types.py               # All domain dataclasses (Message, ToolCall, AgentStep, …)
│   ├── tools/
│   │   ├── base.py                # BaseTool ABC + ToolSchema dataclass
│   │   ├── web_fetch.py           # WebFetch tool
│   │   ├── gmail.py               # Gmail read tool
│   │   └── python_sandbox.py      # Subprocess-isolated Python execution
│   ├── backends/
│   │   ├── base.py                # BaseLLMBackend ABC
│   │   └── lmstudio.py            # OpenAI-compat client wrapping LMStudio
│   ├── agents/
│   │   ├── base.py                # BaseAgent ABC
│   │   └── react.py               # ReAct (Reason + Act) loop agent
│   ├── pipeline/
│   │   └── agent_pipeline.py      # AgentRunPipeline orchestrator
│   ├── config/
│   │   └── schemas.py             # Pydantic v2 config models
│   ├── logging.py                 # Structured logging (run-scoped, JSON + human)
│   └── registry.py                # Dynamic component factory
│
├── configs/
│   ├── base.yaml                  # Shared defaults
│   └── runs/
│       ├── web_research.yaml      # Example: research task using WebFetch
│       ├── email_summary.yaml     # Example: summarise Gmail inbox
│       └── coding_task.yaml       # Example: write and run a Python script
│
├── scripts/
│   └── run_agent.py               # CLI entry point
│
├── runs/                          # Generated outputs (gitignored)
│
├── pyproject.toml
├── .env.example
└── PLAN.md                        ← this file
```

---

## Domain Objects (`core/types.py`)

| Class | Purpose |
|---|---|
| `Message` | A single LLM conversation turn (role, content, metadata) |
| `ToolSchema` | JSON-Schema description of a tool (name, description, parameters) |
| `ToolCall` | LLM-requested tool invocation (tool name, arguments dict, call_id) |
| `ToolResult` | Result of executing a tool (call_id, output, error, latency_ms) |
| `AgentStep` | One full reasoning step (step #, thought text, tool calls, results) |
| `AgentRun` | Complete run record (run_id, task, steps list, final answer, metrics) |
| `LLMResponse` | Raw response from the backend (content, tool calls, usage stats) |

---

## Key Abstractions

### `BaseTool` (ABC)
```python
class BaseTool(ABC):
    name: str
    description: str

    @property
    @abstractmethod
    def schema(self) -> ToolSchema: ...

    @abstractmethod
    def execute(self, arguments: dict) -> ToolResult: ...
```

### `BaseLLMBackend` (ABC)
```python
class BaseLLMBackend(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> LLMResponse: ...
```

### `BaseAgent` (ABC)
```python
class BaseAgent(ABC):
    @abstractmethod
    def run(self, task: str) -> AgentRun: ...
```

### `AgentRunPipeline`
Thin orchestrator that reads a `RunConfig`, builds components via the registry, and delegates execution to the agent. Saves all artefacts under `runs/{run_id}/`.

---

## Configuration (`config/schemas.py`)

```
RunConfig
├── run_id: str
├── task: str
├── agent: AgentConfig
│   └── type, max_steps, system_prompt
├── llm: LLMConfig
│   └── base_url, model, temperature, max_tokens, timeout_s
├── tools: list[ToolConfig]
│   └── type, name, …tool-specific params
└── logging: LoggingConfig
    └── level, log_dir, log_steps
```

YAML inheritance works identically to `benchmark_legal_rag`: a `base_config` key deep-merges a parent YAML, and only overridden keys need to appear in child configs.

---

## Logging Strategy (`logging.py`)

- `setup_run_logging(run_id, log_dir)` — configures handlers once
- `get_logger(__name__)` — returns a `RunAdapter` that injects `[run_id]` into every message
- Two sinks: `run.log` (human-readable) and `run.jsonl` (structured JSON Lines)
- Every `AgentStep` is logged as a structured JSON record with fields: `step`, `thought`, `tool_calls`, `tool_results`, `elapsed_ms`
- Final answer + metrics logged at INFO level in both sinks

---

## Tool Details

### WebFetch
- Input: `url` (string)
- Fetches HTML, strips tags, truncates to `max_chars` (configurable)
- Returns cleaned text + page title + final URL (after redirects)

### Gmail
- Input: `query` (Gmail search string), `max_results` (int)
- Uses OAuth2 via `google-auth` + `google-api-python-client`
- Returns list of email summaries: subject, sender, date, snippet, body
- Credentials path configurable; token cached in `~/.config/small_agent/gmail_token.json`

### PythonSandbox
- Input: `code` (string), optional `timeout_s`
- Executes in a subprocess with a clean environment
- Captures stdout, stderr, exit code
- Hard timeout via `subprocess.run(timeout=...)`
- Working directory is a temp dir, wiped after execution

---

## ReAct Agent (`agents/react.py`)

Implements the **ReAct** (Reasoning + Acting) loop:

1. Append task as first user message
2. Call LLM → get thought + (optional) tool calls
3. If tool calls present: execute each → append results as tool messages
4. If no tool calls (or LLM signals done): extract final answer, end loop
5. Repeat up to `max_steps`

The agent logs every step via `get_logger()` before yielding an `AgentStep`.

---

## Registry (`registry.py`)

Same pattern as `benchmark_legal_rag`: `build(type_path, config_dict)` resolves a dotted class path within the `small_agent` package and instantiates the class. Keeps configs decoupled from imports.

---

## Dependencies

```toml
[project]
dependencies = [
    "pydantic>=2.0",
    "pyyaml",
    "openai",                     # LMStudio is OpenAI-compatible
    "httpx",                      # WebFetch
    "beautifulsoup4",             # HTML parsing
    "google-auth-oauthlib",       # Gmail OAuth2
    "google-api-python-client",   # Gmail API
    "python-dotenv",
]
```

---

## Example Run Flow

```bash
# Set up credentials
cp .env.example .env   # fill in LMStudio URL, Gmail creds path

# Run an agent on a task
python scripts/run_agent.py --config configs/runs/web_research.yaml

# Or override task inline
python scripts/run_agent.py --config configs/runs/web_research.yaml \
    --task "Summarise the latest news about open-source LLMs"
```

Output in `runs/{run_id}/`:
- `run.log` — readable trace of every thought and tool call
- `run.jsonl` — machine-readable step records
- `result.json` — final answer, step count, total tokens, wall time
