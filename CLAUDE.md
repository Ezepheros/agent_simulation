# CLAUDE.md — Small Model Simulation

This file is for anyone (human or AI assistant) reading this codebase who needs to understand how it works, why it is designed the way it is, and where to find things.

---

## What this project is

An experimental framework for running LLM agents powered by **small local models** (via LMStudio) on real-world tasks. The agent has access to tools (web fetch, Gmail, Google Calendar, Python sandbox, clarification) and produces fully structured, step-by-step logs of its reasoning, actions, and results.

The primary goals are:
1. **Observability** — every thought, tool call, tool result, critique, and reasoning token is logged
2. **Configurability** — all behaviour is driven by YAML configs with no code changes required
3. **Pluggability** — tools, backends, agents, and critics are all swappable abstractions
4. **Correctness** — a critic layer catches errors before tools execute

---

## Project layout

```
small_model_simulation/
├── small_agent/               # Main Python package
│   ├── core/types.py          # All domain dataclasses — the shared vocabulary
│   ├── tools/                 # Tool implementations (one file per tool)
│   ├── backends/              # LLM inference adapters
│   ├── agents/                # Agent loop implementations
│   ├── critics/               # Step critic implementations
│   ├── pipeline/              # Run orchestration (config → run → saved output)
│   ├── config/schemas.py      # Pydantic v2 config models with YAML inheritance
│   ├── logging.py             # Structured logging (run-scoped, dual-sink)
│   └── registry.py            # Dynamic component factory
│
├── configs/
│   ├── base.yaml              # Shared defaults inherited by all run configs
│   └── runs/                  # One YAML per task scenario
│
├── scripts/
│   └── run_agent.py           # CLI entry point
│
└── runs/                      # Generated output (gitignored except YAMLs)
    └── {run_id}/
        ├── logs/{run_id}.log
        ├── logs/{run_id}.jsonl
        └── result.json
```

---

## Core design principles

### 1. Everything is a swappable strategy

Every meaningful component — tools, LLM backends, agents, critics — implements an abstract base class (ABC). The pipeline constructs them from config via the registry, never importing them directly. This means:

- Adding a new tool requires only writing one file (`tools/my_tool.py`) and referencing it in a YAML config
- Switching models or backends requires only editing `base.yaml`
- No code needs to change to run different experiments

### 2. Config is the source of truth

All behaviour is defined in YAML. A run config specifies exactly which components to use, with which parameters. Configs use **deep YAML inheritance**: child configs declare `base_config: ../../configs/base.yaml` and only override what differs. This avoids duplication across experiment variants.

The config hierarchy in `config/schemas.py` mirrors the component hierarchy:
- `RunConfig` — top-level, one per run
- `LLMConfig`, `AgentConfig`, `CriticConfig` — component-level
- `ToolConfig` — one per tool instance in the run

### 3. Domain objects are plain dataclasses

`core/types.py` defines all shared data structures as plain Python dataclasses with no framework dependencies. Nothing in the codebase inherits from a framework type. This makes every object:
- Trivially serialisable to JSON (used in `result.json`)
- Testable without any mocks or setup
- Readable without knowing any library internals

### 4. Tools never raise

Every `BaseTool.execute()` implementation catches all exceptions internally and returns a `ToolResult` with `error` set. The agent loop always receives a result object and can reason about failures ("the web fetch timed out — let me try a different URL") rather than crashing.

### 5. Logging is run-scoped

Every log line is tagged with `[run_id]` via a `LoggerAdapter`. Two sinks are always active:
- `.log` — human-readable, for reading during or after a run
- `.jsonl` — one JSON object per line, for structured analysis

Third-party loggers (`httpx`, `httpcore`, `googleapiclient`, etc.) are silenced to WARNING so they don't pollute the output, but errors from them still surface.

---

## How a run works end-to-end

```
scripts/run_agent.py
    → loads RunConfig from YAML (with deep merge from base_config)
    → constructs AgentRunPipeline(cfg)
        → setup_run_logging(run_id, log_dir)   # handlers configured once here
        → _build_backend()                     # LMStudio client
        → _build_tools()                       # one instance per ToolConfig
        → _build_critic(backend)               # optional, shares or owns a backend
        → _build_agent(backend, tools, critic) # ReActAgent
    → pipeline.run()
        → agent.run(task, run_id)
            → for step in range(max_steps):
                → _step(messages, previous_steps)
                    → backend.complete(messages, tools)   # LLM call
                    → if critic: critic.review(...)       # critique before execution
                        → if issues: re-prompt agent once
                    → execute tool calls
                    → append results to messages
                    → return AgentStep
                → if no tool calls → final answer → break
        → return AgentRun
    → _save_result(run)  → runs/{run_id}/result.json
```

---

## The agent loop (`agents/react.py`)

Implements **ReAct** (Reasoning + Acting). The key design decisions:

**Conversation history is the state.** The full message list is passed to the LLM on every step. There is no separate memory or state object — the conversation IS the state.

**The assistant message must carry `tool_calls`.** When the LLM requests tools, the assistant message appended to history must include the structured tool call objects (not just the text content). Without this, the LLM cannot link subsequent `tool` role messages back to what it requested, breaking the context and causing loops. This was a real bug discovered during development.

**No tool calls = final answer.** When the LLM returns a response with no tool calls, the loop ends and the response content becomes the final answer. This is the simplest possible termination signal — no special tokens or phrases required.

**Two failure mode warnings:**
- Empty response (no content, no tool calls) → model likely hit `max_tokens` mid-reasoning
- `<tool_call>` found in reasoning tokens but not in structured output → model leaked its action plan into thinking (Ministral-specific failure mode)

**Today's date is injected into the system prompt** at run time via `{today}` format placeholder. Without this, small models default to their training cutoff date and produce wrong date calculations.

---

## The critic (`critics/`)

The critic is an optional second LLM call that runs **after the agent proposes an action but before tools execute**. It receives:
- The original task
- All completed previous steps (thought + tool calls + results — no prior critiques)
- The current proposed action (thought + proposed tool calls)

It returns either `OK` or a specific error description. If errors are found:
1. The critique is injected as a user message
2. The agent gets one revision attempt
3. The critique exchange is **removed from conversation history** after the revision (it's stored in the `AgentStep` record but doesn't accumulate in context)

**Why remove it from history?** Keeping critiques in context across steps would let the agent "learn" to avoid certain patterns, but would also bloat context and risk the critic's wording influencing future generations in unintended ways. The step record in `result.json` preserves full transparency.

**Why one revision only?** Unlimited revision cycles risk loops where the critic finds different issues each time. One cycle is enough to catch the class of errors we observed (timezone calculations, wrong event IDs, wrong parameter values).

The critic can share the same backend as the agent or use a separate (cheaper/faster) model — controlled by `critic.llm` in the run config.

---

## The backend (`backends/lmstudio.py`)

LMStudio exposes an OpenAI-compatible `/v1/chat/completions` endpoint. The backend uses the `openai` Python client pointed at `http://localhost:1234/v1`.

**Reasoning tokens** (`<think>...</think>`) are extracted from `choice.reasoning_content` — a non-standard field that LMStudio adds to the response when thinking mode is enabled. These are stored in `LLMResponse.reasoning` and logged at INFO level separately from the response content.

**XML tool call fallback:** Some models (e.g. Ministral) write tool calls as `<tool_call><function=name>...` XML in the response content rather than using the structured `tool_calls` field. `_extract_xml_tool_calls()` parses these as a fallback so they execute correctly. This was added specifically because Ministral's prompt template uses XML format while the standard OpenAI client reads structured JSON.

**Message serialisation:** `_to_oai_message()` must include `tool_calls` on assistant messages when they exist. This is required by the API — a `tool` role message that follows an assistant turn without the corresponding `tool_calls` is rejected or misinterpreted by the model.

---

## Tools

### `WebFetchTool`
Fetches a URL with `httpx`, strips HTML via BeautifulSoup, truncates to `max_chars`. No authentication. Removes `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>` tags before extracting text.

### `GmailTool`
Reads emails via the Gmail API (read-only scope). OAuth2 token is cached at `~/.config/small_agent/gmail_token.json` after the first browser-based authorisation. `max_results` and `max_body_chars` are both configurable at the tool level and as defaults via `__init__`. The Gmail service is lazily initialised on first `execute()` call.

### `GoogleCalendarTool` (multiple classes in one file)
Five separate tool classes in `tools/google_calendar.py`, each doing one thing:
- `GetAllCalendarsTool` — list available calendars
- `SearchCalendarEventsTool` — search by query and/or date range
- `CreateCalendarEventTool` — create a new event
- `ModifyCalendarEventTool` — update an existing event by ID
- `RemoveCalendarEventTool` — delete an event by ID

They share the same OAuth2 credentials file as Gmail but use a separate token cache (`calendar_token.json`) and a different scope (`calendar` vs `gmail.readonly`).

### `PythonSandboxTool`
Runs code in a subprocess with a temp working directory. The environment is stripped to only safe variables (`PATH`, `SYSTEMROOT`, etc.) — no secrets are inherited. Hard timeout via `subprocess.run(timeout=...)`. On Windows, the Linux-style `PATH=/usr/bin:/bin` would break Python's ability to find stdlib, so the system PATH is inherited rather than hardcoded.

### `ClarifyTool`
Blocks on `input()` to ask the user a question mid-run. The answer is returned as a `ToolResult` and injected into the conversation as a tool response. The description deliberately discourages overuse — the model should only call this when proceeding with a wrong assumption would cause irreversible harm (e.g. deleting the wrong calendar event).

---

## Registry (`registry.py`)

`build(type_path, config_dict)` takes a dotted class path relative to the `small_agent` package (e.g. `"tools.web_fetch.WebFetchTool"`) and a dict of constructor kwargs, imports the class dynamically, and returns an instance.

This is what allows tool types to be specified as strings in YAML configs. The pipeline never imports tool or agent classes directly — everything goes through `build()`. Adding a new component requires no changes to the pipeline or registry.

---

## Configuration inheritance (`config/schemas.py`)

`RunConfig.from_yaml()` checks for a `base_config` key and recursively loads and deep-merges it. The merge is **override-wins**: every key in the child config replaces the corresponding key in the base. Nested dicts are merged recursively, not replaced wholesale. This means a child config like:

```yaml
base_config: ../../configs/base.yaml
llm:
  model: qwen/qwen2.5-7b-instruct
```

inherits all of `base.yaml` but overrides only the model name — the `base_url`, `temperature`, `max_tokens`, etc. carry through unchanged.

---

## Known failure modes and mitigations

| Failure | Cause | Mitigation |
|---|---|---|
| Agent loops on same tool call | Assistant message missing `tool_calls` field in history | Fixed — `tool_calls` always included in assistant messages |
| Empty final answer | Model hits `max_tokens` mid-reasoning | Warning logged; raise `max_tokens` in config |
| Tool call in reasoning, not executed | Model leaks action into `<think>` block | Warning logged; XML fallback parser for content field |
| System freeze during inference | KV cache spike exceeds available VRAM | Lower context length in LMStudio; reduce `max_body_chars` |
| Wrong date in Gmail queries | Model uses training cutoff date | Today's date injected into system prompt at run time |
| Timezone reasoning errors | Ambiguous task phrasing + model confusion | Critic catches and flags; `ClarifyTool` can ask user |

---

## Adding a new tool

1. Create `small_agent/tools/my_tool.py` implementing `BaseTool`
2. Implement `name`, `description`, `schema`, and `execute()` — `execute()` must never raise
3. Reference it in a run config: `type: tools.my_tool.MyTool`

No other files need to change.

## Adding a new agent type

1. Create `small_agent/agents/my_agent.py` implementing `BaseAgent`
2. Implement `run(task, run_id) -> AgentRun`
3. Reference it in a run config: `agent.type: agents.my_agent.MyAgent`

## Adding a new backend

1. Create `small_agent/backends/my_backend.py` implementing `BaseLLMBackend`
2. Implement `complete(messages, tools) -> LLMResponse`
3. Reference it in a run config: `llm.type: backends.my_backend.MyBackend`
