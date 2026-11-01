# rag-evals-workshop

The code for the four-day **AI Evaluation Engineering for RAG Systems** workshop.
A plain Python project: open the folder in VS Code, install the requirements,
point it at a model, press F5.

It contains three RAG systems over the same employee handbook, and the
evaluators used on each day of the workshop:

| System | What it is | Code |
| --- | --- | --- |
| **Helix** | Vanilla RAG: chunk, embed, retrieve, rerank, assemble the context, generate | `src/helix/` |
| **Helix agent** | Agentic RAG: a tool loop over the same index | `src/agentic/` |
| **Helix graph** | GraphRAG: facts extracted into a graph, answered by linking and traversal | `src/graphrag/` |

Everything runs **live against real models**, and you watch it happen: answers
stream in as the model writes them, and the agent prints each tool call as it
makes it. Use a model on your own laptop through [Ollama](https://ollama.com)
(free, private, no key) or any cloud model through an API key. One option,
`--offline`, swaps in a deterministic stand-in instead; that is the
**reference run**, which reproduces the exact numbers on the slides.

---

## Quick start in VS Code

1. **Get a model.** Either install [Ollama](https://ollama.com), start it, and
   pull two models once:

   ```bash
   ollama pull llama3.1:8b          # answers and tool calls (about 5 GB)
   ollama pull nomic-embed-text     # embeddings (about 300 MB)
   ```

   or have an API key ready for OpenAI, Anthropic or Google (see
   [Choosing a model](#choosing-a-model)).
2. Open the `rag-evals-workshop` folder in VS Code (File > Open Folder) and
   install the Python extension if VS Code suggests it.
3. Open a terminal (Terminal > New Terminal) and run:

   ```bash
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env              # Windows: copy .env.example .env
   python run.py check
   ```

   `check` tests the whole path: the model server or key, one answer, one
   embedding, one tool call, then one question through Helix, streamed. The
   last line should say `CHECK PASSED`. If it does not, it says what to fix.
4. Press **F5** (Run and Debug) and pick what to run from the list:
   *Day 1: hands-on H1*, *Day 3: vanilla vs agentic vs GraphRAG*, *Ask the three
   systems a question*, and so on. Or type the commands below in the terminal.

Python 3.10 or newer.

---

## Choosing a model

The model is set in `.env`, which `.env.example` documents, as
`provider/model`. The project calls every provider over plain HTTPS with
Python's standard library: Ollama through its own API, the cloud providers
through their OpenAI-compatible endpoints. There is no model SDK or gateway
library to install, pin or trust; numpy is the only dependency.

| Setup | In `.env` | Notes |
| --- | --- | --- |
| **Ollama** (default) | `HELIX_GENERATION_MODEL=ollama/llama3.1:8b`<br>`HELIX_EMBEDDING_MODEL=ollama/nomic-embed-text` | Free and private. 16 GB of RAM is comfortable. `ollama/qwen3:8b` calls tools more reliably on Day 3 |
| **OpenAI** | `OPENAI_API_KEY=...`<br>`HELIX_GENERATION_MODEL=openai/gpt-5.4-mini`<br>`HELIX_EMBEDDING_MODEL=openai/text-embedding-3-small` | Fast; a full workshop run costs cents |
| **Anthropic** | `ANTHROPIC_API_KEY=...`<br>`HELIX_GENERATION_MODEL=anthropic/claude-haiku-5-5` | No embedding API: keep Ollama or OpenAI embeddings. Uses Anthropic's OpenAI-compatible endpoint, which Anthropic describes as meant for testing and comparing models |
| **Google** | `GEMINI_API_KEY=...`<br>`HELIX_GENERATION_MODEL=gemini/gemini-3.6-flash`<br>`HELIX_EMBEDDING_MODEL=gemini/gemini-embedding-001` | |

Three more settings:

- `HELIX_JUDGE_MODEL` is the model that grades answers: Day 2's evaluators,
  Day 3's judges and both toolkits. Empty means the answer model grades
  itself; a stronger, different model is the better practice, and comparing
  two judges is one of the Day 3 exercises.
- `HELIX_GRAPH_EXTRACTOR=llm` makes your model extract the GraphRAG facts
  instead of the fixed rules. Compare the two graphs.
- Any command takes `--model`, `--embed` and `--judge` to override `.env` for
  one run: `python run.py day1 --model openai/gpt-5.4-mini --embed openai/text-embedding-3-small`.

**Local and cloud models together.** Each of the three roles can be a
different provider, in the same run. Answer on your laptop and judge with a
cloud model, or the other way round:

```bash
python run.py metrics --model ollama/llama3.1:8b --judge openai/gpt-5.4-mini
python run.py day1 --model anthropic/claude-haiku-5-5 --embed ollama/nomic-embed-text
python run.py check --judge anthropic/claude-sonnet-5-5      # calls each model once, live
```

**Everything is live; nothing is stored or replayed.** Every command calls
the models you name, when you run it. No answer, score or embedding is
cached between commands, and no command shows a saved result in place of a
live one. Within one command, each handbook chunk is embedded once, in
memory, and forgotten when the command ends. The files a command does write
are the work itself: the traces it produced (`data/traces/`), Day 2's
synthetic questions (`data/queries/day2_synthetic.json`), and the sheets you
fill in by hand (`data/annotations/`), because Day 2's labels belong to the
answers your system actually gave. `--offline` is the one exception, and it
stores nothing either: a fixed stub instead of a model, for the reference
numbers on the slides. Its Day 2 files carry `_reference` in their names, so
checking the slides never touches the questions and sheets you work on live.

Pin model versions. Day 4 is about why: an unpinned model makes today's traces
incomparable with yesterday's.

---

## Commands

Everything goes through one file, `run.py`:

| Command | What it does | Day |
| --- | --- | --- |
| `python run.py check` | Check your model connection and the whole path, end to end | before Day 1 |
| `python run.py day1` | Hands-on H1: run six questions live, write six traces, read them, triage the failures. Add `--pause` to stop after each step | 1 |
| `python run.py day2` | Hands-on H2, error analysis: your model writes synthetic questions from dimension tuples (topic, asker, kind of question), Helix answers them with the asker's office and role; then open coding, axial coding (your model proposes groups), counts and saturation. `--annotate` to judge each trace in the terminal, `--label` to add failure modes, `--annotator NAME` for a second person's own sheet, `--new` for new questions | 2 |
| `python run.py agreement` | Exercise 5.A: your labels against a second person's on the same traces. Percent agreement, Cohen's kappa overall and per failure mode (Landis and Koch and McHugh readings), the disagreements grouped by failure mode for the alignment session, then the golden set with a train/dev/test split. `--adjudicate` to decide each disagreement yourself and record why. Run it again after both of you label fresh traces with the revised rubric: it reports round 2 on those traces alone. With three or more sheets it reports Fleiss' kappa | 2 |
| `python run.py evaluators` | Hands-on H3: an evaluator per failure mode, scored against the golden set. Code checks for stale evidence, entitlement, false refusal and missed refusal; one judged mode four ways: our binary judge (Pass and Fail definitions, examples from the train split, reasoning before the answer), G-Eval in our own code, DeepEval GEval, DeepEval DAG; TPR and TNR on dev, the G-Eval threshold read from the labels, and the failure rate corrected for each evaluator's errors with a 95% bootstrap interval. `--mode no_direct_answer` judges another mode, `--repeat 3` checks whether the judge's labels hold, `--split test` once at the end. Offline it prints the exact prompt the judge would receive | 2 |
| `python run.py day3` | Retrieval metrics; groundedness and correctness from a checker and an LLM judge side by side; both validated against human verdicts | 3 |
| `python run.py metrics` | Our metrics beside Ragas and DeepEval on the same traces: retrieval, faithfulness four ways, agent tool use. `--repeat 3` shows how much a judge's score moves | 3 |
| `python run.py architectures` | Vanilla vs agentic vs GraphRAG on 13 questions. Add `--ablations` for the one-switch ablations, `--ids a1,a6` for a few questions | 3 |
| `python run.py day4` | Before/after comparison and the release gate | 4 |
| `python run.py ci` | The release gate as tests: the candidate answers every question live, then Day 2's code checks and a faithfulness floor are asserted on each answer | 4 |
| `python run.py toolkits` | Check the DeepEval and Ragas environment: versions, telemetry off, one live metric from each | before Day 2 |
| `python run.py ask "your question"` | Ask all three systems one question and watch each answer and its trace. Options: `--office berlin`, `--years 4`, `--system agent`, `--trace` | any |
| `python run.py all` | Every day, end to end | |
| `python run.py all --offline` | The reference run: the exact numbers on the slides, no model, a few seconds | |
| `python run.py test` | The test suite (never calls a model) | |
| `python run.py clean` | Delete generated traces and tables. Keeps the sheets people wrote (triage, labels, golden set) unless you add `--all` | |

Each command prints its work as numbered steps, with a sentence on what each
step shows. Output files go to `data/traces/` (one JSON line per trace) and
`data/annotations/` (CSV worksheets you fill in by hand, in the terminal, in
VS Code or in Excel). What you write is never deleted by a script. Each row of
`data/annotations/day2_sheet.csv` carries a fingerprint of the answer it
judged: if that answer changes (you run Day 1 again, or switch models), your
old note moves to the bottom of the sheet as "earlier answer" and the new
answer waits for a fresh judgment. Day 3 validates its judge against the
golden set from `agreement`, or your Day 2 verdicts.

**How long a live run takes.** Day 1 is six answers. `architectures` is about
200 model calls (the agent makes several per question): a few minutes on a
laptop GPU or a cloud model, longer on CPU. Its ablations multiply that by
thirteen, which is why they are opt-in live.

---

## Project layout

```
run.py                   the one entry point (and the --model/--embed/--judge/--offline options)
requirements.txt         numpy and pytest; nothing else
requirements-toolkits.txt  optional: DeepEval and Ragas, for their own environment
constraints-toolkits.txt   advisory floors for their dependencies, and no LiteLLM
.env.example             copy to .env: your model choice and API keys
.vscode/                 F5 run configurations and test settings
src/
  tracing/               the trace format every system writes
    schema.py            Chunk, Span, Trace
    tracer.py            the span() helper, read/write JSON lines
  helix/                 vanilla RAG
    config.py            every setting that changes an answer, plus its hash
    ingest.py            read the handbook, chunk it, embed it, build the index
    bm25.py              keyword search, written out so you can read it
    retrieve.py          keyword, embedding and hybrid retrieval; reranking; access filter
    prompts.py           the prompt template, versioned
    llm.py               the only place that calls a model: answers, streaming, tool calls,
                         embeddings, error hints (and the offline stub)
    console.py           printing a live run: streamed answers, agent steps, progress
    pipeline.py          answer(): the request path, six spans
    structure.py         offices, topics, roles, tables (used by the agent and the graph)
    data/handbook/       the eight handbook pages, four of them with planted problems
  agentic/               agentic RAG
    tools.py             the five tools the agent can call
    agent.py             AgentConfig, the live tool-calling loop, the offline planner
  graphrag/              GraphRAG
    graph.py             nodes, facts and edges
    extract.py           build time: handbook pages -> facts -> graph
    pipeline.py          request time: link, traverse, filter, rank, generate
  toolkits/              DeepEval and Ragas beside our own evaluators (optional)
    env.py               telemetry off, no cloud upload, what is installed
    cases.py             trace -> the test case both toolkits read
    judge.py             your judge model inside each toolkit, through helix.llm
    deepeval_metrics.py  G-Eval, DAG, RAG and agent metrics; code checks as metrics
    ragas_metrics.py     faithfulness, context precision/recall, factual correctness
  evals/                 the evaluators, one file per day's topic
    taxonomy.py          the failure codes: FP1-FP11, AG1-AG5, GR1-GR5
    day1_triage.py       which stage created this failure?
    day2_synthetic.py    test questions from dimensions and tuples
    day2_sheets.py       the traces people read and the sheets they write
    day2_error_analysis.py  open coding, axial coding, counts, saturation
    day2_examples.py     two example annotators for the reference run
    day2_agreement.py    Cohen's and Fleiss' kappa, disagreements, the golden set,
                         its split and its rounds
    day2_evaluators.py   code checks per failure mode, our binary judge and G-Eval
                         code, TPR/TNR, the corrected rate and its interval
    day3_retrieval.py    precision@k, recall@k, MRR, nDCG
    day3_judges.py       groundedness and correctness judges, and checking them
    day3_architectures.py   scoring and attributing all three systems
    day4_compare.py      paired before/after comparison and the release gate
scripts/                 one script per command; each is a readable walkthrough
gates/                   the Day 4 release gate as tests (python run.py ci)
data/queries/            the question sets for each day
data/traces/             written by the scripts
data/annotations/        written by the scripts, filled in by you
tests/                   108 tests (114 in the toolkit environment): the reference results
                         the slides quote, the live path, Day 2, the evaluators, the toolkits
```

---

## How the code works

### 1. Everything writes the same trace

`src/tracing/schema.py` defines three things. A **Chunk** is one piece of
evidence with its provenance: document id, version, last-modified date and
access label. A **Span** is one stage of the request, recording what it kept
and what it dropped. A **Trace** is the full record of one question: every
span, the evidence that reached the model, everything that was dropped along
the way, the answer, and the configuration hash.

All three systems write this same format. That is the main design decision in
the project: the evaluators are written once and work on all three.

### 2. Helix: vanilla RAG

`helix.answer(question, index, config)` in `src/helix/pipeline.py` runs six
steps and records a span for each:

1. `query.transform`: tidy the question (Helix knows the asker's office and
   ignores it, which matters later)
2. `retrieve`: keyword plus embedding search, fused with reciprocal rank
   fusion; the access filter runs here, if it is switched on
3. `rerank`: re-score the shortlist and keep the top three
4. `context.assemble`: fit chunks into the token budget; whatever does not fit
   is dropped and recorded
5. `generate`: build the prompt and call the model
6. `postprocess`: extract citations, detect refusals

`src/helix/config.py` holds two configurations. `DEFAULT` is the system as
shipped, with three configuration defects. `REPAIRED` fixes them: chunks follow
the document's headings, the context budget is larger, and the access filter
is on.

### 3. The four planted problems

| Page | Problem | Failure code |
| --- | --- | --- |
| `parental-leave-2022.md` | Replaced by the 2024 policy but still indexed | FP8 stale, FP9 contradictory |
| `parental-leave-2024.md` | Fixed-size chunks cut the office headings off the tables | FP10 severed context |
| `leaving-the-company.md` | The notice table ranks below the cut-off | FP2 |
| `compensation-bands.md` | Marked hr-only; leaks while the access filter is off | FP11 |

On Day 1 participants find these from the traces, without being told which
questions fail.

### 4. The model layer

`src/helix/llm.py` is the only file that talks to a model. Everything else
calls four functions: `complete` (an answer, streamed if you pass `on_token`),
`chat_with_tools` (one turn of tool calling, for the agent), `embed`, and
`is_refusal`. Live, they send one HTTPS request with `urllib` from the standard
library to whichever model `.env` names: about 250 lines you can read top to
bottom, with nothing between your code and the model.
When a call fails, the error says what to do: start Ollama, `ollama pull` the
model, or set the key.

Live models paraphrase, so the pipeline does not trust exact strings. A
citation counts when the chunk id appears in the answer, with or without its
version and date. A refusal is recognised in its common phrasings, and the
prompt (`prompts.py`, version v2) asks for one fixed refusal sentence to make
that easier. A reasoning model's `<think>` block is stripped before anything
reads the answer.

The **reference run** (`--offline`) replaces the model with a small
deterministic stub that reports the first matching fact in the best-ranked
extract it was given. It never invents anything, so every wrong answer it
gives is caused by what the pipeline gave it. It exists so that the numbers
on the slides are reproducible on any laptop; the live run is where you see
what your model actually does with the same evidence.

### 5. The agent

`src/agentic/agent.py`. Instead of a fixed path, the agent chooses its steps
from five tools in `tools.py`:

| Tool | What it does |
| --- | --- |
| `get_user_profile` | the asker's office, role and years of service |
| `list_offices` | the offices the handbook covers |
| `search_handbook(query, office)` | the same retrieval as Helix, optionally filtered to one office |
| `check_document_status(doc_id)` | is this page current, or replaced by a newer one? |
| `finish(answer)` | stop and answer |

Every tool call is written to the trace as a span (`agent.step.1`,
`agent.step.2`, ...) with its arguments and what came back, and printed as
it happens. Live, your model makes every choice through tool calling; the
code only runs the tools and counts the steps.

`AgentConfig` has one switch per behaviour (look up the profile, filter by
office, split multi-part questions, check versions, act on what the check
says, step budget). Live, a switch takes the capability away: switch off
`check_versions` and the model is no longer offered `check_document_status`
or told to use it; switch off `office_filter` and `search_handbook` loses its
`office` argument. Whether the model then fails is something you observe.
In the reference run, a deterministic planner follows the switches exactly,
so each one reproduces one agent failure, AG1 to AG5, every time.

### 6. The graph

`src/graphrag/`. At **build time**, `extract.py` reads every handbook page and
writes down atomic facts: one table row each, with its office and caregiver
role attached, plus the page's version, date and access label. Two pages with
the same title get a "supersedes" edge from the newer to the older.

At **request time**, `pipeline.py` links the question to offices and topics,
walks the graph from those nodes, filters out superseded and restricted facts,
ranks what is left, and passes the facts to the same generator Helix uses. If
the graph knows nothing about the topic, it falls back to normal retrieval.

`GraphConfig` has one switch per behaviour (alias linking, use the asker's
office, extract from sections or from fixed chunks, keep access labels, drop
superseded facts, fall back, how far to walk). Turning one off reproduces one
graph failure, GR1 to GR5.

### 7. The evaluators, day by day

- **Day 1** `evals/day1_triage.py`: reads a trace and names the stage that
  created the failure. It can prove FP8, FP9, FP10 and FP11 from the trace
  alone; FP2 and FP3 need to know what the right answer was.
- **Day 2** `evals/day2_*.py`: generates test questions from dimensions and
  tuples, then supports open coding (free-form notes on the first thing that
  went wrong), axial coding (grouping notes into failure modes), counting how
  often each occurs, and checking saturation. `day2_agreement.py` measures how
  far two annotators agree (Cohen's kappa, per failure mode), groups the
  disagreements for the alignment session, checks a revised rubric on fresh
  traces, and builds the golden set with a train split that gives every
  judged mode its examples. `day2_evaluators.py` turns each failure mode into
  an evaluator: code where the trace records the fact, a binary judge where
  deciding needs reading, with G-Eval beside it. Each is scored against the
  golden set on its own mode (TPR and TNR), and its raw failure rate is
  corrected for its known errors (Rogan and Gladen), with a bootstrap
  interval.
- **Day 3** `evals/day3_retrieval.py` scores retrieval (precision, recall,
  MRR, nDCG); `day3_judges.py` scores groundedness and correctness separately
  and checks the judge against human labels; `day3_architectures.py` scores
  all three systems on the same questions and attributes every wrong answer to
  a failure code.
- **Day 4** `evals/day4_compare.py`: compares two configurations on the same
  questions and applies a release gate agreed in advance.

---

## What the Day 3 comparison shows

`python run.py architectures` answers 13 questions with six systems. A cell
is `ok` or the failure code assigned to the wrong answer. This is the
**reference run** (`--offline`), the version quoted on the slides:

```
       helix-default helix-repaired    agent-naive          agent    graph-naive          graph
a1       FP9/FP8/FP3            FP4            FP4            FP4             ok             ok
a2              FP10            FP4            AG1             ok        GR1/GR2             ok
a3          FP10/FP9            FP9            AG4            AG4             ok             ok
a4               FP2        FP9/FP2            AG3            FP4             ok             ok
a5          FP10/FP9             ok             ok             ok        GR1/GR2             ok
a6              FP11            FP1            FP1            FP1       FP11/GR4             ok
a7           FP9/FP8    FP9/FP8/FP2        FP8/AG1             ok    FP8/GR3/GR1             ok
a8               FP2            FP2            FP4            FP4             ok             ok
a9                ok             ok             ok             ok             ok             ok
a10               ok             ok             ok             ok            GR5             ok
a11               ok             ok             ok             ok            GR2            GR2
a12         FP10/FP2            FP4             ok             ok        GR1/GR2             ok
a13               ok            FP9            FP1            FP1            GR2             ok
right           4/13           4/13           5/13           7/13           5/13          12/13
```

- The **agent** fixes what a fixed pipeline cannot decide: an office implied
  by who is asking (a2, a12) and a superseded page (a7). It runs out of steps
  when it has to compare five offices (a3).
- The **graph** is strong where graphs are strong: comparisons (a3) and two
  facts about one office (a4). Its score depends on its extraction rules,
  written for these tables; a11 is the question no rule covers.
- No system wins every row, and each new failure is born somewhere vanilla RAG
  does not have.

The script then turns each switch off one at a time and shows which question
breaks and which code it gets.

**Your live run will differ, and that is the exercise.** A real model reads
a severed table better or worse than the stub, calls tools in its own order,
and sometimes refuses where the stub answered. The planted failures are in
the corpus and the configuration, so most of them still show up; which ones,
and under which code, depends on your model. Run the same command with two
models and compare the matrices.

---

## When something goes wrong

| Message | Fix |
| --- | --- |
| `Ollama is not running` | Start the Ollama app, or run `ollama serve` in another terminal |
| `Ollama does not have this model yet` | `ollama pull <model>`, exactly as the message prints it |
| `... needs OPENAI_API_KEY` (or another key) | Put the key in `.env`; check the file is named `.env`, not `.env.txt` |
| `tools WARNING` in `check` | Your model answered without calling the tool. The agent will be weak; try `ollama/qwen3:8b` or a cloud model |
| Very slow answers | Ollama is running on CPU. Use a smaller model, or a cloud model for the long `architectures` run |
| No network in the room | `python run.py <command> --offline` runs everything with the reference stub |

---

## The evaluation toolkits: DeepEval and Ragas

From Day 2 on, two widely used toolkits run beside the evaluators you build,
on the same traces and the same human labels. They show what a toolkit gives
you quickly, and what its defaults cannot see.

| Day | Command | What the toolkit does there |
| --- | --- | --- |
| 1 | `day1` | Nothing yet. The script prints Sara's trace as the test case both toolkits read: the trace already holds every field they need |
| 2 | `evaluators` | DeepEval GEval and a DAG metric judge the same failure mode as the workshop's own binary judge and G-Eval code, with the steps your judge writes in that run; all are checked against your labels |
| 3 | `metrics` | Ragas and DeepEval faithfulness, context precision and recall, factual correctness and agent tool metrics, beside our instruments and the human verdict |
| 4 | `ci` | The candidate configuration answers every question live, then DeepEval's `assert_test` runs the release gate, with Day 2's code checks inside it as metrics |

**Set up a separate environment for them.** The core workshop needs only
numpy. The toolkits bring dozens of packages, including model SDKs the
workshop never calls, so they live in `.venv-toolkits`:

```bash
py -3.12 -m venv .venv-toolkits              # macOS/Linux: python3.12 -m venv .venv-toolkits
.venv-toolkits\Scripts\activate              # macOS/Linux: source .venv-toolkits/bin/activate
pip install -r requirements.txt -r requirements-toolkits.txt -c constraints-toolkits.txt
pip install pip-audit && pip-audit
python run.py toolkits                       # ends TOOLKITS CHECK PASSED
```

What the code guarantees, and where:

- **Your judge, your connection.** Each toolkit accepts a custom model object.
  `src/toolkits/judge.py` gives it one that calls `helix.llm`, so every judge
  call uses the same audited standard-library path as the rest of the
  workshop. No toolkit opens a model connection of its own, and LiteLLM is
  never installed (`constraints-toolkits.txt` makes pip refuse it).
- **No vendor gets your data.** `src/toolkits/env.py` turns off both
  toolkits' telemetry, stops DeepEval reading `.env`, and removes any
  `CONFIDENT_API_KEY`, so DeepEval cannot upload results. The only network
  calls are to the models you named: your Ollama, or the API you chose.
- **Nothing is stored.** DeepEval's `evaluate()` normally saves each run
  and a result cache in `.deepeval/`; the workshop runs it in read-only
  mode, keeps DeepEval's pytest plugin off, and enables no Ragas cache. A
  test runs both toolkits with the home and working folders isolated and
  fails if a single file appears.
- **Known issues, written down.** Ragas 0.4.3 carries CVE-2026-6587, an SSRF
  in its multi-modal image loader, with no fix released. The workshop never
  builds the multi-modal metrics, and a test fails if anything imports them.
  Ragas's cache (CVE-2025-69872, in diskcache) is never enabled.
- **JSON that fails is counted, not hidden.** Toolkit judges need JSON in a
  fixed shape. A reply that does not fit gets one second chance with the
  error; every run reports how many needed it. A small local model may need
  many: use `--judge` with a stronger model for the toolkit commands.

Without the toolkits installed, `evaluators` and `metrics` still run every
instrument of our own and leave the toolkit columns empty. With `--offline`
they compute only what needs no model; the judge and toolkit columns stay
empty, because there is no stored run to show instead.

---

## Tests

```bash
python run.py test
```

In VS Code the Testing panel (the flask icon) finds them automatically. The
tests never call a model. Thirty-five pin every result the slides quote, from
the reference run: the six Day 1 failures, the Day 3 matrix, and the code each
switch produces. If one fails after you change something, a slide no longer
matches the code. Twenty-three start a small fake model server on your own
machine that speaks Ollama's API and the OpenAI-compatible one, and drive the
real network code against it: streaming, tool calling in both formats,
embeddings, JSON output for judges, local answers with a cloud judge in one
run, citation and refusal detection on
paraphrased answers, the agent's switches, judge parsing and the error
messages. Forty cover Day 2: the synthetic tuples, the sheets and what they
never delete, terminal annotation, Cohen's and Fleiss' kappa on worked
examples, every reference figure on the Day 2 slides, the split and the
judge examples it gives train, round 2 on fresh traces, saturation, and the
whole reference run from a clean copy. Ten cover the Day 2 evaluators: the
binary judge with a scripted model, our G-Eval code, the corrected failure
rate and its bootstrap interval, and the safety settings.

In the toolkit environment six more run the real DeepEval and Ragas metrics
through the workshop's judge, with a scripted model that answers in each
toolkit's JSON shape and a guard that fails the test if anything opens a
network connection, and one fails if either toolkit writes a single file.
Elsewhere they are skipped.

---

## References

- Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), NeurIPS 2020
- Barnett et al., [Seven Failure Points When Engineering a Retrieval Augmented Generation System](https://arxiv.org/abs/2401.05856), CAIN 2024
- Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), ICLR 2023
- Edge et al., [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130), 2024
- Robertson and Zaragoza, [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019), 2009
- Cormack, Clarke and Buettcher, [Reciprocal Rank Fusion](https://doi.org/10.1145/1571941.1572114), SIGIR 2009
- Liu et al., [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634), EMNLP 2023
- Es et al., [Ragas: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217), EACL 2024
- DeepEval, [G-Eval](https://www.deepeval.com/docs/metrics-llm-evals) and [data privacy](https://deepeval.com/docs/data-privacy)
- Ragas on PyPI, [ragas 0.4.3](https://pypi.org/project/ragas/); the advisory [CVE-2026-6587](https://advisories.gitlab.com/pypi/ragas/CVE-2026-6587/)
- OpenTelemetry, [Semantic conventions for generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
