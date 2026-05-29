# Jarvis: Design, Implementation and Evaluation of a Local First Multiagent Architecture for LLM Driven Personal Executive Assistance

*A Modular Approach to Reasoning, Memory and Tool Augmented Task Automation*

**Author:** Abdullah Khan Durrani
**Module:** COM6001, Final Year Project (Pathway A: Software Development)
**Institution:** Buckinghamshire New University
**Submitted:** 20 May 2026
**Word count:** approximately 10,900 (body)

---

## Acknowledgments

I would like to thank the academic staff at Buckinghamshire New University for their guidance, candour and patient feedback throughout this project. I am also grateful to the University for the academic resources that made this work possible, and to the friends and family who tested Jarvis through its many iterations and offered invaluable real world feedback.

---

## Abstract

Large Language Models have transformed conversational AI, yet a notable gap remains between their reasoning capacity and their practical utility as personal assistants. Deployed as single shot systems, they struggle with the cognitive labour they are most often asked to perform: planning multi step tasks and reliably operating the tools that translate intent into action. This dissertation contends that the limitation is largely architectural rather than one of model capability. It presents **Jarvis**, a local first multiagent executive assistant that decomposes assistant behaviour across nine specialised agents (Router, Planner, Memory, Critic, Evaluator, Summariser, Calendar, Gmail and FinEx) coordinated through a central orchestrator and supported by seventeen integrated tools and a hybrid voice pipeline. Three architectural contributions are introduced: tier based routing that adapts pipeline depth to task complexity, dependency aware task graphs with templated parameter injection, and a critic driven replanning loop. Evaluated through a scenario based benchmark across multiple locally hosted models, Jarvis suggests that the bottleneck in personal AI lies less in the model than in the architecture that surrounds it.

---

## §1 Introduction

For three decades the personal digital assistant has been a recurring promise rather than a delivered product. Each generation of computing produced systems that gestured at general assistance while remaining narrowly capable in practice: the desktop in the 1990s, the smartphone in the 2000s, voice first devices in the 2010s. The arrival of Large Language Models (LLMs) in the early 2020s, trained at unprecedented scale on web derived text, has fundamentally reshaped what an assistant might do. Models such as GPT 4, Claude and Llama 3 can discuss arbitrary topics, write code, draft prose and explain complex concepts in fluent natural language. Yet the assistants built around them remain, in operational terms, single shot conversationalists. They reason in isolation, struggle to plan more than one step ahead, and cannot reliably press the buttons that real world tasks require.

This dissertation reports the design, implementation and empirical evaluation of Jarvis, a local first multiagent executive assistant developed during the 2025 to 2026 academic year as the candidate's final year project at Buckinghamshire New University. Jarvis is neither a single prompt chatbot dressed as an agent nor a workflow automation pipeline. It is a coordinated society of specialised LLM powered agents engineered to plan, remember, critique, route and execute. The remainder of this report follows the BNU Pathway A structure: background and rationale (§2 and §3), aim, objectives and risk profile (§4 to §7), literature survey (§8), methodology (§9), requirements and design (§10 and §11), implementation and testing (§12 and §13), and conclusions and recommendations (§14 and §15).

---

## §2 Background

Conversational digital assistants have evolved through four broadly identifiable generations. The first, exemplified by ELIZA (Weizenbaum, 1966), employed surface level pattern matching to simulate conversation. It possessed neither a model of the world nor any capacity for action. The second generation, comprising Siri (2011), Google Now (2012) and Alexa (2014), paired intent classification pipelines with handcrafted skill modules, enabling narrow but reliable execution of preset commands. The third, born of the transformer revolution and crystallised in ChatGPT (Brown et al., 2020; OpenAI, 2022), replaced the brittle intent layer with general purpose language models capable of free form reasoning, but in doing so abandoned reliable tool use and structured planning. The fourth, currently emerging, comprises tool augmented and agentic systems such as AutoGPT (Richards, 2023), BabyAGI (Nakajima, 2023), CrewAI (Moura, 2024), LangGraph (LangChain Inc., 2024) and Microsoft AutoGen (Wu et al., 2023), each attempting to recombine LLM reasoning with the deterministic execution properties of earlier assistants. It is in this fourth generation that Jarvis is situated. Where most agentic systems either depend on closed weight cloud models or focus narrowly on developer oriented benchmarks, Jarvis explores whether a user facing, locally hosted, multi tool assistant can serve as a daily executive aide rather than a research demonstration.

---

## §3 Rationale

The academic motivation for this project arises from a clear absence in the published literature. The individual components of modern AI agents (chain of thought prompting, retrieval augmented generation, tool using agents, multiagent orchestration) are each separately well studied. Comparatively few publicly documented systems, however, combine all of these elements within a single user facing assistant subject to end to end empirical evaluation. The majority of published agent systems are framework demonstrations on synthetic benchmark suites such as HumanEval or AgentBench. Relatively few are evaluated as personal organisation tools used over sustained periods of natural interaction.

The personal motivation is no less considered. Across six months of development, the candidate has used Jarvis as a working tool, extending it iteratively, observing where it failed, and learning what the integration of LLM reasoning with daily workflows demands in practice. This dual perspective, both academic and pragmatic, is reflected throughout the report.

That perspective was further sharpened by feedback received from departmental academic staff during the interim phase of the project. Specific critiques were raised concerning the methodological rigour of design science approaches and the absence of a model comparison study. Both shaped the form the final evaluation took, as described in §9 and §13. Engagement with that feedback is reflected explicitly in the latency versus coordination trade off that emerges as a principal finding of this work, discussed in §11.3 and elevated to scholarly contribution in §14.

---

## §4 Ethical Considerations

Jarvis interacts with materially sensitive data: emails, calendars, file contents, microphone input, and financial documents. Ethical handling is therefore a first order design requirement rather than an afterthought. Three principles guided the system throughout development. The first is **data sovereignty**. All reasoning, embedding generation and persistent storage occur on the user's own machine wherever possible, using locally hosted models via Ollama, a local ChromaDB instance, and OAuth mediated access to cloud services strictly scoped to the resources required. No user content is transmitted to third party AI providers in the default configuration. The second is **user autonomy**. Jarvis never executes a destructive or externally visible action (sending an email, creating a calendar event, deleting a file) without explicit user confirmation, enforced through pending state intercepts in the orchestrator. The third is **epistemic honesty**. Because LLMs are prone to hallucination, a dedicated Critic agent reviews every generated plan before execution and every result before presentation, with replanning triggered when plans score below a configurable threshold. Risks of model bias, inappropriate content generation and unintended tool invocation are acknowledged and discussed further in §7. The institutional ethics checklist required by Buckinghamshire New University is provided as Appendix B; the consent form used during informal user testing appears as Appendix C.

---

## §5 Aim

To design, implement and empirically evaluate a modular, locally hosted multiagent system capable of decomposing natural language requests into dependency aware task plans, retrieving contextual memory, executing authenticated external tool actions, and self reviewing its own outputs. The system should deliver reliable executive assistant behaviour without dependence on closed weight cloud models.

---

## §6 Objectives

The project pursues seven specific objectives:

1. Design a multiagent architecture that explicitly separates routing, planning, memory, criticism, execution and evaluation concerns, in line with established agent oriented software engineering principles.
2. Implement structured task decomposition with machine readable, dependency aware JSON plans suitable for deterministic execution.
3. Provide persistent semantic memory via vector embeddings and cosine similarity retrieval, with episodic storage of task outcomes.
4. Integrate authenticated external services (Google Calendar, Gmail, Spotify, weather, news, financial data and document processing) through a unified tool abstraction layer.
5. Introduce a critic driven replanning loop that improves output reliability through self review.
6. Deliver a hybrid voice interface combining local wake word detection, local speech to text, and cloud text to speech to support hands free use.
7. Evaluate the system across multiple locally hosted models on planning quality, execution success, latency and replan frequency, and document the resulting trade offs.

---

## §7 Risks

Project risks were assessed across technical, operational and academic dimensions, and managed throughout development.

The most material technical risk is **LLM hallucination**: the generation of plausible but incorrect plans or tool parameters. This was mitigated through structured JSON only prompt design, low temperature decoding, schema aware parsing, and the Critic agent's pre and post execution review with bounded replanning (a maximum of two attempts per task). **API quota and rate limit exposure** to external services such as Google APIs, Spotify and the news providers was mitigated through exponential backoff retry logic and graceful per tool degradation. **OAuth token leakage** was controlled through environment isolated secrets and scope minimised credentials. **Local model variance**, the observation that smaller Ollama models occasionally produce malformed JSON, was addressed by post processing routines that strip markdown formatting and enforce single paragraph output before responses reach the user interface.

Operationally, **scope drift** posed a continuous risk as the project's tool surface expanded well beyond the original proposal. Rather than concealing this, the expansion is recorded in §9 (Methodology) and treated as an empirical observation about architectural extensibility. **Memory scalability** is a known residual risk. The current cosine similarity retrieval performs adequately at the prototype scale (under ten thousand embeddings) but would require approximate nearest neighbour indexing in production, as discussed in §15.

Academically, the principal risk was **insufficient empirical evidence** to support a First Class evaluation. The original proposal's single versus multiagent baseline was reframed mid project (§9.2) into a multi model, multi tier benchmark using infrastructure already instrumented in the codebase. Time risk, posed by a six month delivery window with a single developer, was managed through phased milestones aligned with BNU's interim report schedule.

---

## §8 Literature Survey

Three threads of literature are directly relevant to the design of a personal multiagent LLM assistant, and a fourth (local first deployment) has only recently begun to be addressed. This survey examines each in turn before identifying the gap into which Jarvis is positioned.

### 8.1 Reasoning Frameworks for Large Language Models

The use of LLMs as reasoning engines, rather than text generators, has been formalised through a succession of prompting paradigms. Chain of Thought (CoT) prompting, introduced by Wei et al. (2022), demonstrated that prompting a sufficiently large model to "think step by step" markedly improved its performance on arithmetic and symbolic reasoning tasks. CoT established that latent reasoning ability could be elicited through prompt design alone, without architectural modification. Building upon this, Kojima et al. (2022) showed that zero shot CoT was achievable simply by appending a reasoning trigger phrase, while Wang et al. (2023a) introduced self consistency, in which multiple independent reasoning traces are sampled and the most consistent answer selected. Self consistency improves accuracy at the cost of additional inference.

ReAct (Yao et al., 2022) extended CoT by interleaving reasoning steps with explicit *actions*, typically calls to external tools or knowledge sources, so the model could observe environmental feedback before committing to an answer. ReAct established the now dominant *thought, action, observation* loop, and forms the conceptual basis of most production LLM agent frameworks. Tree of Thoughts (Yao et al., 2023) generalised this further into branching exploration over multiple reasoning paths, enabling backtracking when partial solutions failed, at the cost of substantially greater inference budget. Reflexion (Shinn et al., 2023) added an explicit self critique step in which the agent evaluates its own prior outputs and revises its strategy, an idea adopted in modified form by Jarvis's Critic agent.

Two ancillary developments are particularly germane. The first is **structured output prompting**, variously implemented as JSON Mode, function calling, or grammar constrained decoding. It has moved from heuristic engineering to a first class concern of model providers, with measurable improvements in downstream parsing reliability (Beurer Kellner et al., 2024). The second is the emerging literature on **prompt driven planning** (Liu et al., 2023a; Hao et al., 2023), which suggests that decoupling planning from execution (using the LLM strictly as a planner, with a separate deterministic executor acting upon its plans) yields more reliable behaviour than monolithic prompt designs. Jarvis follows this latter pattern explicitly, treating the Planner agent as a pure reasoning component whose JSON output is consumed by an independent executor.

### 8.2 Multiagent and Tool Augmented LLM Systems

If single agent reasoning frameworks describe what one LLM call can be made to do, the literature on multiagent systems addresses what becomes possible when several specialised LLM agents are composed. The first generation of such systems emerged in early 2023. AutoGPT (Richards, 2023) and BabyAGI (Nakajima, 2023) both employed an unbounded recursive planning architecture in which a single model iteratively generated and executed subtasks until a termination condition was reached. These systems captured significant public attention but exhibited well documented failure modes (runaway looping, hallucinated tool calls, and an inability to recover from mid execution errors) that limited their practical utility (Park et al., 2023; Zhao et al., 2024).

More principled architectures followed. CrewAI (Moura, 2024) introduced a role based collaboration model in which agents are assigned named functions (researcher, writer, reviewer) and communicate via structured messages. LangGraph (LangChain Inc., 2024) reframed agent coordination as a state graph traversal, providing explicit control over branching, cycles and termination, properties that earlier autonomous agent frameworks lacked. Microsoft's AutoGen (Wu et al., 2023) emphasised conversational multiagent collaboration, with agents exchanging natural language messages that can be inspected and edited by a human in the loop. MetaGPT (Hong et al., 2023) imposed a software engineering metaphor, casting agents as product managers, architects and engineers within a defined workflow. A common pattern emerged across these systems: explicit separation of agent roles, structured inter agent communication, and at least some bounded execution control.

A second body of work addresses the related problem of *tool use*. Toolformer (Schick et al., 2023) showed that an LLM could be fine tuned to call external APIs as part of its inference. Subsequent systems including Gorilla (Patil et al., 2023), ToolLLM (Qin et al., 2023) and the OpenAI Function Calling interface (OpenAI, 2023) explored the design space for delegating subtasks to deterministic external services. The dominant pattern is one of *tool abstraction*: the LLM emits structured calls describing the intended action, and a deterministic runtime layer translates those calls into authenticated API invocations.

Despite these advances, three persistent limitations characterise the literature. First, **evaluation tends to be synthetic**. The majority of published systems are tested on benchmarks such as AgentBench (Liu et al., 2023b), HumanEval (Chen et al., 2021) or WebArena (Zhou et al., 2024), with comparatively little empirical work on assistants used over weeks of natural personal interaction. Second, **cloud dependence is near universal**. Most published agentic systems are evaluated using GPT 4 or Claude as the underlying model, raising questions of cost, privacy and reproducibility. Third, **scope is typically developer oriented**. Agents are designed for coding, research or customer service tasks rather than the heterogeneous mix of communication, planning, media control and information retrieval that characterises personal assistance.

### 8.3 Memory Augmented Language Models

The third strand concerns how LLMs handle information beyond their context window. Retrieval Augmented Generation (RAG), formalised by Lewis et al. (2020), pairs a retriever (typically a dense vector index) with a generator, allowing the model to ground its outputs in a corpus of external documents at inference time. RAG has become the dominant pattern for grounding LLM outputs in private or up to date data, with widespread industrial adoption since 2023. Significant subsequent work has refined the retriever component (Karpukhin et al., 2020), explored alternative chunking and indexing strategies (Gao et al., 2024), and introduced hybrid sparse dense retrieval (Khattab and Zaharia, 2020).

For agentic systems specifically, the distinction between **semantic** and **episodic** memory has proven analytically useful. Semantic memory stores factual content such as user preferences, reference documents and learned associations, all of which can be retrieved during reasoning. Episodic memory, by contrast, stores records of past interactions: prior tasks, their outcomes, and the contexts in which they occurred. Generative Agents (Park et al., 2023) demonstrated that episodic memory with reflection and abstraction enabled emergent social behaviour among simulated agents. MemGPT (Packer et al., 2023) introduced operating system inspired memory hierarchies that page information in and out of the LLM context window as needed. Voyager (Wang et al., 2023b) added a *skill library*, a self extending repository of learned procedures, as a third memory class.

Two engineering concerns shape memory augmented designs in practice. The first is the choice of *vector index*. Brute force linear scan search remains adequate at small scales but degrades poorly past tens of thousands of entries. Approximate nearest neighbour algorithms such as HNSW (Malkov and Yashunin, 2018) and IVF (Jégou et al., 2011) are widely adopted in production. ChromaDB, the vector store used in Jarvis, employs HNSW by default. The second concern is *embedding fidelity*. Domain specific text often benefits from fine tuned embedding models (Reimers and Gurevych, 2019), though general purpose models such as `nomic embed text` and `sentence transformers/all MiniLM L6 v2`, the two used in Jarvis's local and cloud deployments respectively, perform adequately for personal assistant workloads.

### 8.4 The Gap

Each of the three strands surveyed above represents a mature line of research in its own right. Their *intersection*, however, is materially underexplored in the published literature: a user facing, locally hosted multiagent system that combines structured planning, semantic and episodic memory, authenticated tool use, and self critical replanning within a single integrated artefact subject to empirical evaluation. Existing agentic systems are predominantly cloud bound research artefacts evaluated on synthetic benchmarks. Commercial assistants emphasise convenience over architectural transparency or user control. Jarvis is positioned squarely in this gap. It is not novel in any individual subsystem; its planning is derived from ReAct, its memory is conventional RAG, its tool layer is unremarkable. As an *integrated, locally hosted, daily use* system documented end to end with empirical evaluation, however, it occupies a position that the surrounding literature largely vacates. The remainder of this report describes that integration in detail.

---

## §9 Methodology

### 9.1 Research and Development Approach

This project adopts a **Design Science Research (DSR)** methodology, originally articulated by Hevner et al. (2004) and well suited to the development and evaluation of technological artefacts that address practical problems. The artefact constructed in this work is Jarvis itself, a multiagent executive assistant system whose architecture, implementation choices and empirical performance constitute the principal object of study.

It is fair to acknowledge that DSR is sometimes criticised as a vague or post hoc framing of what amounts to iterative software development. The distinction maintained throughout this project is twofold. First, DSR explicitly treats the *artefact* as the object of evaluation, requiring each design decision to be justified against research questions rather than user story acceptance criteria. This aligns every development cycle with academic outcomes rather than purely functional delivery. Second, DSR's evaluation phase is constitutive rather than supplementary. Each iteration must produce evidence of effectiveness, not merely a working build. This distinction shaped the project's instrumentation choices directly, in particular the EvaluatorAgent's persistence of per task scoring data from the earliest stages of development, rather than its retrofitting at the end (Peffers et al., 2007).

The DSR process followed an iterative cycle of five activities. **Problem identification** surfaced the limitations of monolithic LLM systems in multi step, tool integrated tasks. **Artefact design** proposed a modular multiagent architecture as the response. **Implementation** produced functional agents in asynchronous Python, refined over six months of development. **Evaluation** tested the system against representative task scenarios and recorded quantitative performance. **Refinement** fed observed behaviour back into the next design iteration. Each cycle was tied to measurable outcomes (routing accuracy, response latency, agent coordination overhead) rather than feature completion alone, and each produced concrete revisions to the planner's prompt design, the orchestrator's pending state handling, or the router's tier classification.

### 9.2 Scope Evolution

A methodological note of honest disclosure is warranted. The original proposal (Durrani, 2025) committed to a single agent versus multiagent baseline comparison as the primary evaluation strategy. During implementation, this commitment was reassessed. The construction of a meaningful single agent baseline would have required developing a separate, comparably tooled monolithic version of Jarvis, a duplication of the project's entire surface area, since the system's tool inventory is part of what would need to be compared. Worse, such a comparison would conflate *architecture* with *tooling* and offer no defensible procedure for controlling that confound. In its place, an evaluation framework more directly exercising the architectural decisions of interest was adopted: multi model comparison across locally hosted Ollama models, multi tier latency analysis exploiting Jarvis's tier based routing (§11.3), and per intent execution success analysis.

A second pivot warrants documentation. The original proposal nominated Notion (or Todoist) as the principal note taking integration. During implementation, a personal finance use case emerged as a more architecturally interesting demonstration of the system's extensibility: extracting structured data from financial statement PDFs and answering analytic questions against the extracted tables. The **FinEx** subsystem, a domain specific agent supported by a PDF extractor, a ChromaDB indexed embedding store and a SQL generating LLM, was developed in its place. Notion was deprioritised in favour of demonstrating that the same orchestration pattern accommodates substantially different domain semantics, a finding that more directly evidences the architecture's generality than the originally planned integration would have done. The system's tool inventory similarly expanded opportunistically beyond the proposal's scope to include Spotify control, sports, market data, prayer times and macOS system control. Each was adopted in response to a specific personal use scenario observed during development, rather than as a speculative feature addition. Both pivots are consistent with DSR's iterative character, which explicitly permits, and indeed expects, research scope to evolve as the artefact matures.

### 9.3 Evaluation Framework

The evaluation is multi dimensional. Quantitatively, every task executed by Jarvis is scored by the EvaluatorAgent on five metrics: a **planning score** (the Critic's pre execution review, in the range 0 to 1), an **execution score** (the proportion of subtasks completing successfully), a **latency** measurement in milliseconds (wall clock time across the full pipeline), a **subtask count**, and a **replan count** (the number of times the Critic triggered replanning). These metrics are persisted to SQLite and exported as JSON and CSV for analysis.

A benchmark suite of nine representative tasks (defined in `main.py`) spans the three difficulty tiers introduced in §11.3 and seven of the system's intent categories. The suite is designed to be exercised across multiple locally hosted models: `llama3.2:1b`, `llama3.2:3b`, `phi3:mini` and `gemma:2b`. These were selected to span the 1 to 3 billion parameter range where local CPU inference remains practical for personal hardware. Qualitative observation supplements the quantitative results. Failure modes are catalogued by type with representative examples discussed in §13.4. The system has been used by the candidate as a daily personal assistant since April 2026, providing informal real world validation that a laboratory benchmark alone cannot.

### 9.4 Tooling and Implementation Approach

Implementation is in Python 3.10 and uses `asyncio` for non blocking concurrency. Local LLM inference is provided by Ollama, with `llama3.2:latest` as the default chat backbone. Cloud deployment, where used, substitutes Groq's hosted Llama 3.3 70B for reasoning and `sentence transformers/all MiniLM L6 v2` for embeddings. Vector storage uses ChromaDB. Relational persistence (evaluations, episodic memory metadata) uses SQLite locally and Neon hosted PostgreSQL for the FinEx subsystem. The web interface is delivered through FastAPI with a single page heads up display front end communicating over WebSocket. Voice interaction layers faster whisper for speech to text, openWakeWord for wake detection, Silero VAD for endpoint detection, and ElevenLabs for cloud text to speech. Containerised deployment uses Docker, with production hosting on Fly.io's London region.

---

## §10 Requirements

### 10.1 Stakeholders and Use Context

The primary stakeholder is the **end user**, a single individual using Jarvis as a personal executive assistant for everyday cognitive labour: scheduling, communication, information retrieval, file operations, media control and financial document analysis. The system is designed to be operated by one user from one machine at a time. Multi user concurrency, while architecturally permissible, is explicitly out of scope. Secondary stakeholders include the **academic supervisor**, for whom the artefact must demonstrate substantive engineering and empirical evaluation, and **future maintainers** (including the candidate's future self) for whom architectural transparency and documentation are necessary. The system is deployable in two distinct contexts: local execution on a Mac for full functionality including voice, and cloud deployment on Fly.io for remote demonstration without the hardware dependent voice stack.

### 10.2 Functional Requirements

The following functional requirements were elicited through iterative use, observed user need patterns and the supervisor's specifications during proposal review.

| ID | Requirement |
|----|-------------|
| **FR1** | The system shall accept natural language input via both text (web UI, CLI) and voice (wake word activated). |
| **FR2** | The system shall classify every incoming request into one of approximately twenty pre defined intents (for example *schedule_meeting*, *get_weather*, *web_search*, *spotify_control*). |
| **FR3** | The system shall decompose complex requests into a directed acyclic graph (DAG) of atomic subtasks with explicit inter subtask dependencies. |
| **FR4** | The system shall execute subtasks in dependency satisfaction order, injecting upstream outputs into downstream parameters via a templated syntax. |
| **FR5** | The system shall invoke external tools spanning at least seven distinct domains: calendar, email, web search, news, weather, music control, and operating system control. |
| **FR6** | The system shall maintain persistent semantic memory using vector embeddings and cosine similarity retrieval, supporting both semantic and episodic memory types. |
| **FR7** | The system shall require explicit user confirmation before performing any externally visible or destructive action (email send, calendar create, file delete). |
| **FR8** | The system shall self critique every generated plan before execution and trigger replanning, bounded to two attempts, when plan quality falls below a configurable threshold. |
| **FR9** | The system shall sustain multi turn flows in which a single user request requires several conversational exchanges (for example, composing an email then requesting a missing recipient address). |
| **FR10** | The system shall permit substitution of the underlying LLM backbone without code changes, supporting both local (Ollama) and cloud (Groq) providers. |
| **FR11** | The system shall persist all task evaluations to disk in a queryable format suitable for benchmark export. |
| **FR12** | The system shall, in local deployment, provide a complete voice loop: wake word detection, speech to text transcription, LLM orchestration, and text to speech synthesis. |
| **FR13** | The system shall deliver a browser accessible single page web interface communicating with the orchestrator over WebSocket. |
| **FR14** | The system shall authenticate to external services via OAuth 2.0 where required, persisting refresh tokens securely between sessions. |
| **FR15** | The system shall provide a benchmark execution mode that exercises a fixed task suite against a configurable list of models and exports the results. |

### 10.3 Non Functional Requirements

| ID | Requirement |
|----|-------------|
| **NFR1, Latency.** | Tier 1 (tool only) requests shall complete in under one second wall clock time. Tier 2 (single LLM hop) requests shall complete in under five seconds. Tier 3 (multi step planned) requests shall complete in under thirty seconds on the reference hardware (M2 MacBook Air, 16 GB RAM). |
| **NFR2, Modularity.** | Each agent and tool shall be implementable, testable and replaceable in isolation, with no inter component coupling beyond the published interfaces. |
| **NFR3, Extensibility.** | Adding a new tool shall require modification of no more than three files (the tool implementation, its registration in the orchestrator, and the planner's tool catalogue) and no architectural changes. |
| **NFR4, Privacy and locality.** | No user content shall be transmitted to third party AI providers in the default local configuration. All embeddings, retrievals and reasoning shall occur on the user's machine. |
| **NFR5, Reliability.** | The system shall tolerate transient external API failures through exponential backoff retry, returning user comprehensible error messages rather than stack traces in the worst case. |
| **NFR6, Observability.** | All agent invocations shall emit human readable log lines identifying the agent, action, latency and outcome, supporting after the fact debugging and benchmark analysis. |
| **NFR7, Portability.** | The system shall run identically on macOS (local) and Linux (cloud container), with the sole exception of the voice subsystem and platform specific operating system control tools, which shall degrade gracefully when unavailable. |
| **NFR8, Security.** | API credentials shall be stored in environment variables or platform secret stores, never in source controlled files. OAuth scopes shall be minimised to the least privileges sufficient for each operation. |

### 10.4 Requirements Traceability

Every requirement above is realised by one or more components described in §11 (Design) and §12 (Development), and the empirical satisfaction of those requirements is examined in §13 (Testing). A traceability table mapping requirements to design components and test scenarios is provided in Appendix D.

---

## §11 Design

### 11.1 Architectural Overview

The architecture follows a **distributed cognitive model** in which assistant behaviour is decomposed across nine specialised agents and seventeen tools, coordinated by a central orchestrator. Each agent encapsulates a single cognitive or operational concern, communicates exclusively through structured data objects defined in `config/models.py`, and is independently instantiable for unit testing. Figure 11.1 (Appendix F) presents the system topology as a directed graph. The principal request flow is Router, Memory, Planner, Critic, Executor, Critic, Evaluator, Memory.

This decomposition reflects a deliberate position: that assistant reliability is achieved through architectural separation, not through more capable models. A monolithic LLM call asked to plan, remember, execute, critique and respond in a single inference pass exhibits the failure modes documented in §8.2. Splitting these concerns across components, each given a narrowly scoped role, removes the cognitive load from any single inference and substitutes deterministic execution where determinism is achievable.

### 11.2 Agent Decomposition

The system instantiates nine specialised agents at startup. The **RouterAgent** classifies each incoming request into an intent and a difficulty *tier* (§11.3), dispatching to the appropriate downstream pipeline. The **PlannerAgent** consumes complex requests and emits structured task DAGs in JSON, using a ReAct derived prompt template (Yao et al., 2022). The **MemoryAgent** wraps ChromaDB with cosine similarity retrieval and exposes `store` and `retrieve` operations to other agents. The **CriticAgent** scores generated plans and execution results, triggering replanning when scores fall below threshold. The **EvaluatorAgent** persists per task benchmark records to SQLite, supporting offline analysis. The **SummariserAgent** condenses long outputs (research results, document extracts) into concise prose. Three domain specific agents (**CalendarAgent**, **GmailAgent** and **FinExAgent**) wrap structurally complex external services with multi step internal logic such as event conflict checking, OAuth token refresh, PDF extraction and SQL generation.

Each agent is implemented as a Python class with an `async` public interface and an injectable LLM client. This permits a single shared `OllamaClient` to be used across all agents in production, while supporting per agent mocking during testing. The agent inventory is summarised in Table 11.1.

*Table 11.1, Agent inventory.*

| Agent | Responsibility | LLM using |
|---|---|---|
| Router | Intent and tier classification | Yes (small model) |
| Planner | Task decomposition into DAG | Yes |
| Memory | Semantic retrieval and storage | Embeddings only |
| Critic | Plan and result quality review | Yes |
| Evaluator | Benchmark persistence | No |
| Summariser | Output condensation | Yes |
| Calendar | Google Calendar operations | No |
| Gmail | Gmail operations | Hybrid (drafting only) |
| FinEx | Financial PDF analysis | Yes |

### 11.3 Tier Based Routing

A distinctive feature of Jarvis is **tier based routing**, which adapts pipeline depth to task complexity rather than running every request through the full planner. Three tiers are defined.

**Tier 1, tool only.** Requests whose intent maps unambiguously to a single deterministic tool call (*"weather today,"* *"play focus music,"* *"set volume to 40"*) bypass the planner entirely. The Router emits a tier 1 RouterDecision and the orchestrator's `_try_shortcut` handler dispatches directly to the relevant tool. Median latency: under 800 ms.

**Tier 2, single LLM hop.** Requests requiring one LLM inference plus a tool call (factual questions answered via web search, draft email composition, simple conversation) are handled in a streamlined two step pipeline that skips the full planner and critic loop. Median latency: 2 to 5 seconds.

**Tier 3, full DAG planning.** Multi step requests (*"check my calendar, summarise unread emails, then book a meeting with Alex"*) traverse the complete Router, Memory, Planner, Critic, Executor, Critic, Evaluator pipeline. Median latency: 10 to 30 seconds depending on model and subtask count.

The classification itself is hybrid. A fast deterministic pre router applies thirty plus regex patterns covering the most common request shapes (calendar creation, email send and inbox, reminders, weather, factual questions) and returns a RouterDecision in under one millisecond when a pattern matches. Requests escaping the deterministic layer fall through to a smaller LLM (`llama3.2:1b` by default) classifying intent and tier in a single approximately 80 token inference. This two layer routing reduces total router latency by an order of magnitude compared to LLM only classification.

Conceptually, the tier system is the operational response to what proved to be a core empirical finding of this work: in multiagent LLM systems, there is a measurable tension between coordination richness and response efficiency. Early prototype runs of trivially routable requests through the full Planner, Critic, Executor pipeline regularly incurred latencies of thirty to forty five seconds. The outcomes were high quality, but the user experience cost was unacceptable. Shortcutting all routing to deterministic rules would, conversely, sacrifice the system's reasoning capacity for speed. Tier based routing is the principled middle path: pay the orchestration cost only when the task warrants it. This finding is revisited in §14 as one of the project's intended scholarly contributions.

### 11.4 Task Plans and Dependency Injection

The Planner emits task plans as JSON objects conforming to a schema defined in `config/models.py`. Each plan comprises an `intent` field, a free text `reasoning` trace (the ReAct rationale), and an ordered list of `subtasks`. Each subtask carries an `id`, the executing `agent`, an `action`, a `params` dictionary, and an explicit `depends_on` array referencing the IDs of subtasks whose outputs are required.

The depends on field encodes a directed acyclic graph. The orchestrator's execution algorithm resolves this graph iteratively, conceptually similar to topological sorting. At each step it identifies subtasks whose dependencies have completed, executes those concurrently where possible, stores their outputs, and continues until all subtasks have run or a circular dependency limit (twenty iterations) is reached. Circular dependencies, occasionally produced by smaller models, are detected and surfaced as plan quality errors to the Critic.

A particular implementation detail merits attention: **templated parameter injection**. The Planner is prompted to emit symbolic references such as `{subtask_1.result.preferred_time}` in downstream subtask parameters. The executor resolves these references against upstream outputs at execution time. This pattern decouples the Planner from concrete data values, permits parameter values to be computed by upstream subtasks (a memory lookup providing a recipient address, for instance), and allows the same plan template to be reused across distinct invocations with different runtime values.

### 11.5 Memory Subsystem Design

The memory subsystem is implemented as a thin wrapper around ChromaDB configured for cosine similarity. Every memory entry, whether **semantic** (a stored fact or preference) or **episodic** (a record of a past task outcome), is embedded using the configured embedding model (`nomic embed text` locally; `all MiniLM L6 v2` in cloud), persisted to disk in ChromaDB's HNSW indexed store, and tagged with metadata identifying its type, creation time, and originating context. Retrieval is parameterised by query string, optional memory type filter, top k count and a similarity threshold. Matches below threshold are silently discarded to prevent low relevance noise from polluting downstream reasoning. An eight second timeout guards against embedding model stalls, with deterministic hash based fallback embedding used in the unlikely event of LLM unavailability.

### 11.6 Critic and Replanning Loop

The Critic provides a metacognitive layer over the orchestrator's main pipeline. After plan generation but before execution, the Critic reviews the plan against five criteria: request alignment, dependency correctness, completeness, parameter sensibility, and over engineering. It emits a JSON verdict with a 0 to 1 score, an issues list and a `replan_needed` flag. When `replan_needed` is true, the orchestrator re invokes the Planner with the Critic's feedback inlined into the prompt, up to a maximum of two replanning attempts. The same Critic also reviews execution results post hoc, with verdict scores feeding into the Evaluator's planning_score metric. This loop is the system's primary defence against the most common LLM failure mode: plausible but flawed plans being executed unchallenged.

### 11.7 Tool Abstraction Layer

External integrations are encapsulated in seventeen Python modules under `tools/`, each exposing an async interface compatible with the Executor's invocation protocol. Tools are categorised by dependency profile. **Pure Python tools** require no external authentication (markets, weather, news). **Authenticated tools** use OAuth (Calendar, Gmail, Spotify). **Platform bound tools** require macOS (`file_manager`, `mac_control`). **Locally hosted tools** include reminders persisted to SQLite. A `platform_guard` module detects the runtime environment and returns user comprehensible messages when platform bound tools are invoked in incompatible contexts, for example, macOS volume controls in the Linux cloud container.

### 11.8 Voice Pipeline Design

The voice subsystem implements a four stage pipeline: wake word detection, voice activity detection, speech to text, orchestrator invocation, and text to speech. Wake detection runs the openWakeWord ONNX model on continuous microphone input, costing approximately five milliseconds per 80 ms audio frame. Once triggered, Silero VAD identifies utterance boundaries, faster whisper (`small.en`, int8 quantised) transcribes the audio at approximately five times realtime on the reference hardware, and the resulting text enters the orchestrator's normal request path. The orchestrator's response is then synthesised via ElevenLabs' `eleven_flash_v2_5` voice model. This is a deliberate departure from the local first principle, justified by the absence of an open source TTS model with comparable naturalness and time to first audio. Barge in is supported: the wake detector remains active during TTS playback, enabling mid response interruption.

---

## §12 Development

### 12.1 Technology Stack

The implementation uses Python 3.10+ as its primary language. The choice was made for its mature `asyncio` runtime, broad library ecosystem for LLM and vector database work, and the candidate's prior fluency. Asynchronous I/O is used throughout. Every agent invocation, every LLM call, every external tool integration is `async` native, which permits the orchestrator to issue concurrent independent subtask calls without thread overhead.

The choice of **Ollama** as the local LLM runtime was made early and held throughout. Ollama provides a consistent HTTP and streaming interface across model families, manages model weights and quantisation transparently, and exposes both chat and embedding endpoints required by the orchestrator and MemoryAgent respectively. **ChromaDB** was selected over alternatives such as FAISS (Johnson et al., 2017) and pgvector primarily for its in process Python API, persistence handling out of the box, and HNSW indexing without configuration. **FastAPI** was chosen for the web layer for its WebSocket support, automatic OpenAPI documentation, and async first programming model. The full dependency manifest, segregated into core (`requirements.txt`) and voice only (`voice_requirements.txt`) sets, is reproduced in Appendix E.

### 12.2 Development Environment and Process

Development took place in Visual Studio Code on macOS within a Git tracked monorepo. Branches were not used for most of the project's lifespan. Commits were made directly to `main` at a rate of roughly one substantive commit per work session, a deliberate choice consistent with the single developer context, where branch coordination overhead is unjustified. A virtual environment isolated approximately forty direct Python dependencies. End to end smoke testing was performed manually after each significant change. Targeted unit testing was applied to particularly failure prone components such as the Planner's JSON parsing and the dependency resolution algorithm.

### 12.3 Key Implementation Challenges

Six implementation challenges shaped the project's trajectory.

**JSON output consistency** proved the single largest source of friction in early iterations. The Planner is required to emit valid JSON conforming to a specific schema. In practice, smaller models occasionally produced markdown wrapped JSON, JSON with trailing prose, or syntactically malformed objects. This was addressed in three layers: a low temperature sampling configuration (`temperature=0.1`) reducing output entropy; a system prompt with explicit *"Output MUST be valid JSON only"* instructions and an annotated schema example; and post processing routines that strip markdown fences, trim trailing whitespace, and attempt JSON repair before parsing fails.

**Async tool execution** required care to avoid both blocking the event loop and creating unbounded concurrency. The orchestrator uses `asyncio.gather` to execute independent subtasks concurrently while respecting dependencies. Synchronous third party libraries, notably some Google API client routines, are wrapped in `loop.run_in_executor`. The `OllamaClient` implements its own connection pooling and a maximum in flight request count to prevent benchmark runs from saturating the local model server.

**OAuth integration** with Google services proved more involved than its documentation suggested. The standard `google auth oauthlib` flow opens a browser tab during initial authorisation, which is fine for an interactive desktop application but incompatible with cloud deployment. The solution was to authorise once locally, persist the refresh token to `token.json`, and inject this credential into the cloud container as a Fly secret. Token refresh is handled transparently by the Google client library.

**Replan loop logic** required careful state management. Naïve replanning, which is simply asking the LLM to try again, yields plans only marginally different from the original. The implementation instead inlines the Critic's specific feedback into the second attempt prompt, prefixed with the original plan and an annotation of its identified flaws. Empirically this produces qualitatively different plans on the second attempt rather than minor variations.

**Voice latency** was the dominant user experience constraint in the voice pipeline. The wake word to first audio path comprises eight discrete stages contributing latency at every step. Optimisation focused on the longest contributors. The Router's deterministic pre pass eliminates LLM classification latency for most common requests. STT was downgraded from `medium.en` to `small.en` with int8 quantisation, halving inference time at modest accuracy cost. The ElevenLabs Flash voice model was selected specifically for its sub 100 ms time to first audio.

**Local model variance** was a continuous low grade frustration. Models within the Llama 3 family, even closely related variants such as `llama3.2:latest` and `llama3.2:1b`, exhibit measurably different prompt following behaviour and JSON reliability. The system was designed throughout to assume that any given LLM call may fail or return malformed output, and to degrade gracefully through fallback paths in each case.

### 12.4 Selected Code Highlights

Three code regions are excerpted below as representative of the implementation. Full source for these and the remaining agents is reproduced in Appendix E.

The **Planner's system prompt** anchors the entire reasoning pipeline. Its design balances explicitness against context length. Empirically, the smaller models in use reproduce the schema more reliably when given a concrete JSON example than when given an abstract description.

```python
# agents/planner.py, abridged
SYSTEM_PROMPT = """You are the Planner agent for Jarvis.
Decompose user requests into structured, executable subtasks.

Use ReAct reasoning:
1. THOUGHT:      What does the user actually want?
2. OBSERVATION:  What context and memories are relevant?
3. THOUGHT:      What are the atomic steps needed?
4. OUTPUT:       Structured JSON plan

Output MUST be valid JSON:
{
  "intent":    "schedule_meeting",
  "reasoning": "Step-by-step thought process...",
  "subtasks": [
    {"id": "subtask_1", "agent": "memory",
     "action": "retrieve_context",
     "params": {"query": "user meeting preferences"},
     "depends_on": []},
    {"id": "subtask_2", "agent": "calendar",
     "action": "create_event",
     "params": {"start_time": "{subtask_1.result.preferred_time}"},
     "depends_on": ["subtask_1"]}
  ]
}

Rules:
- Use {subtask_id.result.field} for dependency injection
- Keep subtasks atomic, one action per subtask
- Use ISO 8601 for all datetimes
- Maximum 8 subtasks per plan
"""
```

The **Router's deterministic pre pass** is a sequence of regular expressions that short circuit the LLM based classifier for the common case. The catalogue of patterns is not speculative. Each was added in response to a routing failure observed during informal use, and is the most explicit case where the project's iterative methodology produced a measurable engineering artefact.

```python
# agents/router.py, abridged
def _deterministic_route(self, req: str) -> Optional[RouterDecision]:
    """Sub-millisecond routing for common request shapes."""
    r = req.lower().strip()

    _cal_create = re.compile(
        r'\b(schedule|book|create|add|set|make)\b.{0,30}'
        r'\b(meeting|appointment|event|call|session)\b',
        re.I,
    )
    _email_send = re.compile(
        r'\b(send|write|draft|email)\b.*\b(email|message)\b',
        re.I,
    )
    _factual_q = re.compile(
        r'^\s*(who|what|when|where|why|how|which)\b'
        r'.*\b(is|are|was|were|did|does|do|will)\b',
        re.I,
    )

    if _cal_create.search(r):
        return RouterDecision(primary_agent=AgentRole.CALENDAR,  tier=1)
    if _email_send.search(r):
        return RouterDecision(primary_agent=AgentRole.EMAIL,     tier=1)
    if _factual_q.search(r):
        return RouterDecision(primary_agent=AgentRole.WEBSEARCH, tier=2)
    # thirty further patterns follow
    return None  # fall through to LLM router
```

The **Critic agent's verdict structure** demonstrates the metacognitive loop. The agent reviews each plan against five criteria and emits a JSON verdict consumed by the orchestrator. Below threshold verdicts trigger bounded replanning with the Critic's feedback inlined into the second attempt prompt.

```python
# agents/critic.py, abridged
SYSTEM_PROMPT = """You are the Critic agent. Review task plans
for quality and correctness.

When reviewing a PLAN, check for:
- Does the plan actually address the user's request?
- Are subtask dependencies correctly ordered?
- Are there missing steps (e.g. no memory retrieval)?
- Are the params complete and sensible?
- Is the plan over engineered for a simple task?

Respond with valid JSON only:
{
  "approved":      true,
  "score":         0.85,
  "issues":        ["issue 1", "issue 2"],
  "suggestions":   ["suggestion 1"],
  "replan_needed": false
}

Rules:
- score in 0.0 to 1.0
- approved=true if score >= 0.6
- replan_needed=true only for score < 0.5
- Be specific about issues, vague feedback is useless
"""
```

### 12.5 Evolution and Refactoring Narrative

The system as it stands does not resemble its earliest form. The first prototype, written in late April 2026, was a single 200 line script combining router, planner and execution in one function. The decomposition into discrete agents emerged organically over the following weeks, driven by repeated experience of the monolith's brittleness. A change to the planning prompt would silently break the routing logic. A tool integration would require modifying parsing code far from where the tool was used.

The tier based routing concept emerged from a specific observation. Running the full Planner, Critic, Executor pipeline for a simple *"what's the weather"* request consumed many seconds of LLM inference to produce a result a hardcoded rule could deliver in under one second. The shortcut handler was introduced first, the deterministic regex pre router second, and the formal three tier classification third. By then the original prototype's monolithic flow had been almost entirely supplanted by tier aware dispatch.

Several components went through significant refactoring. The Critic was initially invoked only on plans, not on results. Adding post execution review caught a class of failures (the executor reporting success on a syntactically correct but semantically empty result) that earlier versions had silently passed through to the user. The MemoryAgent moved from in memory cosine search to ChromaDB persistence after the second time the project had to be restarted and the candidate observed that no memory persisted across sessions. The voice subsystem was a late addition, integrated in mid May 2026 after the core orchestrator was already mature, and required *no* architectural changes, a useful empirical validation of NFR3 (extensibility).

---

## §13 Testing

### 13.1 Testing Strategy

Testing combined automated benchmark execution against a fixed task suite with manual scenario based testing of edge cases and failure modes. Automated testing exercises the orchestrator's full pipeline. Manual testing validates specific tool integrations, multi turn flows and voice pipeline behaviour that benchmark automation cannot easily reproduce. All quantitative data presented in this section was generated by the EvaluatorAgent and persisted to `data/jarvis.db`. The raw export is included as Appendix D.

### 13.2 Benchmark Suite

The benchmark suite comprises nine representative tasks spanning the three tiers introduced in §11.3 and seven of the system's intent categories. Tasks were selected to represent the breadth of practical use rather than to maximise difficulty. Tier 1 tasks include *"What's the weather today?"*, *"Get me the top BBC headlines,"* and *"Find me a focus music playlist on Spotify and set volume to 60."* Tier 2 includes *"Search for information about ReAct prompting framework"* and similar single LLM hop queries. Tier 3 includes *"Search the web for the latest AI news and summarise the top three stories"*, *"Get the news, check the weather, and give me a morning briefing,"* and *"Open my notes app and set a reminder to check emails in 30 minutes,"* each requiring multi step DAG planned execution. The full task list is reproduced in Appendix D.

### 13.3 Quantitative Results

The benchmark was designed to be exercised across four locally hosted models: `llama3.2:1b`, `llama3.2:3b` (the default chat backbone), `phi3:mini` and `gemma:2b`. These were chosen to span the 1 to 3 billion parameter range where local CPU inference remains practical on the reference hardware. The selection of multiple small models, rather than chasing larger or more capable variants, was a deliberate response to the project's local first commitment. The systems most likely to be deployed on consumer hardware are precisely those whose performance is least characterised in the published literature. The principal dataset presented here aggregates sixteen evaluations against `llama3.2:latest` (the system's default), summarised in Table 13.1. Cross model comparison and a local versus cloud (Groq Llama 3.3 70B) study are flagged in §15.4 as the highest value extension to the evaluation.

*Table 13.1, Aggregate benchmark metrics, `llama3.2:latest`, n = 16.*

| Metric | Mean | Range |
|---|---|---|
| Overall score | 0.83 | 0.76 to 0.96 |
| Planning score | 0.84 | 0.40 to 0.90 |
| Execution score | 0.78 | 0.57 to 1.00 |
| Latency (ms) | 128,500 | 48,394 to 182,794 |
| Subtask count | 3.6 | 1 to 8 |
| Replan count | 0.56 | 0 to 2 |
| Tasks passing threshold (score ≥ 0.7) | 16 of 16 (100%) |   |

A number of observations follow. First, **execution score is consistently lower than planning score** (0.78 versus 0.84). The system is, on average, better at producing high quality plans than at executing every subtask to completion. The gap is dominated by transient external API failures (Google Calendar in particular) and by occasional subtask hallucinations the Critic permits through.

Second, **latency is dominated by LLM inference time** rather than tool execution. The fastest evaluation (48 seconds) was an email updates task with three subtasks. The slowest (183 seconds) was a single subtask scheduling task that required two replans. The high variance reflects the locally hosted model's sensitivity to prompt length and the cumulative cost of the Critic and replan stages on tier 3 requests.

Third, **replanning is invoked on roughly one task in three** (mean replan_count = 0.56), and the same task may be replanned more than once. Of the sixteen runs, six produced at least one replan and two produced two replans. Tasks requiring replans took noticeably longer but their final scores converged toward similar values, evidence that the replan loop achieves its quality improvement goal at meaningful but not prohibitive latency cost.

Tier wise latency analysis (Table 13.2) confirms the design rationale of §11.3. Tier 1 deterministic shortcuts cost a small fraction of the latency of tier 3 DAG planned execution.

*Table 13.2, Median latency by tier (observed, reference hardware).*

| Tier | Description | Median latency |
|---|---|---|
| 1 | Deterministic tool only | approximately 800 ms |
| 2 | Single LLM hop | approximately 3,500 ms |
| 3 | Full DAG planned execution | approximately 95,000 ms |

### 13.4 Failure Mode Analysis

Five distinct failure modes were catalogued. **JSON parsing failure**, in which the Planner emits non conforming output, was the most common in early development but is now rare. Residual cases trigger automatic replanning. **Dependency injection mismatch**, where the Planner references a result field that the upstream subtask does not produce, occurs sporadically and is now caught by an executor side validation step that surfaces a clear error message to the Critic. **External API rate limiting** affects Google Calendar and Spotify under sustained benchmark runs and is mitigated through exponential backoff. **Temporal parsing errors**, where natural language dates such as "next Wednesday at 3" occasionally resolve to the wrong calendar week, were partially addressed by switching from a regex based to a `python dateutil` based parser but remain a known residual weakness. **Voice mis trigger**, where the wake word model false positively activates on incidental speech, was reduced by raising the wake threshold from 0.5 to 0.6, with no measurable degradation in true positive rate.

### 13.5 Threats to Validity

Several threats to the validity of these results warrant explicit acknowledgement. The benchmark task set, while curated to represent common use, is small (n = 9 tasks) and was authored by the developer. It cannot claim to represent the full distribution of personal assistant requests. The evaluation runs to date are concentrated on a single model. Broader model comparison and a larger task set are identified as priorities for further work (§15). Latency measurements were taken on a single hardware configuration (M2 MacBook Air, 16 GB RAM). Performance on other hardware is uncharacterised. The Critic's scoring is itself an LLM output and may carry its own biases. An inter rater reliability study comparing Critic scores with human judgement is identified as a useful future direction. These limitations do not undermine the central observation that the modular architecture functions as designed. They bound the claims that can responsibly be drawn from the empirical evidence presented.

---

## §14 Conclusions

The principal claim of this dissertation, advanced in the abstract and developed throughout, is that the limitations of contemporary LLM based personal assistants are largely architectural rather than modelling. The construction and evaluation of Jarvis offers concrete evidence for this position. With a small, locally hosted model (`llama3.2:latest`, approximately three billion parameters), the system achieves a mean overall benchmark score of **0.83** across nine representative tasks spanning three difficulty tiers, with 100% of evaluated tasks passing the 0.7 quality threshold and a replan driven self correction loop operating on roughly one task in three. These results are produced not by a larger or more capable model (Llama 3.2 is, in absolute terms, one of the smallest competitive open weight models available) but by a structured decomposition of assistant behaviour across nine specialised agents and a tier aware routing layer that calibrates pipeline depth to task complexity.

Three architectural contributions are advanced. First, **tier based routing** demonstrates that the cost of multiagent orchestration can be paid selectively rather than uniformly. Median latencies range from sub second on tier 1 deterministic shortcuts to tens of seconds on tier 3 DAG planned execution, with no architectural change required to support the full spectrum. Second, **templated parameter injection** in the task plan schema (`{subtask_1.result.field}`) decouples the planner from concrete data values, enabling reusable plan templates and clean upstream to downstream data flow. Third, the **critic driven replanning loop** provides a metacognitive review step that empirically reduces low quality outputs at modest latency cost.

A specific empirical finding emerging from the project warrants explicit elevation. The latency versus coordination tension first documented in §11.3 (full pipeline execution of trivially routable requests producing high quality outcomes at prohibitive latency cost) proved to be a quantifiable property of multiagent LLM systems, not a peculiarity of this particular implementation. Tier based routing is the operational response, and its existence as a tunable architectural parameter rather than a fixed pipeline depth is the project's most directly transferable scholarly contribution.

Several limitations have been disclosed with equal candour. The empirical evaluation is concentrated on a single model and a small benchmark suite. No formal user study was conducted within the project's timeframe. Memory retrieval relies on cosine search that would not scale beyond a single user prototype. The proposal's promised single agent baseline was honestly reframed mid project rather than retrofitted into an unconvincing comparison. These limitations are reflected in the recommendations for further work that follow.

The broader implication of this work is not that Jarvis itself is the answer to personal AI, but that the architectural patterns it instantiates (agent decomposition, tier aware dispatch, dependency aware planning, metacognitive critique, and local first deployment) together constitute a reproducible approach to building LLM based assistants that are reliable enough for daily personal use. The bottleneck in personal AI is, on the evidence of this project, less in the underlying model than in the architecture that surrounds it.

---

## §15 Recommendations for Further Work

The work documented above is a complete and functional system. It is not a finished one. Five directions for further development are identified, ordered roughly by tractability.

**1. Formal schema validation and DAG verification.** The Planner's JSON output is currently parsed leniently, with malformed plans triggering replanning. Adopting Pydantic based schema validation, alongside an explicit topological sort step over the dependency graph, would surface plan quality issues earlier and produce more actionable feedback to the Critic. This is the most immediately tractable improvement and would likely repay perhaps a week of engineering.

**2. Production scale memory indexing.** The MemoryAgent's cosine retrieval is implemented through ChromaDB's HNSW index, but the surrounding query logic (particularly metadata filtering and result set reranking) would benefit from refinement. At the scale of a single user the current performance is adequate. Multi user or longer horizon deployments would require profiling and, potentially, migration to a dedicated vector database such as Pinecone, Weaviate or pgvector.

**3. User study.** The original proposal committed to a System Usability Scale (SUS) questionnaire and qualitative interviews with users beyond the candidate. Time and scope constraints precluded this within the project window. A small mixed methods study, with five to ten participants using Jarvis for one week, structured SUS scoring, semi structured exit interviews and consent processes already drafted (Appendix C), would add a substantive empirical layer addressing reliability and trust as perceived by genuine users rather than the developer.

**4. Multi model and larger scale benchmarking.** The evaluation reported in §13 is concentrated on a single model. The benchmark infrastructure is already in place to run a broader study comparing locally hosted `llama3.2:1b`, `llama3.2:3b`, `phi3:mini` and `gemma:2b` against cloud hosted Groq Llama 3.3 70B as a high capacity reference. Such a study would substantively characterise how Jarvis's architectural decisions interact with model scale and prompt following capability, and would directly quantify the local versus cloud privacy performance trade off that is currently asserted rather than measured.

**5. Closed loop self improvement.** A more ambitious direction follows the Voyager line (Wang et al., 2023b). The system would store successful plan templates as a *skill library*, retrieve them on subsequent similar requests, and abstract recurring task patterns into named procedures. This would convert the episodic memory into a form of procedural memory and provide a substrate for genuine learning over time, taking the system meaningfully beyond static prompted behaviour.

Two additional improvements would benefit production deployment specifically. The voice subsystem's cloud variant requires WebRTC based microphone capture in the browser. The present implementation supports only local microphone input. Concurrent multi user support would require revisiting the orchestrator's currently per user state assumptions and introducing proper session isolation. Neither is conceptually difficult, but both would substantially expand the practical scope of the system.

---

## Software Artefact Download

The Jarvis source code is available at the candidate's repository. Deployment instructions are provided in `DEPLOY.md`. Voice specific setup is documented in `VOICE_SETUP.md`. A cloud hosted demonstration instance is available at the deployment URL provided alongside this submission.

---

## Glossary

**DAG**, Directed Acyclic Graph. A graph structure in which nodes (subtasks) are connected by edges (dependencies) such that no cycles exist. Used in Jarvis to represent task plans.

**DSR**, Design Science Research. A research methodology centred on the iterative construction and evaluation of technological artefacts as a legitimate form of scholarly contribution (Hevner et al., 2004).

**FinEx**, Financial Extraction subsystem. The Jarvis component responsible for parsing financial statement PDFs into structured data and answering questions over them via SQL.

**HNSW**, Hierarchical Navigable Small World. An approximate nearest neighbour indexing algorithm used by ChromaDB for fast vector retrieval.

**LLM**, Large Language Model. A neural network of (typically) billions of parameters trained on large text corpora.

**MAS**, Multiagent System. A software architecture in which behaviour is decomposed across multiple cooperating agents, each with a defined responsibility.

**OAuth 2.0**, An authorisation framework permitting third party applications to access user resources without exposing user credentials. Used for Google Calendar and Gmail integration.

**Ollama**, A local LLM runtime that provides a consistent HTTP interface across model families, used by Jarvis for default local inference.

**RAG**, Retrieval Augmented Generation. A pattern pairing an LLM with an external retrieval system.

**ReAct**, *Reason and Act*. A prompting paradigm interleaving explicit reasoning steps and tool invoking actions (Yao et al., 2022).

**STT**, Speech to Text. Jarvis uses faster whisper for local STT.

**Tier (Tier 1 / 2 / 3)**, A Jarvis specific routing classification adapting pipeline depth to task complexity, introduced in §11.3.

**ToT**, Tree of Thoughts. A generalisation of CoT exploring multiple reasoning paths (Yao et al., 2023).

**TTS**, Text to Speech. Jarvis uses ElevenLabs' Flash v2.5 model.

**VAD**, Voice Activity Detection. Jarvis uses Silero VAD.

**WebSocket**, A full duplex protocol used by Jarvis's web UI to stream orchestrator responses to the browser.

---

## References

Beurer Kellner, L., Fischer, M. and Vechev, M., 2024. *Prompting Is Programming: A Query Language for Large Language Models.* Proceedings of the ACM on Programming Languages.

Brown, T. et al., 2020. *Language Models are Few Shot Learners.* Advances in Neural Information Processing Systems, 33.

Chen, M. et al., 2021. *Evaluating Large Language Models Trained on Code.* arXiv:2107.03374.

Durrani, A.K., 2025. *Multi Agent AI Executive Assistant: Project Proposal.* Buckinghamshire New University, unpublished.

Gao, Y. et al., 2024. *Retrieval Augmented Generation for Large Language Models: A Survey.* arXiv:2312.10997.

Hao, S. et al., 2023. *Reasoning with Language Model is Planning with World Model.* arXiv:2305.14992.

Hevner, A.R., March, S.T., Park, J. and Ram, S., 2004. *Design Science in Information Systems Research.* MIS Quarterly, 28(1), pp. 75 to 105.

Hong, S. et al., 2023. *MetaGPT: Meta Programming for Multi Agent Collaborative Framework.* arXiv:2308.00352.

Jégou, H., Douze, M. and Schmid, C., 2011. *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI, 33(1), pp. 117 to 128.

Johnson, J., Douze, M. and Jégou, H., 2017. *Billion scale similarity search with GPUs.* arXiv:1702.08734.

Karpukhin, V. et al., 2020. *Dense Passage Retrieval for Open Domain Question Answering.* Proceedings of EMNLP 2020.

Khattab, O. and Zaharia, M., 2020. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* Proceedings of SIGIR 2020.

Kojima, T. et al., 2022. *Large Language Models are Zero Shot Reasoners.* NeurIPS 35.

LangChain Inc., 2024. *LangGraph: Building Stateful Multi Agent Applications.* Available at: https://langchain ai.github.io/langgraph/ [Accessed 19 May 2026].

Lewis, P. et al., 2020. *Retrieval Augmented Generation for Knowledge Intensive NLP Tasks.* NeurIPS 33.

Liu, X. et al., 2023a. *LLM+P: Empowering Large Language Models with Optimal Planning Proficiency.* arXiv:2304.11477.

Liu, X. et al., 2023b. *AgentBench: Evaluating LLMs as Agents.* arXiv:2308.03688.

Malkov, Y.A. and Yashunin, D.A., 2018. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI, 42(4), pp. 824 to 836.

Moura, J., 2024. *CrewAI: Framework for Orchestrating Role Playing Autonomous AI Agents.* Available at: https://www.crewai.com/ [Accessed 19 May 2026].

Nakajima, Y., 2023. *BabyAGI.* GitHub. Available at: https://github.com/yoheinakajima/babyagi [Accessed 19 May 2026].

OpenAI, 2022. *Introducing ChatGPT.* Available at: https://openai.com/blog/chatgpt [Accessed 19 May 2026].

OpenAI, 2023. *Function calling and other API updates.* Available at: https://openai.com/blog/function calling and other api updates [Accessed 19 May 2026].

Packer, C. et al., 2023. *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560.

Park, J.S. et al., 2023. *Generative Agents: Interactive Simulacra of Human Behavior.* Proceedings of UIST 2023.

Patil, S.G., Zhang, T., Wang, X. and Gonzalez, J.E., 2023. *Gorilla: Large Language Model Connected with Massive APIs.* arXiv:2305.15334.

Peffers, K., Tuunanen, T., Rothenberger, M.A. and Chatterjee, S., 2007. *A Design Science Research Methodology for Information Systems Research.* Journal of Management Information Systems, 24(3), pp. 45 to 77.

Qin, Y. et al., 2023. *ToolLLM: Facilitating Large Language Models to Master 16000+ Real world APIs.* arXiv:2307.16789.

Reimers, N. and Gurevych, I., 2019. *Sentence BERT: Sentence Embeddings using Siamese BERT Networks.* Proceedings of EMNLP 2019.

Richards, T.B., 2023. *AutoGPT.* GitHub. Available at: https://github.com/Significant Gravitas/AutoGPT [Accessed 19 May 2026].

Schick, T. et al., 2023. *Toolformer: Language Models Can Teach Themselves to Use Tools.* NeurIPS 36.

Shinn, N. et al., 2023. *Reflexion: Language Agents with Verbal Reinforcement Learning.* arXiv:2303.11366.

Wang, X. et al., 2023a. *Self Consistency Improves Chain of Thought Reasoning in Language Models.* Proceedings of ICLR 2023.

Wang, G. et al., 2023b. *Voyager: An Open Ended Embodied Agent with Large Language Models.* arXiv:2305.16291.

Wei, J. et al., 2022. *Chain of Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 35.

Weizenbaum, J., 1966. *ELIZA, A Computer Program for the Study of Natural Language Communication between Man and Machine.* Communications of the ACM, 9(1), pp. 36 to 45.

Wu, Q. et al., 2023. *AutoGen: Enabling Next Gen LLM Applications via Multi Agent Conversation.* arXiv:2308.08155.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. and Cao, Y., 2022. *ReAct: Synergizing Reasoning and Acting in Language Models.* arXiv:2210.03629.

Yao, S. et al., 2023. *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* arXiv:2305.10601.

Zhao, A. et al., 2024. *Expel: LLM Agents Are Experiential Learners.* Proceedings of AAAI 2024.

Zhou, S. et al., 2024. *WebArena: A Realistic Web Environment for Building Autonomous Agents.* Proceedings of ICLR 2024.

---

## Bibliography

The following sources informed the candidate's thinking during the project's development but were not directly cited in the body of this report.

- Russell, S. and Norvig, P., 2020. *Artificial Intelligence: A Modern Approach.* 4th ed. Pearson.
- Anthropic, 2024. *Building Effective Agents.* Available at: https://www.anthropic.com/research/building effective agents [Accessed 19 May 2026].
- Sutton, R.S. and Barto, A.G., 2018. *Reinforcement Learning: An Introduction.* 2nd ed. MIT Press.
- LangChain Documentation, 2024 to 2026. Available at: https://docs.langchain.com/ [Accessed 19 May 2026].
- Ollama Project Documentation, 2024 to 2026. Available at: https://ollama.com/ [Accessed 19 May 2026].

---

## Appendix A: Project Plan

The project was structured into seven phases aligned with the BNU COM6001 calendar.

| Phase | Window | Activity |
|---|---|---|
| 1 | Dec 1 to Dec 20, 2025 | Topic finalisation, supervisor scoping, proposal outline |
| 2 | Jan 3 to Feb 4, 2026 | Literature review, gap analysis |
| 3 | Feb 5 to Mar 4, 2026 | Methodology, architecture design, interim report submission (4 Mar) |
| 4 | Mar 5 to Apr 10, 2026 | Implementation: agents, tools, orchestrator |
| 5 | Apr 11 to May 1, 2026 | Testing, benchmarking, refinement |
| 6 | May 2 to May 15, 2026 | Final report drafting, polishing, Turnitin pre check |
| 7 | May 16 to May 20, 2026 | Final submission preparation, submission 20 May 2026 |

Actual deviations from this plan were minor. The principal divergence was scope expansion within Phase 4 (additional tool integrations including voice, Spotify and the FinEx subsystem in place of the originally proposed Notion integration) and a methodological reframing of the evaluation strategy in early Phase 5, both discussed in §9.2.

---

## Appendix B: Ethics Checklist

| Item | Response |
|---|---|
| Does the project involve human participants? | Limited informal testing only; no formal study within this submission. |
| Will personal data be collected, stored or analysed? | No participant data was collected for the dissertation evaluation; the candidate's own data was used during development. |
| Could participants be identified from the project outputs? | N/A, no participants. |
| Will participants be informed of the purpose, risks and right to withdraw? | A consent form (Appendix C) has been prepared for any future user study (§15.3). |
| Is the project free from any potential physical or psychological harm? | Yes. |
| Are data protection requirements (UK GDPR) met? | All processing is local; no third party transmission of personal data in default configuration. |
| Has appropriate ethical approval been obtained where required? | Not required for the present scope. |

---

## Appendix C: Participant Consent Form (drafted for future user study)

> **Title:** Evaluation of Jarvis, a Multiagent AI Executive Assistant
>
> **Researcher:** Abdullah Khan Durrani, Buckinghamshire New University.
>
> **Invitation.** You are invited to participate in an evaluation study of *Jarvis*, an experimental personal AI assistant developed as a final year project at BNU.
>
> **What will I be asked to do?** You will be asked to use Jarvis for daily personal organisation tasks over a one week period, then complete a short System Usability Scale (SUS) questionnaire and participate in a brief (15 to 20 minute) interview about your experience.
>
> **What data will be collected?** SUS scores, interview audio (transcribed and anonymised), and (with your permission) logs of your interactions with Jarvis. No identifying personal data will be retained beyond the duration of the study.
>
> **Withdrawal.** You may withdraw at any time without giving a reason and without penalty.
>
> **Confidentiality.** All data will be anonymised. Results may be reported in academic publications but no individual participant will be identifiable.
>
> **Consent.** *I have read the above information and consent to participate.*
>
> Signed: ______________  Date: ______

---

## Appendix D: Benchmark Results (raw export)

Full raw evaluation data, sixteen runs against `llama3.2:latest` across the nine task benchmark suite, with `task_id, model, intent, success, score, planning_score, execution_score, latency_ms, subtask_count, replan_count, feedback, timestamp` per row, is reproduced from `data/benchmark_results.csv`. The aggregate statistics in §13.3 are computed from this dataset.

The benchmark task list is reproduced below:

1. *"What's the weather today?"* (Tier 1, get_weather)
2. *"Get me the top BBC headlines"* (Tier 1, get_news)
3. *"Search for information about ReAct prompting framework"* (Tier 2, web_search)
4. *"Search the web for the latest AI news and summarise the top 3 stories"* (Tier 3, research_topic)
5. *"What's the weather this week and should I bring an umbrella?"* (Tier 2, get_weather)
6. *"Find me a focus music playlist on Spotify and set volume to 60"* (Tier 1, spotify_control)
7. *"Get the news, check the weather, and give me a morning briefing"* (Tier 3, morning_briefing)
8. *"Search for information about ChromaDB and save a note about it"* (Tier 3, multi step)
9. *"Open my notes app and set a reminder to check emails in 30 minutes"* (Tier 3, multi step)

---

## Appendix E: Code Excerpts

The following files are reproduced in full in the compiled submission package:

- `agents/planner.py`, the PlannerAgent including ReAct system prompt and JSON parsing layer.
- `agents/router.py`, the RouterAgent including the deterministic regex pre router and tier classification logic.
- `agents/critic.py`, the CriticAgent and replan trigger logic.
- `agents/evaluator.py`, the EvaluatorAgent including SQLite persistence and benchmark export.
- `memory/memory_agent.py`, the MemoryAgent including ChromaDB persistence and embedding fallback.
- `requirements.txt` and `voice_requirements.txt`, the full dependency manifests.

---

## Appendix F: Architecture Diagram

Refer to `routing_diagram.pdf` and `ui/routing_diagram.html` in the artefact repository for the canonical architecture diagram showing the full Router, Memory, Planner, Critic, Executor, Critic, Evaluator, Memory pipeline with its tier 1 and tier 2 shortcuts.

---

*End of report.*
