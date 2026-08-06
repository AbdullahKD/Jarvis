# LinkedIn Post — J.A.R.V.I.S

---

For my final-year dissertation, I didn't want to build another chatbot.

I wanted to build something that could actually *do* the thing — and reason about it like a team would.

So I built **J.A.R.V.I.S**: a multi-agent AI executive assistant that takes a request by voice or text and carries it all the way to a finished action. Reply to an email. Book a meeting. Pull a market briefing. Not a suggestion — the action itself.

The part I'm most proud of isn't a single model. It's the pipeline behind it.

Instead of one model guessing an answer, five specialised agents each own one job:

→ **Router** decides how hard the request is
→ **Planner** breaks it into ordered, executable steps
→ **Critic** sanity-checks the plan before anything runs
→ **Evaluator** scores the result 0–1, so weak answers get caught
→ **Summariser** writes the one clean reply you actually see

A ChromaDB memory carries context across turns, so it isn't starting cold every time you speak to it.

And it runs **locally, at $0** — a private Llama model on your own machine, with the audio never leaving the device. One environment variable flips the entire stack to a 70B cloud model when you want more power. No code changes.

12+ live integrations. A dedicated financial-analyst sub-agent that reads annual reports and answers with cited SQL. Voice in, voice out.

Built solo, start to finish.

The big lesson: reliability in AI doesn't come from a bigger model. It comes from giving each step one job and letting the system check its own work.

Architecture overview in the comments. Happy to talk through any layer of it. 👇

#AI #MultiAgent #LLM #MachineLearning #Python #LocalFirst #SoftwareEngineering
