For my final-year dissertation I wanted to answer one question: could I build an AI assistant that actually does things, and runs entirely on my own machine?

The result is Jarvis, and I designed and built all of it myself.

What makes it different is how it thinks. Instead of one model guessing at an answer, a small team of agents handles every request. One routes it, one plans the steps, one checks the work, one scores the result, and one writes the reply. It listens and talks back in a natural voice, remembers context between conversations, and connects to the tools I use every day, like Gmail, my calendar, markets, news and files. It even has a sub-agent that reads company annual reports and answers financial questions straight from them.

It runs a local model for free and keeps my data on the machine, then switches to a more powerful cloud model with a single setting when I need it.

I put together a short architecture document that walks through the whole thing: how a single request travels from your voice to a finished action, what each part does, and how it all connects. It's attached below.

I learned more building this than in any module. If you work on multi-agent systems, local LLMs or voice, I would love to compare notes, and I am open to opportunities where I can keep building things like this.

#AI #MachineLearning #MultiAgentSystems #LLM #SoftwareEngineering
