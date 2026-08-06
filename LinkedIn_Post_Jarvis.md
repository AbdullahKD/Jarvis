# LinkedIn Post — Jarvis

Six months ago I set out to answer a question for my final-year dissertation: could I build a genuinely useful AI assistant that runs on my own machine, not someone else's cloud?

The result is Jarvis — a multi-agent AI executive assistant I designed and built end to end.

It's not a chatbot with a wrapper. Under the hood, every request flows through a pipeline of specialised agents — a router decides intent, a planner breaks complex tasks into steps, and critic and evaluator agents check the work before it reaches me. It runs locally on my Mac with Llama 3.2 via Ollama, and deploys to the cloud on Groq when I need more horsepower.

A few things it actually does:
- Reads and replies to my Gmail in-thread, and manages my Google Calendar
- Talks back in a natural voice (Whisper for listening, ElevenLabs for speaking) and knows when I've stopped talking
- Pulls live markets, news, sports, and weather into a morning briefing
- Runs FinEx, a sub-agent that reads company annual reports and answers financial questions
- Remembers context across conversations with a semantic memory store

The hardest parts weren't the features — they were the unglamorous ones: cutting end-to-end latency, making voice feel conversational, and keeping the whole thing stable when any one service fails.

I learned more building this than in any module. Huge thanks to everyone who supported the project along the way.

Always happy to talk multi-agent systems, local LLMs, or voice interfaces — and I'm open to opportunities where I can keep building things like this.

#AI #MachineLearning #MultiAgentSystems #LLM #SoftwareEngineering #Python
