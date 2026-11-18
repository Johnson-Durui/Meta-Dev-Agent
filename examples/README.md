# Meta-Dev Showcase Examples

This directory contains curated sample projects built and evolved with **Meta-Dev-Agent** workflows, demonstrating multi-file full-stack web apps, live API dashboards, and hardware-coupled AI agents.

| Example | Stack | Description |
| :--- | :--- | :--- |
| [`ai_daily_briefing/`](./ai_daily_briefing/) | Python, RSS/Atom, Vanilla JS/CSS | Automated morning AI news & X/Twitter feed aggregator with local JSON history and responsive web reader. |
| [`meme_radar/`](./meme_radar/) | FastAPI, `httpx`, Vanilla JS, Docker | Real-time DEX Screener token radar dashboard with boost/momentum signals, watchlist persistence, and cloud deploy configs. |
| [`talking_plant/`](./talking_plant/) | `faster-whisper`, `sentence-transformers`, `pyserial`, Arduino C++ | Voice-interactive desktop robotics companion coupling speech-to-text, semantic emotion/intent classification, TTS, and dual-servo + NeoPixel Arduino firmware. |

Each showcase project maintains its own isolated dependencies (`requirements.txt`) so the root `Meta-Dev-Agent` orchestrator remains lightweight.
