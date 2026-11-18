# AI Daily Briefing

一个本地可运行的 AI 早报网页应用（`Meta-Dev-Agent` 多文件展示案例），自动汇总：

- AI 领域重点新闻（Google News RSS 聚合）
- X 上 Gemini、Grok、Claude、ChatGPT 的动态观察线
- 中文摘要、本地历史归档与响应式阅读界面

## 目录结构

- `generate_briefing.py` — 抓取 RSS、生成中文摘要并输出 `data/latest.json` 与 `data/history/`
- `launch_briefing.py` — 刷新数据并启动本地 HTTP 预览服务
- `index.html` / `app.js` / `styles.css` — 响应式前端页面
- `Open_AI_Daily_Briefing.command` — macOS 一键启动脚本

## 快速体验（离线演示模式）

先生成一份演示数据：

```bash
python examples/ai_daily_briefing/generate_briefing.py --demo
```

再启动静态服务：

```bash
python -m http.server 8765 -d examples/ai_daily_briefing
```

浏览器访问 `http://127.0.0.1:8765/index.html`。

## 真实抓取

```bash
export X_RSS_TEMPLATE="https://nitter.net/{handle}/rss"
python examples/ai_daily_briefing/launch_briefing.py
```
