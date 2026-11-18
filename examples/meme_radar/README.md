# MemeRadar Live

一个基于 DEX Screener 官方实时数据的 Meme 币跟踪工作台（`Meta-Dev-Agent` 多文件全栈展示案例）。

## 当前能力

- 实时热门 Meme 代币榜单
- 新币 / Boost / 量能 / 动量信号流
- 价格、24h 涨跌、成交量、流动性、市值追踪
- 项目社媒入口追踪
- 观察列表和本地设置持久化

## 安装依赖

```bash
cd examples/meme_radar
pip install -r requirements.txt
```

## 启动方式

```bash
python -m uvicorn server:app --app-dir examples/meme_radar --host 127.0.0.1 --port 8124
```

然后访问 `http://127.0.0.1:8124`。

## 容器与云端部署

本目录内置公开部署配置：

- [`Dockerfile`](./Dockerfile)
- [`render.yaml`](./render.yaml)
- [`railway.json`](./railway.json)

## 数据源

- DEX Screener 官方 API
- 关键词种子搜索与最新 Boost / Token Profile 聚合
