# Agent: match3-game-agent
# Description: 50x50 三消游戏推理服务 + WebSocket 实时前端，使用 Match3UNet 模型做消除判定
# Version: 1.0.0
# Dependencies: match3-cnn-trainer 训练好的 checkpoint
# Stack: Python >=3.9, FastAPI, PyTorch >=2.0, WebSocket, HTML5 Canvas

---

## 概述

本 Agent 提供完整的 50×50 三消游戏 playable demo：
- **后端**: FastAPI + WebSocket，加载 `Match3UNet` checkpoint，提供实时推理
- **前端**: Vanilla JS + HTML5 Canvas，50×50 棋盘渲染，WebSocket 实时接收动画指令
- **游戏循环**: 交换 → 模型判定消除 mask → 消除 → 重力下落 → 连锁反应 → WebSocket 推送每一帧状态

---

## 项目结构

```
match3-game/
├── backend/
│   ├── app.py            # FastAPI + WebSocket 服务
│   ├── model_service.py  # 加载 Match3UNet，封装 predict()
│   ├── game_engine.py    # Match3Game: 棋盘/交换/消除/重力/连锁
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── renderer.js   # Canvas 渲染器
│       ├── game.js       # 游戏逻辑 + WebSocket 客户端
│       └── api.js        # API 调用封装
├── README.md
└── AGENTS.md
```

---

## 后端模块

### model_service.py

加载 `match3_cnn_trainer/checkpoints/match3_unet_epochbest.pt`，提供：
- `predict(board: np.ndarray) -> (mask, patterns)`: 返回模型预测的消除 mask 和 Pattern 类型信息

### game_engine.py

`Match3Game` 类：
- `board`: 50×50 int 数组，6 种水果
- `swap(y1, x1, y2, x2)`: 相邻交换，若交换后模型无匹配则自动回滚
- `eliminate(mask)`: 将 mask 位置置为 -1（空）
- `apply_gravity()`: 每列元素下落，顶部随机生成新水果
- `resolve() -> CascadeResult`: 循环执行 模型预测 → 消除 → 重力，直到无匹配；记录每轮消除坐标
- `score`: 基于消除格子数 × 连锁倍率

### app.py (FastAPI)

HTTP 端点:
- `POST /api/new_game` → 返回新棋盘
- `GET /api/board` → 当前棋盘
- `POST /api/swap` → 执行交换并返回完整连锁结果

WebSocket 端点:
- `WS /ws/game` → 前端连接后，每次交换通过 WebSocket 实时推送动画帧：
  - `{"type": "swap", "from": [y1,x1], "to": [y2,x2]}`
  - `{"type": "highlight", "cells": [...], "pattern": "H4"}`
  - `{"type": "eliminate", "cells": [...]}`
  - `{"type": "gravity", "column_drops": [...]}`
  - `{"type": "spawn", "cells": [...]}`
  - `{"type": "cascade_end", "score": 1230}`

---

## 前端模块

### renderer.js

Canvas 2D 渲染 50×50 网格：
- 每个格子：纯色方块 + 水果 emoji（🍎🍋🍇🍊🫐🥝）
- 选中高亮、匹配高亮、消除粒子效果
- 重力下落的位移动画

### game.js

- WebSocket 连接到 `/ws/game`
- 鼠标点击/拖拽选择两个相邻格子，发送 swap 消息
- 监听 WebSocket 消息，驱动动画队列（按顺序播放 swap → highlight → eliminate → gravity → spawn → cascade_end）
- 分数更新、连锁计数显示

### api.js

封装 `fetch` 调用 HTTP 端点和 WebSocket 连接管理。

---

## 游戏时序

```
玩家点击两个相邻格子
  → 前端发送 swap 消息 (WebSocket)
  → 后端 GameEngine.swap()
  → 后端 ModelService.predict() 获取消除 mask
  → 若无匹配: 回滚 swap，发送 "invalid_swap"
  → 若有匹配:
       发送 "swap" 动画帧
       loop:
         发送 "highlight" 帧
         发送 "eliminate" 帧
         GameEngine.apply_gravity()
         发送 "gravity" + "spawn" 帧
         再次 predict() 检查新匹配
       until mask 全 false
       发送 "cascade_end" 帧
  → 前端按顺序播放动画队列
```

---

## 启动命令

```bash
cd match3-game/backend
pip install -r requirements.txt
# 确保 checkpoint 存在:
#   ../match3_cnn_trainer/checkpoints/match3_unet_epochbest.pt
python app.py
# 打开 http://localhost:8000
```

---

## 依赖

```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
websockets>=11.0
numpy>=1.24.0
torch>=2.0.0
python-multipart>=0.0.6
```

> PyTorch 请按 CUDA 版本单独安装，见 match3_cnn_trainer/requirements.txt 说明。
