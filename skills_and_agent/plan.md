# Match3 Skill & Agent 细分与优化计划

## 分析结论

两个原始SKILL (match3-conv-transformer-trainer 和 match3-pattern-trainer) 存在以下问题：
1. **边界不清**: Skill中混入大量可执行代码实现，违反"Skill=知识规范，Agent=可执行工作流"的原则
2. **内容冗余**: 两者共享Pattern定义、数据生成逻辑、训练流程等，却各自重复
3. **版本冲突**: PyTorch版本、Python版本、参数命名不一致
4. **架构重叠**: 一个是CNN基础版，一个是Transformer高级版，但缺乏共享基础层

## 重组方案

### 阶段1: 提取共享领域知识 → 3个基础Skills
- **match3-pattern-core**: Pattern类型定义、检测规则、标注标准 (领域规范)
- **match3-data-generation**: 数据生成策略、难度控制、注入规范、FruitOne-Hot编码 (领域规范)
- **match3-training-foundation**: 训练通用规范、损失设计原则、评估标准、6阶段课程学习 (方法论)

### 阶段2: 提取模型架构知识 → 2个架构Skills
- **match3-cnn-unet-trainer**: U-Net架构规范、CNN训练最佳实践、One-Hot输入适配
- **match3-hybrid-transformer-trainer**: RoPE+Conv+Transformer架构规范、多任务训练策略

### 阶段3: 构建可执行Agent → 2个Agents
- **match3-cnn-trainer-agent**: CNN完整训练工作流代码（含6阶段课程学习、动态fruit种类、One-Hot编码）
- **match3-hybrid-trainer-agent**: Hybrid完整训练工作流代码

### 阶段4: 共享工具Skill → 1个
- **match3-shared-utils**: 显存监控、推理API、CLI工具、旧检查点兼容加载

## 课程学习6阶段规划

| 阶段 | 标识 | 棋盘尺寸 | Fruit种类 | Match长度 | 输入编码 | Epoch范围 | 目标 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Stage1 | `10` | 10×10 | 6 (固定) | 3~5 | One-Hot | 1–20 | 基础Pattern特征学习 |
| Stage2 | `25` | 25×25 | 6 (固定) | 3~5 | One-Hot | 21–40 | 中等棋盘空间关系 |
| Stage3 | `50` | 50×50 | 6 (固定) | 3~5 | One-Hot | 41–60 | 大棋盘复杂多Pattern |
| Stage3.5 | `-1` | 正方形随机10~50 | 6 (固定) | 3~5 | One-Hot | 61–80 | 过渡：尺寸变化，比例不变 |
| Stage4 | `-3` | 长方形随机10~50 | 6 (固定) | 3~5 | One-Hot | 81–100 | 宽高独立随机，适应任意比例 |
| Stage5 | `-2` | 长方形随机10~50 | 随机5~12 | 随机5~8 | One-Hot | 101–120 | 全面泛化（尺寸/比例/种类/长度） |

## 关键技术决策

### Fruit 正n边形顶点编码
- 使用1D Rotary Position Embedding替代one-hot编码
- 固定32维输出，与fruit种类数无关
- 支持动态5~12种fruit种类（Stage5）
- 旧one-hot检查点可通过stem层零填充适配

### 动态数据生成
- `Match3BoardGenerator.generate_board()`支持运行时覆盖 `num_fruit_types` / `min_match_length` / `max_match_length`
- Stage5每样本独立随机化所有参数
- 随机尺寸棋盘通过嵌入50×50画布统一batch处理

## 输出文件

| 类型 | 文件名 |
|------|--------|
| Skill | match3-pattern-core.skill.md |
| Skill | match3-data-generation.skill.md |
| Skill | match3-training-foundation.skill.md |
| Skill | match3-cnn-unet-trainer.skill.md |
| Skill | match3-hybrid-transformer-trainer.skill.md |
| Skill | match3-shared-utils.skill.md |
| Agent | match3-cnn-trainer-agent.agent.md |
| Agent | match3-hybrid-trainer-agent.agent.md |
