# World→Action Guidance 对比矩阵 (LIBERO)

在现有 `QwenGR00T + WAM`（Qwen3-VL-2B 原生视觉 policy + DINO 世界模型监督头）基础上，公平比较把
**世界模型信号注入 action 生成**的多种方式（M0–M6+）。详见根计划《StarVLA World→Action Guidance Code Plan》。

## 设计要点
- **保留原始 action 条件**：`M_a=[h_act; H_qwen-context]`，world 信号只「增」不「删」。
- **直接 guidance 用 passive world**（`o_t,l,Q_W→ẑ_{t+H}`），不引入需动作输入的 FDM（避免 `a→ẑ→a`）。
- **双 query 控制组 M0-Q**：排除「仅加 future query 就改 Qwen 表征」的混淆。
- 统一开关 `framework.wam.guidance.*`，默认全关 → 等价原生 WAM。

## 目录
```
compare/
├── base/base_compare.yaml        # 公共底座（2B recipe，≈exp6）
├── experiment_manifest.yaml      # 每个实验只存「差异」
├── generate_configs.py           # base + manifest → configs/<name>.yaml（OmegaConf.merge）
├── run_compare.sh                # 统一启动：run_compare.sh <exp> [init_ckpt]
├── configs/                      # 生成的完整 yaml（勿手改，改 manifest 再生成）
└── README.md
```

## 用法（基本）
```bash
# 生成配置（已自带；改了 base/manifest 后重跑）
python examples/LIBERO/compare/generate_configs.py                          # 全部
python examples/LIBERO/compare/generate_configs.py m0q_dual_query_control   # 单个

# 启动一个实验（8 卡 DeepSpeed ZeRO-2）；第 2 个参数=可选 init checkpoint（热启/warmup 链接）
bash examples/LIBERO/compare/run_compare.sh <exp_name> [init_checkpoint]
```
完整的训练顺序 + 命令见下方 **「训练执行顺序」**。

## 实验矩阵
| 实验 | mode | signal | 状态 |
| --- | --- | --- | --- |
| m0_baseline | none | none | ✅ 可跑（纯 policy 基线）|
| m0q_dual_query_control | none | none(双query) | ✅ 可跑 |
| m1_1_hfuture_concat | concat | h_future | ✅ 可跑 |
| m1_2_zpred_concat | concat | z_pred | ✅ 可跑 |
| m1_3_dzpred_concat | concat | delta_z_pred | ✅ 可跑 |
| warmup_oracle_{absolute,delta} | concat | (Δ)z_pred, bridge=oracle | ✅ 已完成（Stage1）|
| warmup_predicted_{absolute,delta} | concat | (Δ)z_pred, bridge=predicted | ✅ 已完成（Stage2 = concat 基线）|
| m3_qformer_best | qformer | z_pred | 🔲 待跑（结构）|
| m4_alternate_best | alternate_xattn | z_pred | 🔲 待跑（cross block 交替 action/world）|
| m5_dual_gate_best | dual_xattn | z_pred | 🔲 待跑（gated world cross-attn，gate init 0）|
| m6_adaln_best | adaln | z_pred | 🔲 待跑（pooled world→AdaLN，zero-init）|
| m6plus_dual_adaln_best | dual_xattn_adaln | z_pred | 🔲 待跑（dual cross-attn + AdaLN）|
| m5_sample_dual_gate_best | dual_xattn | z_pred | 🔲 采样消融：M5 + 帧级均匀采样 |
| m6plus_sample_dual_adaln_best | dual_xattn_adaln | z_pred | 🔲 采样消融：M6+ + 帧级均匀采样 |

**P4 已定：S\* = dino_future**（预测的绝对未来 DINO；`signal=z_pred` + `world_target=absolute`）。h_future 弃用，
`m2_sa_hfuture`（sa_fusion 写死用 h_future）已剔除。**结构对比的 5 个结构 + concat 基线全部从 `warmup_oracle_absolute`（Stage1）热启、训同样步数**（concat-from-oracle = `warmup_predicted_absolute`；若结构改 50k 则 concat 也按 50k 重跑，保证同步数公平）。

## 因果消融（评测，plan §11）
推理时设 `framework.wam.guidance.world_eval_mode`：`correct|off|zero|shuffled|wrong_task|gt`。
只有 `shuffled/wrong_task` 明显掉 SR，才证明 action branch 真在用 world branch。
（`gt` 需要未来帧，live LIBERO eval 无未来帧 → 仅离线分析可用。）

## warm-up 在做什么（仅 z_pred / Δz_pred 这类「预测潜变量」信号需要）
病：训练初期 world head 随机 → `z_pred` 是噪声 → action 学会忽略 world；等 world head（靠 passive）预测准了，action 早就放弃用 world 了（冷启动耦合）。h_future（M1.1）是 Qwen hidden、天生有意义，**不需要** warm-up。

DIAL 三段解（oracle → predicted → e2e）：
```
Stage1  warmup_oracle_*     bridge=oracle    detach_world=true
        action 喂【GT 未来 DINO】(batch 的 dino_1) → 学「给完美信号怎么用」；world head 同时被 passive 训。解耦。
            │ (存 ckpt)
Stage2  warmup_predicted_*  bridge=predicted detach_world=true   ← load Stage1
        改喂【world head 真实预测】→ action 适应预测误差。仍不回传 world head。
            │
Stage3  (e2e)               bridge=predicted detach_world=false  ← 把 Stage2 配置 detach_world 改 false 再 load
        action loss 也回传 world head，联合微调。
```
oracle 是**训练拐杖**（eval 没有未来帧），所以必须 Stage2 撤掉。`*_absolute` / `*_delta` 对应两种 world target（绝对未来 DINO / 标准化 Δ），各配一套。

## 训练执行顺序（按 plan §13，含命令）
> 唯一硬 ckpt 依赖 = `warmup_oracle → warmup_predicted`。`m0q` / `m1_1` 是各自独立从 base 训的基线/处理组，**不产出给别人用的 ckpt**；卡在结构实验（m2–m6+）前面的是「**先有 m1 的 SR 结果来定最佳信号**」（改 config，不是传 ckpt）。所有实验都在 **LIBERO-plus** 上评测，跟 81.7 对齐。

```bash
CMP=examples/LIBERO/compare/run_compare.sh
OUT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_compare

# ── P0 基线 ───────────────────────────────────────────────
# m0 已完成（单 query, 81.7 on plus）
bash $CMP m0q_dual_query_control          # 双 query 公平基线（对照 81.7 看双 query 代价）

# ── P3 信号选择：三条并行，都从 base 训 ────────────────────
bash $CMP m1_1_hfuture_concat             # h_future：直接训（无需 warmup）
# z_pred / Δz_pred 走 warmup 链（防冷启动），硬依赖在这：
bash $CMP warmup_oracle_absolute
bash $CMP warmup_predicted_absolute  $OUT/<日期>_cmp_warmup_oracle_absolute   # = 正确版 M1.2
bash $CMP warmup_oracle_delta
bash $CMP warmup_predicted_delta     $OUT/<日期>_cmp_warmup_oracle_delta      # = 正确版 M1.3

# ── P4 已定：S* = dino_future（signal=z_pred + world_target=absolute）。

# ── P5-P8 结构对比：全部从 warmup_ORACLE_absolute 热启（公平的关键！）─────────
#    为什么是 oracle 不是 predicted：concat 基线 warmup_predicted_absolute 本身 = 「concat，从 oracle 出发、
#    bridge=predicted 训 N 步」。要 apples-to-apples，每个结构都该 = 「mode=X，从同一 oracle 出发、训同样 N 步」。
#    从 predicted 出发会让结构多吃一整轮 predicted 训练 → 不公平。oracle ckpt 的 world head 已被 passive 训好
#    （预测非噪声）→ 无冷启动耦合。run_compare.sh 第 2 参数可传 run 目录（自动定位 ckpt 文件）。
ORC_ABS=$(ls -d $OUT/*_cmp_warmup_oracle_absolute | sort | tail -1)   # 自动取最新的 oracle_absolute run
echo "结构实验热启自: $ORC_ABS"
# 后训练步数：oracle ckpt 已近收敛 → 可减半（MAX_STEPS=50000）。但 5 结构 + concat 基线必须同步数才公平：
#   选项A(推荐)：定 50k，并把 concat 也按 50k 从 oracle 重跑一遍（即下面的 warmup_predicted_absolute）。
#   选项B：结构也用 100k（去掉 MAX_STEPS），直接拿现成 warmup_predicted_absolute@100k 当 concat 基线。
export MAX_STEPS=50000
bash $CMP warmup_predicted_absolute "$ORC_ABS"   # concat 基线（选项A：与结构同 50k 从 oracle 重跑）
bash $CMP m3_qformer_best           "$ORC_ABS"   # M3 Q-Former 压缩 196→64
bash $CMP m4_alternate_best         "$ORC_ABS"   # M4 cross block 交替 action/world
bash $CMP m5_dual_gate_best         "$ORC_ABS"   # M5 gated world cross-attn（gate init 0）
bash $CMP m6_adaln_best             "$ORC_ABS"   # M6 pooled world → AdaLN（zero-init）
bash $CMP m6plus_dual_adaln_best    "$ORC_ABS"   # M6+ dual cross-attn + AdaLN
bash $CMP m5_sample_dual_gate_best       "$ORC_ABS"   # M5 + 帧级均匀采样（看 long/长轨迹任务是否提升）
bash $CMP m6plus_sample_dual_adaln_best  "$ORC_ABS"   # M6+ + 帧级均匀采样（看 long/长轨迹任务是否提升）

# ── P9 因果消融 + 效率（对结构 best 模型）────────────────
# eval 时设 guidance.world_eval_mode: correct|off|zero|shuffled 比 SR（只有 shuffled 明显掉才算真用了 world）
```

### 要点
1. **唯一硬 ckpt 链** = `warmup_oracle → warmup_predicted`；其余实验互相独立、可并行。
2. **`m1_2_zpred_concat` / `m1_3_dzpred_concat`（from-scratch）可不跑** —— 它们是「朴素冷启动版」；定信号用 warmup 链结果（正确版 M1.2/M1.3）。想做「warmup 有没有用」消融时再跑它们。
3. **S\* 已定 = z_pred（预测潜变量）→ 结构实验（m3–m6+ + concat 基线）一律从 `warmup_ORACLE_absolute` 热启**（不是 predicted！见上方注释：从 oracle 出发训 bridge=predicted 才和 concat 基线同起跑线）。oracle ckpt 的 world head 已被 passive 训好 → 无冷启动；新结构子模块（qformer/world_attn/world_to_temb/pooler）按 strict=False 自动 fresh init。
4. 热启后是**后训练**（is_resume=false：只载权重，optimizer/scheduler 从头、cosine warmup 重走）。oracle 已近收敛 → 后训练步数可减半（`MAX_STEPS=50000 bash $CMP ...`）。**唯一硬约束：5 结构 + concat 基线同步数**（要减半就把 concat 也按 50k 从 oracle 重跑）。
