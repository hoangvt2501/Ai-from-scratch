# Tối ưu hóa Policy gần (PPO)

> A2C vứt bỏ từng rollout sau một lần cập nhật. PPO bao bọc policy gradient theo tỷ lệ quan trọng bị cắt để bạn có thể thực hiện 10+ epochs trên cùng một dữ liệu mà policy không phát nổ. Schulman và cộng sự (2017). Vẫn là thuật toán policy-gradient mặc định vào năm 2026.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 9 · 06 (TĂNG CƯỜNG), Giai đoạn 9 · 07 (Diễn viên-Nhà phê bình)
**Thời lượng:** ~75 phút

## Vấn đề

A2C (Bài 07) là policy bật: gradient `E_{π_θ}[A · ∇ log π_θ]` yêu cầu dữ liệu được lấy mẫu từ `π_θ` *hiện tại*. Thực hiện một bản cập nhật và `π_θ` thay đổi; dữ liệu bạn đã sử dụng bây giờ đã tắt policy. Sử dụng lại nó và gradient của bạn bị sai lệch.

Rollouts rất đắt. Trên Atari, một rollout trên 8 môi trường × 128 bước = 1024 chuyển tiếp và hàng chục giây thời gian môi trường. Vứt bỏ điều đó sau một bước gradient là lãng phí.

Tối ưu hóa Policy khu vực tin cậy (TRPO, Schulman 2015) là bản sửa lỗi đầu tiên: hạn chế mỗi bản cập nhật để sự phân kỳ KL giữa policy cũ và mới nằm dưới `δ`. Về mặt lý thuyết, sạch sẽ, nhưng yêu cầu giải gradient liên hợp cho mỗi bản cập nhật. Không ai chạy TRPO vào năm 2026.

PPO (Schulman et al. 2017) thay thế ràng buộc vùng tin cậy cứng bằng một mục tiêu bị cắt đơn giản. Một dòng mã bổ sung. Mười epochs mỗi rollout. Không có gradients liên hợp. Đảm bảo lý thuyết đủ tốt. Chín năm sau, nó vẫn là thuật toán gradient policy mặc định cho mọi thứ từ MuJoCo đến RLHF.

## Khái niệm

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**Tỷ lệ quan trọng.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

Đây là tỷ lệ likelihood của policy mới so với policy thu thập dữ liệu. `r_t = 1` có nghĩa là không thay đổi. `r_t = 2` có nghĩa là policy mới có khả năng mất `a_t` gấp đôi so với cũ.

**Người mang thai hộ bị cắt.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

Hai thuật ngữ:

- Nếu lợi thế `A_t > 0` và tỷ lệ cố gắng vượt qua `1 + ε`, clip sẽ làm phẳng gradient - đừng đẩy một hành động tốt xa hơn `+ε` trên xác suất cũ.
- Nếu lợi thế `A_t < 0` và tỷ lệ cố gắng vượt quá `1 - ε` (có nghĩa là chúng ta sẽ làm cho một hành động xấu có nhiều khả năng xảy ra hơn so với việc cắt giảm của nó), clip giới hạn gradient - đừng đẩy một hành động xấu xuống dưới `-ε`.

`min` xử lý theo hướng khác: nếu tỷ lệ đã di chuyển theo hướng * có lợi *, bạn vẫn nhận được gradient (không có kẹp ở bên có thể làm tổn thương bạn).

`ε = 0.2` điển hình. Vẽ mục tiêu như một hàm của `r_t`: một hàm tuyến tính từng phần với mái bằng ở "mặt tốt" và sàn phẳng ở "mặt xấu".

**Toàn bộ PPO loss.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

Cấu trúc diễn viên-nhà phê bình tương tự như A2C. Ba hệ số, thường là `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`.

**Vòng lặp training.**

1. Thu thập `N × T` chuyển tiếp trên `N` môi trường song song cho mỗi bước `T`.
2. Lợi thế tính toán (GAE), đóng băng chúng dưới dạng hằng số.
3. Đóng băng `π_{θ_old}` dưới dạng ảnh chụp nhanh về `π_θ` hiện tại.
4. Đối với `K` epochs, đối với mỗi lô nhỏ `(s, a, A, V_target, log π_old(a|s))`:
   - Tính toán `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`.
   - Áp dụng `L^{CLIP}` + giá trị loss + entropy.
   - Bước Gradient.
5. Loại bỏ rollout. Quay lại bước 1.

`K = 10` và minibatch là 64 là một bộ hyperparameter tiêu chuẩn. PPO mạnh mẽ: con số chính xác hiếm khi quan trọng trong vòng ±50%.

**Biến thể hình phạt KL.** Bài báo ban đầu đề xuất một giải pháp thay thế bằng cách sử dụng hình phạt KL thích ứng: `L = L^{PG} - β · KL(π_θ || π_old)` với `β` được điều chỉnh dựa trên KL quan sát được. Phiên bản cắt trở nên thống trị; biến thể KL tồn tại trong RLHF (trong đó KL đến policy tham chiếu là một ràng buộc riêng biệt mà bạn luôn muốn).

## Tự xây dựng

### Bước 1: chụp `log π_old(a | s)` tại thời điểm rollout

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

Ảnh chụp nhanh được chụp một lần, tại rollout thời điểm. Nó không thay đổi trong quá trình cập nhật epochs.

### Bước 2: tính ưu điểm GAE (Bài 07)

Tương tự như A2C. Bình thường hóa trên toàn batch.

### Bước 3: cập nhật người thay thế bị cắt

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

Mô hình "cắt → không gradient" là trái tim của PPO. Nếu policy mới đã trôi quá xa theo hướng có lợi, quá trình cập nhật sẽ dừng lại.

### Bước 4: giá trị và entropy

Thêm MSE tiêu chuẩn vào mục tiêu phê bình và phần thưởng entropy cho diễn viên, giống như A2C.

### Bước 5: chẩn đoán

Ba điều cần theo dõi mỗi bản cập nhật:

- **Ý nghĩa KL **`E[log π_old - log π_θ]`. Nên ở trong `[0, 0.02]`. Nếu nó thổi qua `0.1`, hãy giảm `K_EPOCHS` hoặc `LR`.
- **Phân số clip **- phần của samples có tỷ lệ nằm ngoài `[1-ε, 1+ε]`. Nên được `~0.1-0.3`. Nếu `~0`, clip không bao giờ triggers → nâng `LR` hoặc `K_EPOCHS`. Nếu `~0.5+`, bạn đang lắp quá nhiều rollout → hạ thấp chúng.
- **Giải thích variance** `1 - Var(V_target - V_pred) / Var(V_target)`. Chỉ số chất lượng của nhà phê bình. Nên leo lên 1 khi nhà phê bình tìm hiểu.

## Cạm bẫy

- **Hệ số clip bị điều chỉnh sai.** `ε = 0.2` là tiêu chuẩn trên thực tế. Đi `0.1` làm cho các bản cập nhật quá rụt rè; `0.3+` mời gọi sự bất ổn.
- **Quá nhiều epochs.** `K > 20` thường xuyên gây mất ổn định vì policy trôi dạt xa so với `π_old`. Giới hạn epochs, đặc biệt là đối với các mạng lớn.
- **Không chuẩn hóa phần thưởng. **Thang điểm phần thưởng lớn ăn vào phạm vi clip. Chuẩn hóa phần thưởng (đang chạy std) trước khi tính toán lợi thế.
- **Quên chuẩn hóa lợi thế. **Chuẩn hóa mỗi batch zero-mean/unit-std là tiêu chuẩn. Bỏ qua nó sẽ phá hỏng PPO trên hầu hết các benchmarks.
- **Learning rate không bị phân rã.** PPO hưởng lợi từ sự phân rã LR tuyến tính về không. LR không đổi thường tồi tệ hơn.
- **Lỗi toán học tỷ lệ quan trọng.** Luôn `exp(log_new - log_old)` cho sự ổn định về số chứ không phải `new / old`.
- **Dấu gradient sai.** Tối đa hóa người thay thế = *giảm thiểu* `-L^{CLIP}`. Dấu hiệu bị lật là lỗi PPO phổ biến nhất.

## Ứng dụng

PPO là thuật toán RL mặc định của năm 2026 trên một số miền đáng ngạc nhiên:

| Trường hợp sử dụng | PPO biến thể |
|----------|-------------|
| MuJoCo / điều khiển robot | PPO với Gaussian policy, GAE(0.95) |
| Atari / trò chơi rời rạc | PPO với policy phân loại, lăn 128 bước rollouts |
| RLHF cho LLMs | PPO với hình phạt KL đối với model tham chiếu, phần thưởng từ RM khi kết thúc phản hồi |
| agents trò chơi quy mô lớn | IMPALA + PPO (AlphaStar, OpenAI năm) |
| Lý luận LLMs | GRPO (Bài 12) — PPO biến thể không có nhà phê bình |
| Dữ liệu chỉ tùy chọn | DPO — sụp đổ dạng đóng của PPO+KL, không có sampling trực tuyến |

Hình dạng loss PPO * - thay thế cắt + giá trị + entropy - là giàn giáo cho DPO, GRPO và gần như mọi RLHF pipeline.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Bài tập

1. **Dễ dàng.** Chạy PPO trên 4×4 GridWorld với `ε=0.2, K=4`. So sánh hiệu suất mẫu với A2C (một epoch mỗi rollout) ở các bước môi trường phù hợp.
2. **Trung bình.** Quét `K ∈ {1, 4, 10, 30}`. Cốt truyện quay trở lại so với các bước môi trường và theo dõi KL trung bình cho mỗi bản cập nhật. KL phát nổ ở `K` nào trong nhiệm vụ này?
3. **Cứng.** Thay thế người thay thế bị cắt bằng hình phạt KL thích ứng (`β` tăng gấp đôi nếu `KL > 2·target`, giảm một nửa nếu `KL < target/2`). So sánh lợi nhuận cuối cùng, độ ổn định và độ không có kẹp.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Tỷ lệ quan trọng | "r_t(θ)" | 'π_θ(a\ | s) / π_old (a\ | s)`; sai lệch so với policy đã thu thập dữ liệu. |
| Người mang thai hộ bị cắt | "Thủ thuật chính của PPO" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; phẳng gradient qua kẹp ở phía có lợi. |
| Khu vực tin cậy | "Ý định TRPO / PPO" | Giới hạn KL của mỗi bản cập nhật để đảm bảo cải thiện đơn điệu. |
| Hình phạt KL | "Khu vực tin cậy mềm" | PPO thay thế: 'L - β · KL(π_θ \ | \ | π_old) `. Adaptive `β '. |
| Phân số clip | "Tần suất cắt triggers" | Chẩn đoán - nên là 0,1-0,3; bên ngoài có nghĩa là điều chỉnh sai. |
| Đa epoch training | "Tái sử dụng dữ liệu" | K epochs trên mỗi rollout; variance chi phí được giao dịch cho hiệu quả mẫu. |
| Bật policy | "Chủ yếu là trên policy" | PPO trên danh nghĩa là policy nhưng K>1 epochs sử dụng dữ liệu hơi lệch policy một cách an toàn. |
| PPO-KL | "Người kia PPO" | KL-penalty; được sử dụng trong RLHF mà KL-to-reference đã là một ràng buộc. |

## Đọc thêm

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - tờ giấy.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) - TRPO, tiền thân của PPO.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) - mọi PPO hyperparameter đều bị loại bỏ.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - Hướng dẫnGPT; công thức PPO trong RLHF.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) - triển lãm hiện đại sạch sẽ với PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — PPO một tệp tham khảo được nhiều bài báo sử dụng.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) - công thức production để PPO về ngôn ngữ models; đọc cùng với Bài 09 (RLHF).
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — bài báo "37 tối ưu hóa cấp mã"; thủ thuật nào PPO chịu tải và thủ thuật nào là văn hóa dân gian.
