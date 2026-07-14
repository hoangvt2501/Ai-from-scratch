# RL cho trò chơi — AlphaZero, MuZero và kỷ nguyên lý luận LLM

> 1992: TD-Gammon đánh bại các nhà vô địch con người ở môn cờ thỏ cáo với TD thuần túy. 2016: AlphaGo đánh bại Lee Sedol. 2017: AlphaZero thống trị cờ vua, shogi và cờ vây từ đầu. 2024: DeepSeek-R1 đã chứng minh công thức tương tự, với GRPO thay thế PPO, hoạt động dựa trên lý luận. Trò chơi là benchmark thúc đẩy mọi đột phá trong giai đoạn này.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 9 · 05 (DQN), Giai đoạn 9 · 08 (PPO), Giai đoạn 9 · 09 (RLHF), Giai đoạn 9 · 10 (MARL)
**Thời lượng:** ~120 phút

## Vấn đề

Trò chơi có mọi thứ RL muốn. Phần thưởng sạch (win/loss). episodes vô hạn (đặt lại tự phát). Mô phỏng hoàn hảo (trò chơi * là * trình mô phỏng). Không gian hành động rời rạc hoặc nhỏ liên tục. Cấu trúc đa agent buộc sự mạnh mẽ của đối thủ.

Và trò chơi là cách mọi bước đột phá lớn của RL đều được thử nghiệm. TD-Gammon (cờ thỏ cáo, 1992). Atari-DQN (2013). AlphaGo (2016). AlphaZero (2017). OpenAI Năm (Dota 2, 2019). AlphaStar (StarCraft II, 2019). MuZero (học model, 2019). AlphaTensor (phép nhân ma trận, 2022). AlphaDev (thuật toán sắp xếp, 2023). DeepSeek-R1 (suy luận toán học, 2025) — minh chứng mới nhất cho thấy các kỹ thuật RL trò chơi hoạt động trên văn bản.

Capstone này khảo sát ba kiến trúc mang tính bước ngoặt - AlphaZero, MuZero và GRPO - thông qua một lăng kính thống nhất duy nhất: **tự phát + tìm kiếm + cải thiện policy**. Mỗi khái quát hóa trước đó; GRPO đặc biệt là công thức của AlphaZero được áp dụng cho lý luận LLM, với tokens là hành động và xác minh toán học là tín hiệu chiến thắng.

## Khái niệm

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**Vòng lặp thống nhất.**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017).** Silver và cộng sự. Cho một trò chơi (cờ vua, shogi, cờ vây) với các quy tắc đã biết:

- Mạng lưới giá trị Policy: Một tháp `f_θ(s) → (p, v)`. `p` là một prior đối với các động thái pháp lý. `v` là kết quả trò chơi được mong đợi.
- Tìm kiếm cây Monte Carlo (MCTS): ở mỗi nước đi, mở rộng một cây có thể tiếp tục. Sử dụng `(p, v)` làm prior + bootstrap. Chọn nút theo UCB (PUCT): `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- Tự chơi: chơi trò chơi agent vs agent. Ở `t` di chuyển, `π_t` phân phối lượt truy cập MCTS trở thành mục tiêu policy training.
- Loss: `L = (v - z)² - π · log p + c · ||θ||²`. `z` là kết quả trò chơi (+1 / 0 / -1).

Không có kiến thức của con người. Không có phỏng đoán thủ công. Một công thức duy nhất thành thạo cờ vua, shogi và cờ vây sau vài chục triệu trò chơi tự chơi mỗi trò chơi.

**MuZero (2019).** Schrittwieser và cộng sự. Loại bỏ yêu cầu rằng các quy tắc phải được biết.

- Thay vì một môi trường cố định, hãy học một *động lực tiềm ẩn model* `(h, g, f)`:
  - `h(s)`: mã hóa quan sát thành trạng thái tiềm ẩn.
  - `g(s_latent, a)`: Dự đoán trạng thái tiềm ẩn tiếp theo + phần thưởng.
  - `f(s_latent)`: Dự đoán policy prior + giá trị.
- MCTS chạy trong *không gian tiềm ẩn đã học*. Cùng tìm kiếm, cùng một vòng lặp training.
- Hoạt động trên cờ vây, cờ vua, shogi * và * Atari - một thuật toán, không có kiến thức về quy tắc.

**Stochastic MuZero (2022).** Thêm động lực học ngẫu nhiên và các nút cơ hội; mở rộng sang các trò chơi class cờ thỏ cáo.

**Muesli, Gumbel MuZero (2022-2024).** Cải tiến về hiệu quả mẫu và tìm kiếm xác định.

**GRPO (2024-2025).** Công thức DeepSeek-R1. Cùng một vòng lặp hình AlphaZero, được áp dụng cho suy luận model ngôn ngữ:

- "Game": trả lời một bài toán / mã hóa / suy luận. "Win" = verifier (trường hợp thử nghiệm đạt, câu trả lời bằng số khớp) trả về 1.
- Policy: LLM. Hành động: tokens. Trạng thái: prompt + phản hồi cho đến nay.
- Không có người chỉ trích (V_φ kiểu PPO). Thay vào đó, đối với mỗi prompt, hãy lấy mẫu `G` hoàn thành từ policy. Tính toán phần thưởng cho mỗi phần. Sử dụng `A_i = (r_i - mean_r) / std_r` **lợi thế tương đối nhóm** làm tín hiệu cho bản cập nhật kiểu REINFORCE.
- Hình phạt KL để tham chiếu policy để tránh trôi (như RLHF).
- loss đầy đủ:

`L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

Không có model thưởng, không có nhà phê bình, không MCTS. Đường cơ sở tương đối nhóm thay thế cả ba. Phù hợp hoặc vượt quá chất lượng PPO-RLHF về lý luận benchmarks ở một phần nhỏ của máy tính.

**Công thức R1 đầy đủ.** DeepSeek-R1 (DeepSeek 2025) là hai models trong một bài báo:

- **R1-Zero.** Bắt đầu từ model cơ sở DeepSeek-V3. Không có SFT. Áp dụng GRPO trực tiếp với hai thành phần phần thưởng: *accuracy phần thưởng* (dựa trên quy tắc — câu trả lời cuối cùng đã phân tích cú pháp thành số chính xác / mã có vượt qua các bài kiểm tra đơn vị không) và *phần thưởng định dạng* (phần hoàn thành có bao bọc chain-of-thought của nó trong các thẻ `<think>…</think>` không). Qua hàng nghìn bước, độ dài phản hồi trung bình tăng từ ~100 lên ~10.000 tokens và điểm benchmark toán học tăng lên mức gần như xem trước o1. Người model học cách suy luận từ đầu. Nhược điểm: chuỗi suy nghĩ của nó thường không thể đọc được, kết hợp ngôn ngữ và thiếu sự bóng bẩy về phong cách.
- **R1.** Khắc phục các vấn đề về khả năng đọc của R1-Zero với bốn stage pipeline:
  1. **SFT khởi động nguội.** Thu thập vài nghìn bản trình diễn CoT dài với định dạng rõ ràng. Có giám sát-tinh chỉnh model cơ sở trên chúng. Điều này cung cấp một điểm khởi đầu dễ đọc.
  2. **GRPO định hướng lý luận.** Đăng ký GRPO với phần thưởng accuracy + định dạng cộng với phần thưởng * nhất quán ngôn ngữ * để ngăn chuyển mã.
  3. **sampling từ chối + SFT vòng 2. **Lấy mẫu ~600 nghìn quỹ đạo suy luận từ các RL checkpoint, chỉ giữ lại những người có câu trả lời cuối cùng đúng và CoT có thể đọc được, đồng thời kết hợp với ~200 nghìn ví dụ SFT không suy luận (viết, QA, tự nhận thức). Fine-tune lại cơ sở.
  4. **GRPO toàn phổ.** Thêm một vòng RL bao gồm cả lý luận (phần thưởng dựa trên quy tắc) và alignment chung (phần thưởng dựa trên sở thích helpfulness/harmlessness).

Kết quả khớp với o1 trên AIME và MATH-500 ở trọng lượng mở, và đủ nhỏ để chưng cất. Bài báo tương tự cũng phát hành sáu models chưng cất đậm đặc (Qwen-1,5B đến Llama-70B) bằng SFT trên traces lý luận của R1 - không RL vào học sinh. Distillation của một giáo viên RL mạnh mẽ luôn đánh bại RL từ đầu ở thang điểm của học sinh.

**Tại sao GRPO thay vì PPO lý luận.** Ba lý do trong bài báo DeepSeekMath (tháng 2 năm 2024): (1) không có mạng giá trị để huấn luyện, giảm một nửa bộ nhớ; (2) đường cơ sở nhóm xử lý phần thưởng cuối quỹ đạo thưa thớt mà các nhiệm vụ lý luận tạo ra một cách tự nhiên; (3) chuẩn hóa mỗi prompt tạo ra lợi thế có thể so sánh được giữa các vấn đề có độ khó cực kỳ khác nhau, điều mà nhà phê bình duy nhất của PPO không thể làm được.

**Miễn phí tìm kiếm so với dựa trên tìm kiếm.** Trò chơi đã phân nhánh:

- *Perfect-information games with long horizons* (Go, cờ vua): vẫn dựa trên tìm kiếm. AlphaZero / MuZero chiếm ưu thế.
- *LLM reasoning*: chưa có MCTS nào trong production; GRPO trên rollouts đầy đủ, tốt nhất của N cho inference tính toán. Process models phần thưởng (PRM) gợi ý về tìm kiếm cấp bước được thêm trở lại.

## Tự xây dựng

Mã trong `code/main.py` triển khai **GRPO trong thu nhỏ** - một tên cướp với nhiều nhóm mẫu. Thuật toán giống như trên LLM; chỉ có policy và môi trường là đơn giản hơn. Nó dạy *loss* và *lợi thế tương đối nhóm*, đó là sự đổi mới năm 2025.

### Bước 1: một môi trường xác minh nhỏ

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

Trong GRPO thực, trình xác minh chạy các bài kiểm tra đơn vị hoặc kiểm tra độ bình đẳng toán học.

### Bước 2: policy: softmax tokens câu trả lời trên K mỗi prompt

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

Tương đương với đầu ra lớp cuối cùng của một LLM có điều kiện trên một prompt.

### Bước 3: lợi thế sampling nhóm và lợi thế tương đối nhóm

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Lợi thế tương đối nhóm là thủ thuật DeepSeek 2024. Không cần nhà phê bình. "Đường cơ sở" là giá trị trung bình nhóm và chuẩn hóa sử dụng std nhóm.

### Bước 4: so sánh với đường cơ sở REINFORCE (không có giá trị)

Cùng một thiết lập, cùng một tính toán, REINFORCE đơn giản. GRPO hội tụ nhanh hơn và ổn định hơn.

### Bước 5: quan sát entropy và KL

Chẩn đoán tương tự như RLHF: trung bình KL để tham chiếu, entropy policy, phần thưởng theo thời gian. Khi những điều này ổn định, training được thực hiện.

## Cạm bẫy

- **Phần thưởng cho việc hack thông qua trò chơi của người xác minh.** GRPO thừa hưởng rủi ro của RLHF: nếu người xác minh sai hoặc có thể khai thác, LLM sẽ tìm thấy lỗ hổng. Người xác minh mạnh mẽ (nhiều trường hợp thử nghiệm, bằng chứng chính thức) rất quan trọng.
- **Quy mô nhóm quá nhỏ. **Variance của đường cơ sở nhóm giống như `1/√G`. Dưới `G = 4`, tín hiệu lợi thế nhiễu; lựa chọn tiêu chuẩn là `G = 8` `64`.
- **Độ dài bias.** LLM lần hoàn thành có độ dài khác nhau có xác suất log khác nhau. Chuẩn hóa theo số lượng token hoặc sử dụng log-prob cấp trình tự hoặc cắt bớt đến độ dài tối đa.
- **Chu kỳ tự chơi thuần túy.** training kiểu AlphaZero có thể bị mắc kẹt trong các vòng lặp thống trị trong các trò chơi tổng chung. Giảm thiểu bởi các nhóm đối thủ đa dạng (chơi giải đấu, Bài 10).
- **Tìm kiếm-policy không khớp.** AlphaZero huấn luyện policy bắt chước kết quả tìm kiếm. Nếu mạng policy quá nhỏ để thể hiện sự phân phối của tìm kiếm, training sẽ bị đình trệ.
- **Sàn tính toán.** MuZero / AlphaZero cần điện toán lớn. Một lần cắt bỏ thường kéo dài hàng trăm GPU giờ. Có các bản demo thu nhỏ (ví dụ: AlphaZero trên Connect Four) để học tập.
- **Phạm vi của trình xác minh.** Các bài kiểm tra đơn vị vượt qua giải pháp có lỗi sẽ củng cố lỗi. Thiết kế trình xác minh để phát hiện các trường hợp biên.

## Ứng dụng

Bối cảnh RL trò chơi năm 2026, theo miền:

| Tên miền | Phương pháp thống trị |
|--------|-----------------|
| Trò chơi cờ có tổng bằng không hai người chơi (Cờ vây, cờ vua, shogi) | AlphaZero / MuZero / KataGo |
| Trò chơi bài thông tin không hoàn hảo (poker) | CFR + học sâu (DeepStack, Libratus, Pluribus) |
| Atari / trò chơi pixel | Muesli / MuZero / IMPALA-PPO |
| Chiến lược nhiều người chơi lớn (Dota, StarCraft) | PPO + tự chơi + giải đấu (OpenAI Five, AlphaStar) |
| Lý do LLM math/code | GRPO (DeepSeek-R1, Qwen-RL, sao chép mở) |
| LLM alignment | DPO / RLHF-PPO (không phải GRPO; người xác minh là ưu tiên không thể xác minh được) |
| Người máy | PPO + DR (không phải trò chơi RL, nhưng sử dụng các công cụ gradient policy giống nhau) |
| Bài toán tổ hợp | Các biến thể AlphaZero (AlphaTensor, AlphaDev) |

* Công thức * - tự phát, cải tiến tăng cường tìm kiếm, policy distillation - spans văn bản, pixel và điều khiển vật lý. GRPO là trường hợp trẻ nhất; nhiều hơn nữa đang đến.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-game-rl-designer.md`:

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## Bài tập

1. **Dễ dàng.** Triển khai kẻ cướp GRPO trong `code/main.py`. Huấn luyện trên 2 prompts × 4 câu trả lời tokens mỗi câu trả lời. Hội tụ trong < 1.000 bản cập nhật với `G=8`.
2. **Trung bình. **Cắm PPO (cắt) và REINFORCE vani. So sánh hiệu quả mẫu và variance phần thưởng với GRPO trên cùng một tên cướp.
3. **Khó.** Mở rộng đến một "chuỗi suy luận" có độ dài 2: agent phát ra hai tokens và trình xác minh thưởng cho cặp. Đo lường cách GRPO xử lý việc gán tín dụng qua các chuỗi hai bước. (Gợi ý: tính toán lợi thế nhóm trên mỗi *chuỗi đầy đủ*, lan truyền đến cả hai vị trí token.)

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| MCTS | "Tìm kiếm cây bằng mạng đã học" | Tìm kiếm cây Monte Carlo; UCB1/PUCT lựa chọn với `(p, v)` priors đã học. |
| AlphaZero | "Tự phát + MCTS" | Ròng giá trị Policy được huấn luyện để phù hợp với lượt truy cập MCTS và kết quả trò chơi. |
| MuZero | "AlphaZero đã học model" | Cùng một vòng lặp nhưng trong không gian tiềm ẩn thông qua động lực học được. |
| GRPO | "Không có PPO phê bình" | Tối ưu hóa Policy tương đối nhóm; CỦNG cố với đường cơ sở trung bình nhóm + KL. |
| PUCT | "UCB của AlphaZero" | `Q + c · p · √N / (1 + N_a)` - cân bằng ước tính giá trị với prior. |
| Tự chơi | "Agent so với bản thân trong quá khứ" | Tiêu chuẩn cho tổng bằng không; tín hiệu training đối xứng. |
| Chơi giải đấu | "Tự chơi dựa trên dân số" | Quá khứ + hiện tại + người khai thác được lấy mẫu làm đối thủ. |
| Phần thưởng người xác minh | "RL có thể kiểm chứng" | Phần thưởng đến từ một công cụ kiểm tra xác định (bài kiểm tra vượt qua, câu trả lời phù hợp). |
| Phần thưởng Process | "PRM" | Chấm điểm từng bước suy luận, không chỉ là câu trả lời cuối cùng. |

## Đọc thêm

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) - bài báo giới thiệu GRPO và đường cơ sở tương đối nhóm.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) - công thức R1 bốn giai đoạn đầy đủ cộng với cắt bỏ R1-Zero.
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) - CFR + học sâu trên quy mô lớn.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) - tờ báo bắt đầu tất cả.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — tài liệu tham khảo production để áp dụng GRPO với các chức năng phần thưởng tùy chỉnh.
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) - sao chép mở công thức R1 ở nhiều quy mô.
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) - khung sách giáo khoa để tự chơi, tìm kiếm và "phần thưởng được thiết kế" mà R1 khởi tạo ở LLM quy mô.
