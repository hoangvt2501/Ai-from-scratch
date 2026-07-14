# Mạng Q sâu (DQN)

> 2013: Mnih đã huấn luyện một mạng Q-learning trên các pixel thô, đánh bại mọi RL agent cổ điển trên bảy trò chơi Atari. 2015: mở rộng lên 49 trò chơi, được xuất bản trên tạp chí Nature, châm ngòi cho kỷ nguyên RL sâu. DQN là Q-learning cộng với ba thủ thuật làm cho việc xấp xỉ hàm ổn định.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 3 · 03 (Backpropagation), Giai đoạn 9 · 04 (Q-learning, SARSA)
**Thời lượng:** ~75 phút

## Vấn đề

Học Q dạng bảng cần một giá trị Q riêng biệt cho mỗi cặp (trạng thái, hành động). Một bàn cờ có ~10⁴³ trạng thái. Một khung Atari là 210×160×3 = 100.800 features. RL dạng bảng chết ở hàng nghìn trạng thái, chứ đừng nói đến hàng tỷ.

Cách khắc phục là rõ ràng trong nhận thức muộn màng: thay thế bảng Q bằng mạng nơ-ron, `Q(s, a; θ)`. Nhưng nhận thức muộn phải mất nhiều thập kỷ. Xấp xỉ hàm ngây thơ với Q-learning phân kỳ theo "bộ ba chết người" - xấp xỉ hàm + bootstrapping + học ngoài policy. Mnih et al. (2013, 2015) đã xác định ba thủ thuật kỹ thuật giúp ổn định học tập:

1. **Trải nghiệm phát lại** làm mất tương quan các chuyển tiếp.
2. **Mạng đích** đóng băng mục tiêu bootstrap.
3. **Cắt phần thưởng** bình thường hóa gradient độ lớn.

DQN trên Atari là lần đầu tiên một kiến trúc duy nhất với một bộ hyperparameter duy nhất giải quyết hàng chục vấn đề điều khiển từ các pixel thô. Mọi thứ "RL sâu" được xây dựng kể từ đó - DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57 - được xếp chồng lên nhau trên cơ sở ba thủ thuật này.

## Khái niệm

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**Mục tiêu.** DQN giảm thiểu loss TD một bước trên hàm Q thần kinh:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = mạng trực tuyến, được cập nhật từng bước theo gradient descent. `θ^-` = mạng đích, được sao chép định kỳ từ `θ` (cứ sau ~10.000 bước). `D` = bộ đệm phát lại của các quá trình chuyển đổi trong quá khứ.

**Ba thủ thuật, theo thứ tự quan trọng:**

**Trải nghiệm phát lại.** Một bộ đệm vòng của các chuyển tiếp `~10⁶`. Mỗi bước training lấy mẫu một lô nhỏ một cách ngẫu nhiên. Điều này phá vỡ mối tương quan thời gian (các khung hình liên tiếp gần như giống hệt nhau), cho phép mạng học hỏi từ các chuyển đổi bổ ích hiếm hoi nhiều lần và khử tương quan các bản cập nhật gradient liên tiếp. Nếu không có nó, TD trên policy với mạng nơ-ron sẽ phân kỳ trên Atari.

**Mạng mục tiêu.** Sử dụng cùng một `Q(·; θ)` mạng ở cả hai bên của phương trình Bellman làm cho mục tiêu di chuyển mỗi bản cập nhật - "đuổi theo đuôi của chính bạn". Cách khắc phục: giữ một mạng thứ hai `Q(·; θ^-)` với trọng số bị đóng băng. Mỗi `C` bước, hãy sao chép `θ → θ^-`. Điều này ổn định mục tiêu hồi quy cho hàng nghìn bước gradient cùng một lúc. Cập nhật mềm `θ^- ← τ θ + (1-τ) θ^-` (được sử dụng trong DDPG, SAC) là một biến thể mượt mà hơn.

**Cắt phần thưởng.** Độ lớn phần thưởng của Atari thay đổi từ 1 đến 1000+. Cắt thành `{-1, 0, +1}` ngăn bất kỳ trò chơi đơn lẻ nào thống trị gradient. Sai khi mức độ phần thưởng quan trọng; tốt cho Atari nơi chỉ có dấu hiệu mới quan trọng.

**Double DQN.** Hasselt (2016) sửa lỗi tối đa hóa bias: sử dụng mạng trực tuyến để * chọn * hành động, mạng mục tiêu để * đánh giá * nó.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

Thay thế thả vào, luôn tốt hơn. Sử dụng nó theo mặc định.

**Các cải tiến khác (Rainbow, 2017):** ưu tiên phát lại (mẫu chuyển đổi lỗi TD cao nhiều hơn), kiến trúc đấu tay đôi (đầu `V(s)` và lợi thế riêng biệt), mạng nhiễu (khám phá đã học), trả về n-bước, Q phân phối (C51/QR-DQN), khởi động nhiều bước. Mỗi thêm một vài phần trăm; lợi ích gần như cộng thêm.

## Tự xây dựng

Mã ở đây chỉ không có numpy stdlib — chúng tôi sử dụng MLP một lớp ẩn cuộn bằng tay trên một GridWorld nhỏ liên tục, vì vậy mỗi bước training chạy trong micro giây. Thuật toán này giống với Atari DQN trên quy mô lớn.

### Bước 1: phát lại bộ đệm

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

~50.000 sức chứa cho Atari; 5.000 đủ cho môi trường đồ chơi của chúng tôi.

### Bước 2: một mạng Q nhỏ (MLP thủ công)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Forward pass: tuyến tính → ReLU → tuyến tính. Đó là toàn bộ lưới.

### Bước 3: cập nhật DQN

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

Hình dạng là Q-learning từ Bài 04 với hai điểm khác biệt: (a) chúng ta backprop thông qua một `Q(·; θ)` có thể vi phân thay vì lập chỉ mục một bảng, (b) mục tiêu sử dụng `Q(·; θ^-)`.

### Bước 4: vòng lặp bên ngoài

Đối với mỗi episode, hãy hành động ε tham lam trên `Q(·; θ)`, đẩy chuyển tiếp vào bộ đệm, lấy mẫu một lô nhỏ, thực hiện một bước gradient, đồng bộ hóa định kỳ `θ^- ← θ`. Mẫu:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

Trên GridWorld nhỏ bé của chúng tôi với trạng thái 16 độ mờ một nóng, agent học được policy gần như tối ưu trong ~500 episodes. Trên Atari, hãy chia tỷ lệ này lên 200 triệu khung hình và thêm một bộ trích xuất feature CNN.

## Cạm bẫy

- **Bộ ba chết người. **Xấp xỉ chức năng + tắt policy + khởi động có thể phân kỳ. DQN giảm thiểu với lưới mục tiêu + phát lại; không loại bỏ một trong hai.
- **Thăm dò.** ε phải phân rã, thường là từ 1,0 đến 0,01 trong ~10% đầu tiên của training. Nếu không có đủ thăm dò sớm, Q-net hội tụ đến một lưu vực địa phương.
- **Đánh giá quá cao. **`max` trên Q ồn ào là thiên về phía trên. Luôn sử dụng Double DQN trong production.
- **Thang phần thưởng.** Cắt hoặc chuẩn hóa phần thưởng; độ lớn gradient tỷ lệ thuận với độ lớn của phần thưởng.
- **Phát lại bộ đệm coldstart. **Đừng tập luyện cho đến khi bộ đệm có vài nghìn lần chuyển tiếp. gradients sớm trên ~20 samples overfit.
- **Tần suất đồng bộ hóa mục tiêu. **Quá thường xuyên ≈ không có mạng mục tiêu; quá hiếm ≈ các mục tiêu cũ. Atari DQN sử dụng 10.000 bước môi trường. Quy tắc chung: đồng bộ hóa mỗi ~1/100 đường chân trời training.
- **Tiền xử lý quan sát. **Atari DQN stacks 4 khung hình để tạo trạng thái Markov. Bất kỳ môi trường nào có thông tin vận tốc đều cần xếp khung hoặc trạng thái lặp lại.

## Ứng dụng

Vào năm 2026, DQN hiếm khi hiện đại nhưng vẫn là thuật toán tham chiếu ngoài policy:

| Nhiệm vụ | Phương pháp lựa chọn | Tại sao không phải DQN? |
|------|------------------|--------------|
| Hành động rời rạc giống như Atari | Cầu vồng DQN hoặc Muesli | Cùng một framework, nhiều thủ thuật hơn. |
| Kiểm soát liên tục | SAC / TD3 (Giai đoạn 9 · 07) | DQN không có mạng policy. |
| Trên policy / thông lượng cao | PPO (Giai đoạn 9 · 08) | Không có bộ đệm phát lại; dễ dàng hơn để mở rộng quy mô. |
| RL ngoại tuyến | CQL / IQL / Quyết định Transformer | Mục tiêu Q bảo thủ, không có vụ nổ khởi động. |
| Không gian hành động rời rạc lớn (đề xuất) | DQN với embedding hành động, hoặc IMPALA | Tốt; vấn đề trang trí. |
| LLM RL | PPO / GRPO | Cấp trình tự, không phải cấp bước; loss khác nhau. |

Các bài học vẫn đi qua. Phát lại và mạng mục tiêu xuất hiện trong SAC, TD3, DDPG, SAC-X, bộ đệm tự phát của AlphaZero và mọi phương thức RL ngoại tuyến. Cắt phần thưởng tồn tại dưới dạng chuẩn hóa lợi thế trong PPO. Kiến trúc là bản thiết kế.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-dqn-trainer.md`:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Vẽ đường cong trở lại trên mỗi episode. Bao nhiêu episodes cho đến khi giá trị trung bình đang chạy vượt quá -10?
2. **Trung bình.** Tắt mạng mục tiêu (sử dụng mạng trực tuyến cho cả hai bên của mục tiêu Bellman). Đo lường training không ổn định - lợi nhuận dao động hay phân kỳ?
3. **Khó.** Thêm Double DQN: sử dụng mạng trực tuyến để chọn `argmax a'`, mạng mục tiêu để đánh giá. So sánh bias `Q(s_0, best_a)` và true `V*(s_0)` sau 1.000 episodes với và không có Double DQN trên GridWorld ồn ào.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| DQN | "Học Q sâu" | Q-learning với chức năng Q thần kinh, bộ đệm phát lại và mạng đích. |
| Trải nghiệm phát lại | "Chuyển tiếp xáo trộn" | Bộ đệm vòng được lấy mẫu đồng đều mỗi gradient bước; khử tương quan dữ liệu. |
| Mạng mục tiêu | "Bootstrap đông lạnh" | Bản sao định kỳ của Q được sử dụng trong mục tiêu Bellman; ổn định training. |
| Bộ ba chết người | "Tại sao RL phân kỳ" | Xấp xỉ chức năng + khởi động + tắt policy = không đảm bảo hội tụ. |
| DQN gấp đôi | "Sửa chữa để tối đa hóa bias" | Mạng trực tuyến chọn hành động, mạng mục tiêu đánh giá nó. |
| Đấu tay đôi DQN | "Đầu V và A" | Phân hủy Q = V + A - trung bình (A); cùng một đầu ra, dòng chảy gradient tốt hơn. |
| Cầu vồng | "Tất cả các thủ thuật" | DDQN + PER + đấu tay đôi + n-step + ồn ào + phân phối trong một. |
| MỖI | "Phát lại ưu tiên" | Chuyển đổi mẫu tỷ lệ thuận với độ lớn lỗi TD. |

## Đọc thêm

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - bài báo hội thảo NeurIPS năm 2013 bắt đầu RL sâu.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - tờ Nature, DQN 49 trận.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — đấu tay đôi DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) - tờ giấy thủ thuật xếp chồng lên nhau.
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) - giải thích hiện đại rõ ràng.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — cách xử lý sách giáo khoa của "bộ ba chết người" (xấp xỉ chức năng + khởi động + tắt policy) mà mạng mục tiêu của DQN và bộ đệm phát lại được thiết kế để thuần hóa.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) - tham khảo DQN một tệp được sử dụng trong các nghiên cứu cắt bỏ; tốt để đọc cùng với phiên bản từ đầu của bài học này.
