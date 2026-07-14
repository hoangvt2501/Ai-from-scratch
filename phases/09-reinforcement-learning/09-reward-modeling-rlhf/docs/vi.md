# Mô hình phần thưởng & RLHF

> Con người không thể viết một hàm phần thưởng cho "phản hồi trợ lý tốt", nhưng họ có thể so sánh hai câu trả lời và chọn câu trả lời tốt hơn. Phù hợp với một model phần thưởng cho những so sánh đó, sau đó RL ngôn ngữ model chống lại nó. Christiano 2017. Hướng dẫn GPT 2022. Công thức biến GPT-3 thành ChatGPT. Vào năm 2026, nó chủ yếu được thay thế bằng DPO - nhưng model tinh thần vẫn còn.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 05 (Tình cảm), Giai đoạn 9 · 08 (PPO)
**Thời lượng:** ~45 phút

## Vấn đề

Bạn đã huấn luyện một model ngôn ngữ về mục tiêu dự đoán token tiếp theo. Nó viết tiếng Anh ngữ pháp. Nó cũng nói dối, lan man và từ chối từ chối. Bạn không thể sửa chữa điều này bằng nhiều pretraining hơn - văn bản web là vấn đề, không phải là phương thuốc chữa bệnh.

Bạn muốn một *phần thưởng vô hướng* có nội dung "phản hồi A tốt hơn phản hồi B cho lệnh X". Viết hàm phần thưởng đó bằng tay là không thể. "Hữu ích" không phải là một biểu thức dạng đóng trên tokens. Nhưng con người có thể so sánh hai đầu ra và đánh dấu một sở thích. Điều đó rẻ để thu thập trên quy mô lớn.

RLHF (Christiano và cộng sự 2017; Ouyang và cộng sự 2022) chuyển đổi sở thích thành model phần thưởng, sau đó tối ưu hóa LM thông qua PPO với phần thưởng đó. Trong ba bước: SFT → RM → PPO. Đó là công thức mà shipped ChatGPT, Claude, Gemini và mọi LLM liên kết khác vào năm 2023–2025.

Vào năm 2026, bước PPO chủ yếu được thay thế bằng DPO (Giai đoạn 10 · 08) vì nó rẻ hơn và gần như tốt cho việc điều chỉnh alignment. Nhưng phần thưởng model* vẫn làm nền tảng cho mọi bộ lấy mẫu Best-of-N, mọi pipeline phần thưởng RL từ có thể xác minh và mọi lý do model sử dụng phần thưởng process model. Hiểu RLHF và bạn hiểu toàn bộ alignment stack.

## Khái niệm

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**Giai đoạn 1: Fine-Tuning có giám sát (SFT).** Bắt đầu từ model cơ sở pretrained. Fine-tune dựa trên các minh họa do con người viết về hành vi mục tiêu (phản hồi theo hướng dẫn, câu trả lời hữu ích, v.v.). Kết quả: một model `π_SFT` *thiên về hành vi tốt* nhưng vẫn có không gian hành động không giới hạn.

**Giai đoạn 2: Phần thưởng Model training.**

- Thu thập các cặp phản hồi `(y_+, y_-)` prompts `x`, được con người dán nhãn là "y_+ được ưa thích hơn y_-".
- Huấn luyện một model `R_φ(x, y)` phần thưởng để chỉ định điểm cao hơn cho `y_+`.
- Loss: **hậu cần theo cặp Bradley-Terry**:

`L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

σ là sigmoid. Sự khác biệt về phần thưởng ngụ ý tỷ lệ ưu tiên. BT đã là tiêu chuẩn từ năm 1952 (Bradley-Terry) và là sự lựa chọn thống trị trong RLHF hiện đại.

- `R_φ` thường được khởi tạo từ model SFT với một đầu vô hướng ở trên cùng. Cùng transformer xương sống; một lớp tuyến tính duy nhất xuất ra phần thưởng.

**Giai đoạn 3: PPO đấu với RM với hình phạt KL.**

- Khởi tạo policy `π_θ` có thể huấn luyện từ `π_SFT`. Giữ một `π_ref = π_SFT` * tham chiếu * bị đóng băng.
- Phần thưởng khi kết thúc câu trả lời `y`:

`r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

Hình phạt KL ngăn `π_θ` trôi dạt tùy tiện khỏi `π_SFT` - đó là một *chính quy hóa*, không phải là một khu vực đáng tin cậy. `β` thường `0.01`-`0.05`.
- Chạy PPO (Bài 08) với phần thưởng này. Lợi thế được tính trên quỹ đạo cấp token, nhưng RM chỉ ghi điểm phản hồi đầy đủ.

**Tại sao lại là KL?** Nếu không có nó, PPO sẽ vui vẻ tìm thấy các chiến lược hack phần thưởng - RM chỉ được huấn luyện về các lần hoàn thành trong phân phối. Một phản hồi ngoài phân phối có thể đạt điểm cao hơn bất kỳ phản hồi nào do con người viết. KL giữ `π_θ` gần ống góp nơi RM được huấn luyện. Nó là núm quan trọng nhất trong RLHF.

**Tình trạng năm 2026:**

- **DPO** (Rafailov 2023): đại số dạng đóng thu gọn Giai đoạn 2 + 3 thành một loss duy nhất được giám sát trên dữ liệu ưu tiên. Không có RM, không có PPO. Cùng chất lượng trên alignment benchmarks cho một phần nhỏ của tính toán. Được đề cập trong Giai đoạn 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO với đường cơ sở tương đối nhóm thay vì nhà phê bình, phần thưởng từ *người xác minh* (chạy mã / câu trả lời toán học phù hợp) thay vì RM do con người huấn luyện. Chiếm ưu thế về lý luận models. Được đề cập trong Giai đoạn 9 · 12.
- **Process models phần thưởng (PRM):** điểm các giải pháp từng phần (mỗi bước suy luận), được sử dụng trong cả biến thể RLHF và GRPO để suy luận.
- **AI / RLAIF hiến pháp: **Sử dụng LLM căn chỉnh để tạo tùy chọn thay vì con người. Chia tỷ lệ ngân sách ưu tiên.

## Tự xây dựng

Bài học này sử dụng các "prompts" và "phản hồi" tổng hợp nhỏ được biểu diễn dưới dạng chuỗi. RM là một công cụ ghi điểm tuyến tính trên một biểu diễn túi tokens. Không có LLM thực sự - *hình dạng* của pipeline quan trọng, không phải thang đo. Xem `code/main.py`.

### Bước 1: dữ liệu ưu tiên tổng hợp

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

Trong RLHF thực tế, điều này được thay thế bằng những người dán nhãn của con người. Hình dạng - `(prompt, preferred_response, rejected_response)` - giống hệt nhau.

### Bước 2: Phần thưởng Bradley-Terry model

Điểm tuyến tính: `R(x, y) = w · bag(y)`. Huấn luyện để giảm thiểu loss nhật ký theo cặp BT:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

Sau vài trăm bản cập nhật, `w` gán trọng số dương cho từ tốt tokens và tiêu cực cho xấu.

### Bước 3: policy giống PPO lên trên RM

policy đồ chơi của chúng tôi tạo ra một token duy nhất từ một từ vựng. Chúng ta chấm điểm token dưới RM, tính toán `log π_θ(token | prompt)`, thêm hình phạt KL-to-reference và áp dụng PPO thay thế đã cắt.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Bước 4: giám sát KL

Theo dõi có nghĩa là `KL(π_θ || π_ref)` mọi bản cập nhật. Nếu nó len lỏi qua `~5-10` policy đã trôi dạt xa so với `π_SFT` - `β` thấp hơn đang tăng lên hoặc hack phần thưởng đang bắt đầu. Đây là chẩn đoán hàng đầu trong RLHF thực.

### Bước 5: công thức production với TRL

Một khi bạn hiểu pipeline đồ chơi, đây là vòng lặp tương tự như một người dùng thư viện thực sự viết nó. [TRL](https://huggingface.co/docs/trl) của Hugging Face là triển khai tham chiếu - `RewardTrainer` cho Giai đoạn 2 và `PPOTrainer` (với KL-to-reference được tích hợp sẵn) cho Giai đoạn 3.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

Ba điều thư viện làm cho bạn. `adap_kl_ctrl=True` thực hiện lịch trình β thích ứng: nếu quan sát thấy KL vượt quá `target_kl`, β tăng gấp đôi; nếu dưới một nửa, β một nửa. model tham chiếu bị đóng băng theo quy ước - bạn không được vô tình chia sẻ parameters với `policy`. Và đầu giá trị sống trên cùng một xương sống với policy (`AutoModelForCausalLMWithValueHead` gắn đầu MLP vô hướng), đó là lý do tại sao TRL báo cáo `policy/kl` và `value/loss` riêng biệt.

## Cạm bẫy

- **Tối ưu hóa quá mức / hack phần thưởng.** RM không hoàn hảo; `π_θ` tìm thấy các lần hoàn thành đối nghịch đạt điểm cao nhưng xấu. Các triệu chứng: phần thưởng tăng vô thời hạn trong khi điểm đánh giá của con người ổn định hoặc giảm. Khắc phục: dừng sớm, tăng `β`, mở rộng dữ liệu RM training.
- **Hack độ dài.** Các RM được huấn luyện về các câu trả lời hữu ích thường ngầm thưởng độ dài. policy học cách đệm các câu trả lời. Khắc phục: phần thưởng chuẩn hóa độ dài hoặc RLAIF với RM nhận biết độ dài.
- **RM quá nhỏ.** RM ít nhất cần phải lớn bằng policy. Một RM nhỏ không thể ghi điểm trung thực các kết quả đầu ra của policy.
- **Điều chỉnh KL. **Quá thấp β → trôi dạt và phần thưởng hack. Quá cao β → policy hầu như không thay đổi. Thủ thuật tiêu chuẩn là một β * thích ứng * nhắm mục tiêu KL cố định trên mỗi bước.
- **Nhiễu dữ liệu tùy chọn. **~30% nhãn của con người bị nhiễu hoặc mơ hồ. Hiệu chỉnh bằng cách training RM trên dữ liệu được lọc theo thỏa thuận hoặc sử dụng temperature trên BT.
- **Vấn đề tắt policy.** Dữ liệu PPO hơi lệch policy sau epoch đầu tiên. Theo dõi phân số clip như trong Bài 08.

## Ứng dụng

RLHF vào năm 2026 được phân lớp:

| Lớp | Mục tiêu | Phương pháp |
|-------|--------|--------|
| Hướng dẫn theo dõi, hữu ích, vô hại | Alignment | DPO (Giai đoạn 10 · 08) được ưu tiên hơn RLHF-PPO. |
| Tính đúng đắn của suy luận (toán, mã) | Khả năng | GRPO với phần thưởng người xác minh (Giai đoạn 9 · 12). |
| Nhiệm vụ nhiều bước dài | Agentic | PPO / GRPO với phần thưởng process models qua các bước. |
| Hành vi an toàn / từ chối | Sự An Toàn | RLHF-PPO với RM an toàn riêng biệt, hoặc AI Hiến pháp. |
| Best-of-N tại inference | alignment nhanh | Sử dụng RM tại thời điểm giải mã; không cần policy training. |
| Phần thưởng distillation | Điện toán Inference | Huấn luyện một "đầu phần thưởng" nhỏ trên một LM bị đóng băng. |

RLHF là phương pháp vào năm 2022–2024. Vào năm 2026, production alignment pipelines là DPO đầu tiên, chỉ PPO cho các bước chuyên sâu về RM hoặc quan trọng về an toàn.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Bài tập

1. **Dễ dàng.** Huấn luyện phần thưởng Bradley-Terry model trong `code/main.py` trên 500 cặp ưu tiên tổng hợp. Đo theo cặp accuracy trên 100 cặp được giữ lại. Nên vượt quá 90%.
2. **Trung bình.** Chạy vòng lặp RLHF PPO đồ chơi với `β ∈ {0.0, 0.1, 1.0}`. Đối với mỗi người, hãy vẽ điểm RM so với KL-to-reference trên các bản cập nhật. Cái nào chạy hack phần thưởng?
3. **Khó.** Triển khai DPO (tùy chọn dạng đóng-likelihood loss) trên cùng một dữ liệu ưu tiên và so sánh với RLHF-PPO pipeline tính toán được sử dụng và điểm RM cuối cùng đạt được.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| RLHF | "Alignment RL" | SFT ba giai đoạn + RM + PPO pipeline (Christiano 2017, Ouyang 2022). |
| Phần thưởng Model (RM) | "Lưới ghi bàn" | Hàm vô hướng đã học phù hợp với các tùy chọn theo cặp thông qua Bradley-Terry. |
| Bradley-Terry | "loss hậu cần theo cặp" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; mục tiêu RM tiêu chuẩn. |
| Hình phạt KL | "Ở gần tài liệu tham khảo" | `β · KL (π_θ \ | \ | π_ref)' trong phần thưởng; bộ chính quy chống hack phần thưởng. |
| Hack phần thưởng | "Định luật Goodhart" | Policy khai thác các lỗ hổng RM; triệu chứng: phần thưởng lên, đánh giá con người phẳng. |
| RLAIF | "Tùy chọn được gắn nhãn AI" | RLHF nơi các nhãn đến từ một LM khác thay vì con người. |
| PRM | "Phần thưởng Process Model" | Chấm điểm các bước suy luận một phần; được sử dụng trong pipelines lý luận. |
| AI hiến pháp | "Phương pháp của Anthropic" | Tùy chọn do AI tạo được hướng dẫn bởi các quy tắc rõ ràng. |

## Đọc thêm

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) - tờ báo bắt đầu RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - công thức đằng sau ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — RLHF trước đó để tóm tắt.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - DPO; vỡ nợ sau RLHF vào năm 2026.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) - vòng lặp RLAIF và tự phê bình.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) - bài báo HH.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) - production `RewardTrainer` và `PPOTrainer`. Đọc nguồn huấn luyện viên để biết chi tiết về KL thích ứng và đầu giá trị.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) bởi Lambert, Castricato, von Werra, Havrilla - hướng dẫn kinh điển của pipeline ba giai đoạn với sơ đồ.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) - thư viện; `examples/` có RLHF scripts đầu cuối cho Llama, Mistral và Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) - quan điểm giả thuyết phần thưởng; điều kiện tiên quyết cần thiết để suy nghĩ về hack phần thưởng.
