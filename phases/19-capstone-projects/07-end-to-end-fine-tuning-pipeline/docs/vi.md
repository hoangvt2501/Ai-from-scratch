# Capstone 07 — Fine-Tuning Pipeline đầu cuối (Dữ liệu đến SFT đến DPO để phân phối)

> Một model 8B được huấn luyện trên dữ liệu của riêng bạn, DPO phù hợp với sở thích của riêng bạn, được lượng tử hóa, giải mã suy đoán và được phục vụ với tokens 1 triệu đô la có thể đo lường được. stack mở năm 2026 là Axolotl v0.8, TRL 0.15, Unsloth để lặp lại, GPTQ/AWQ/GGUF cho quantization, vLLM 0.7 với EAGLE-3 để giao bóng. Điểm mấu chốt là chạy toàn bộ pipeline một cách có thể lặp lại - YAML vào, phục vụ endpoint ra - và xuất bản thẻ model theo Framework Mở Model năm 2026.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (pipeline), YAML (configs), Bash (scripts)
**Kiến thức tiên quyết:** Giai đoạn 2 (ML), Giai đoạn 3 (DL), Giai đoạn 7 (transformers), Giai đoạn 10 (LLMs từ đầu), Giai đoạn 11 (LLM kỹ thuật), Giai đoạn 17 (cơ sở hạ tầng), Giai đoạn 18 (an toàn)
**Các giai đoạn được thực hiện: **P2 · P3 · P7 · P10 · P11 · P17 · Trang 18
**Thời lượng:** 35 giờ

## Vấn đề

Mọi đội AI nghiêm túc vào năm 2026 đều có một fine-tuning pipeline được khai thác. Không phải vì họ ship một model cơ sở biên giới, mà bởi vì sự thích ứng hạ nguồn - SFT miền, DPO chống lại các sở thích được dán nhãn, các bản nháp chưng cất để giải mã đầu cơ, phục vụ với EAGLE-3 - là nơi các chiến thắng có thể đo lường được. Axolotl v0.8 xử lý các cấu hình SFT đa GPU. TRL 0.15 xử lý DPO và GRPO. Unsloth giúp bạn lặp lại một GPU nhanh chóng. vLLM 0.7 với EAGLE-3 đẩy giải mã thông lượng 2-3x mà không cần loss chất lượng. Dụng cụ hoạt động; thủ công nằm trong YAML, vệ sinh dữ liệu và kỷ luật đánh giá.

Bạn sẽ chạy cơ sở 8B (Llama 3.3, Qwen3 hoặc Gemma 3) thông qua SFT, sau đó DPO dữ liệu theo nhiệm vụ cụ thể, lượng tử hóa để phân phát và đo lường mức tăng so với lm-evaluation-harness, RewardBench-2, MT-Bench-v2 và MMLU-Pro. Bạn sẽ tạo ra một thẻ model theo Framework Mở Model 2026. Vấn đề là khả năng tái tạo - một lệnh chạy lại toàn bộ pipeline từ đầu đến cuối.

## Khái niệm

pipeline có năm giai đoạn. **Dữ liệu**: dedup (MinHash / Datatrove), bộ lọc chất lượng (bộ phân loại kiểu Nemotron-CC), tẩy tế bào chết PII, kiểm tra vệ sinh phân chia chống ô nhiễm benchmark công cộng. **SFT**: Axolotl YAML, ZeRO-3 trên 8xH100, lịch cosin, trình tự đóng gói, 2-3 epochs. **DPO hoặc GRPO**: TRL config, 1 epoch, các cặp ưu tiên được gắn nhãn con người hoặc model đánh giá, điều chỉnh beta. **Lượng hóa**: GPTQ + AWQ + GGUF để triển khai linh hoạt. **Phục vụ**: vLLM 0.7 với đầu đầu cơ EAGLE-3 (hoặc SGLang với SpecForge), triển khai K8s, HPA khi chờ đợi.

Cắt bỏ là sản phẩm: chỉ SFT so với SFT + DPO so với SFT + GRPO trên ba benchmarks nhiệm vụ cụ thể. Các chỉ số phục vụ: tokens/s ở batch 1/8/32, tỷ lệ chấp nhận EAGLE-3, tokens $/1M. Đánh giá an toàn: Tỷ lệ vượt qua Llama Guard 4. Thẻ Model: đánh giá bias, hạt giống khả năng tái tạo, cấp phép dữ liệu.

## Kiến trúc

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## Stack

- Dữ liệu: Datatrove để dedup, bộ phân loại Nemotron-CC về chất lượng, Presidio cho PII
- Cơ sở: Llama 3.3 8B, Qwen3 14B hoặc Gemma 3 12B
- SFT: Axolotl v0.8 với ZeRO-3, Flash Attention 3, trình tự đóng gói
- Điều chỉnh tùy chọn: TRL 0.15 cho DPO hoặc GRPO; Unsloth để lặp lại một GPU
- Quantization: GPTQ (Marlin), AWQ, GGUF qua llama.cpp
- Phục vụ: vLLM 0.7 với giải mã đầu cơ EAGLE-3 (hoặc SGLang 0.4 + SpecForge)
- Đánh giá: lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro
- Đánh giá an toàn: Llama Guard 4, ShieldGemma-2
- Cơ sở hạ tầng: Plugin thiết bị Kubernetes + NVIDIA, HPA trên chỉ số chờ đợi
- Observability: W & B cho training, Langfuse cho inference

## Tự xây dựng

1. **Data pipeline.** Chạy Datatrove dedup trên kho dữ liệu thô. Áp dụng bộ phân loại chất lượng kiểu Nemotron-CC. Presidio tẩy tế bào chết PII. Viết train/val phân tách với hạt giống rõ ràng.

2. **Kiểm tra ô nhiễm.** Đối với mỗi lần phân tách xác thực, hãy tính toán MinHash so với các bộ thử nghiệm MMLU-Pro, MT-Bench-v2, RewardBench-2. Từ chối bất kỳ sự trùng lặp nào.

3. **Axolotl SFT.** YAML với ZeRO-3, FA3, đóng gói theo trình tự. 2-3 epochs trên 8xH100. Đăng nhập vào W&B.

4. **TRL DPO / GRPO.** Tham gia checkpoint SFT, chạy một epoch DPO trên các cặp ưu tiên (hoặc GRPO có phần thưởng có thể xác minh trên math/code). Quét beta.

5. **Định lượng.** Tạo ra ba định lượng: GPTQ-INT4-Marlin, AWQ-INT4, GGUF-Q4_K_M cho llama.cpp. Kích thước bản ghi và thông lượng danh nghĩa.

6. **Phục vụ với giải mã đầu cơ.** vLLM 0.7 config với các đầu dự thảo EAGLE-3 được huấn luyện thông qua Red Hat Speculators. Đo tỷ lệ chấp nhận và độ trễ đuôi ở batch 1/8/32. Báo cáo $ / 1M tokens so với Anthropic / OpenAI trên cùng một đánh giá.

7. **Ma trận đánh giá.** Chạy lm-eval-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro trên cơ sở, chỉ SFT, SFT+DPO, SFT+GRPO. Tạo một bảng.

8. **Đánh giá an toàn.** Tỷ lệ vượt qua Llama Guard 4 trên bộ phát triển. Bộ lọc đầu ra ShieldGemma-2.

9. **Thẻ Model.** Mẫu MOF 2026: dữ liệu, training, đánh giá, an toàn, giấy phép, phần khả năng tái tạo với YAML và commit SHA.

## Ứng dụng

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## Sản phẩm bàn giao

`outputs/skill-finetuning-pipeline.md` mô tả sản phẩm. Một lệnh duy nhất chạy dữ liệu thông qua SFT thông qua DPO qua định lượng thông qua giao bóng thông qua đánh giá và phát ra thẻ model + endpoint được phục vụ.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | Eval delta so với cơ sở | Mức tăng đo được trên các nhiệm vụ mục tiêu (MMLU-Pro, MT-Bench-v2, nhiệm vụ cụ thể) |
| 20 | Pipeline khả năng tái tạo | Một lệnh chạy lại từ đầu đến cuối với các hạt giống giống hệt nhau |
| 20 | Vệ sinh dữ liệu | Tỷ lệ khử trùng, độ phủ chà PII, kiểm tra ô nhiễm màu xanh lá cây |
| 20 | Phục vụ hiệu quả | tokens/s ở bs = 1/8/32, tỷ lệ chấp nhận EAGLE-3, $ / 1M tokens |
| 15 | Thẻ Model + đánh giá an toàn | Độ hoàn thiện MOF năm 2026 + Tỷ lệ vượt qua Llama Guard 4 |
| **100** |||

## Bài tập

1. Chạy chỉ SFT so với SFT+DPO so với SFT+GRPO trên cùng một benchmark dành riêng cho nhiệm vụ. Báo cáo phương thức ưu tiên nào thắng và bằng bao nhiêu.

2. Hoán đổi Llama 3.3 8B lấy Qwen3 14B. Đo lường tokens 1 triệu đô la ở chất lượng phù hợp.

3. Đo lường tỷ lệ chấp nhận EAGLE-3 trên dữ liệu miền so với ShareGPT chung. Báo cáo delta và ý nghĩa của nó đối với ngân sách độ trễ.

4. Tiêm 1% ô nhiễm (rò rỉ câu trả lời MMLU-Pro vào dữ liệu training) và chạy lại đánh giá. Xem MMLU-Pro accuracy nhảy một cách phi thực tế. Xây dựng một cổng CI kiểm tra ô nhiễm để bắt được điều này.

5. Thêm SFT LoRA thay thế cho fine-tune đầy đủ. Đo khoảng cách chất lượng ở bộ nhớ thấp hơn 10 lần.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Axolotl | "Huấn luyện viên SFT" | Huấn luyện viên điều khiển YAML hợp nhất cho SFT, DPO và distillation |
| TRL | "Bộ dò tùy chọn" | Hugging Face thư viện cho DPO, GRPO PPO trên LLMs |
| GRPO | "Tối ưu hóa policy tương đối nhóm" | Công thức RL của DeepSeek R1 với phần thưởng có thể xác minh |
| ĐẠI BÀNG-3 | "Dự thảo giải mã suy đoán" | Dự thảo đầu dự đoán N tokens phía trước; vLLM xác minh với model mục tiêu |
| Bộ Tài chính | "Model Cởi mở Framework" | Tiêu chuẩn 2026 để phân loại các bản phát hành model về dữ liệu, mã, giấy phép |
| Kiểm tra ô nhiễm | "Vệ sinh phân chia" | Phát hiện rò rỉ bộ thử nghiệm vào training dựa trên MinHash |
| Tỷ lệ chấp nhận | "Số liệu EAGLE / MTP" | Phần tokens dự thảo mà mục tiêu model chấp nhận |

## Đọc thêm

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) - huấn luyện viên SFT / DPO tham chiếu
- [TRL documentation](https://huggingface.co/docs/trl) — triển khai tham chiếu DPO và GRPO
- [Unsloth](https://github.com/unslothai/unsloth) — tham chiếu lặp lại một GPU
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) - GRPO phương pháp luận
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — stack phục vụ tham khảo
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — huấn luyện viên giải mã suy đoán thay thế
- [Model Openness Framework 2026](https://isocpp.org/) — tiêu chuẩn phân loại phát hành mở
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — Canonical Eval Runner
