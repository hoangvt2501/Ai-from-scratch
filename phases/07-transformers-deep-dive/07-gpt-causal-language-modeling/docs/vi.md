# GPT — Mô hình ngôn ngữ nhân quả

> BERT nhìn thấy cả hai phía. GPT chỉ nhìn thấy quá khứ. Mặt nạ tam giác là dòng mã đơn lẻ quan trọng nhất trong AI hiện đại.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 02 (Self-Attention), Giai đoạn 7 · 05 (Transformer đầy đủ), Giai đoạn 7 · 06 (BERT)
**Thời lượng:** ~75 phút

## Vấn đề

Một ngôn ngữ model trả lời một câu hỏi: cho `t-1` tokens đầu tiên, phân phối xác suất trên token `t` là gì? Huấn luyện về tín hiệu đó - dự đoán token tiếp theo - và bạn sẽ nhận được một model có thể tạo ra văn bản tùy ý từng token một.

Để huấn luyện nó từ đầu đến cuối trên toàn bộ chuỗi song song, bạn cần dự đoán của mỗi vị trí chỉ phụ thuộc vào các vị trí trước đó. Nếu không, model gian lận một cách tầm thường bằng cách nhìn vào câu trả lời.

Mặt nạ nhân quả làm điều này. Nó là một ma trận tam giác trên duy nhất gồm các giá trị `-inf` được thêm vào điểm số attention trước softmax. Sau softmax, các vị trí đó trở thành 0. Mỗi vị trí chỉ có thể tham gia vào chính nó và các vị trí trước đó. Và bởi vì bạn áp dụng nó một lần cho toàn bộ dãy, bạn sẽ nhận được N dự đoán token tiếp theo song song trong một forward pass.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2024), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi - tất cả đều là transformers nhân quả chỉ decoder với cùng một vòng lõi. Chỉ cần dữ liệu lớn hơn, tốt hơn và RLHF tốt hơn.

## Khái niệm

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### Mặt nạ

Cho một chuỗi độ dài `N`, hãy xây dựng một ma trận `N × N`:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

Thêm `M` vào điểm attention thô trước khi softmax. `exp(-inf) = 0`, vì vậy các tư thế đeo mặt nạ không đóng góp trọng lượng. Mỗi hàng của ma trận attention chỉ là phân phối xác suất trên các vị trí trước đó.

Chi phí thực hiện: một cuộc gọi `torch.tril()`. Thời gian tính toán: nano giây. Tác động trên sân: mọi thứ.

### training song song, inference nối tiếp

Training: chuyển tiếp toàn bộ dãy `(N, d_model)` một lần, tính toán N tổn thất entropy chéo (một lần cho mỗi vị trí), tổng, backprop. Song song dọc theo trình tự. Đây là lý do tại sao GPT training thang đo - bạn process 1 triệu tokens trong một batch trong một GPU đường chuyền.

Inference: bạn tạo ra token theo token. Cho người `[t1, t2, t3]` ăn, lấy `t4`. Cho ăn `[t1, t2, t3, t4]`, lấy `t5`. Cho ăn `[t1, t2, t3, t4, t5]`, lấy `t6`. KV cache (Bài 12) lưu các trạng thái ẩn của `t1…tn` để bạn không phải tính toán lại chúng từng bước. Nhưng độ sâu nối tiếp ở inference = chiều dài đầu ra. Đó là thuế tự hồi quy và tại sao giải mã là nút thắt cổ chai về độ trễ của mọi LLM.

### The loss - từng ca một

Đưa ra tokens `[t1, t2, t3, t4]`:

- Đầu vào: `[t1, t2, t3]`
- Mục tiêu: `[t2, t3, t4]`

Đối với mỗi vị trí `i`, hãy tính toán `-log P(target_i | inputs[:i+1])`. Tổng. Đây là entropy chéo cho toàn bộ chuỗi.

Mỗi transformer LM bạn đã nghe nói về các chuyến tàu trên loss này. Pre-training, fine-tuning, SFT - cùng loss, dữ liệu khác nhau.

### Chiến lược giải mã

Sau training, sampling lựa chọn quan trọng hơn mọi người nghĩ.

| Phương pháp | Chức năng | Trường hợp sử dụng |
|--------|--------------|-------------|
| Tham lam | Argmax từng bước | Nhiệm vụ xác định, hoàn thành mã |
| Temperature | Chia logits cho T, mẫu | Nhiệm vụ sáng tạo, T cao hơn = đa dạng hơn |
| Top-k | Chỉ lấy mẫu từ top-k tokens | Giết đuôi xác suất thấp |
| Top-p (nhân) | Mẫu từ tập hợp nhỏ nhất với prob tích lũy ≥ p | 2020+ mặc định; Thích ứng với hình dạng phân phối |
| Tối thiểu-p | Giữ tokens với `p > min_p * max_p` | 2024+; giỏi hơn trong việc loại bỏ đuôi dài hơn top-p |
| Giải mã suy đoán | Dự thảo model đề xuất N tokens, model lớn xác minh | Giảm độ trễ 2–3× ở cùng chất lượng |

Vào năm 2026, min-p + temperature 0,7 là giá trị mặc định hợp lý cho các models trọng số mở. Giải mã đầu cơ là tiền đặt cược cho bất kỳ production inference stack nào.

### Điều gì đã làm cho "công thức GPT" hoạt động

1. **Chỉ Decoder.** Không encoder chi phí. Một lần attention + FFN mỗi lớp.
2. **Mở rộng quy mô.** 124 triệu → 1,5 tỷ → 175 tỷ → nghìn tỷ. Định luật tỷ lệ Chinchilla (Bài 13) cho bạn biết cách sử dụng điện toán.
3. **Học tập trong ngữ cảnh.** Xuất hiện vào khoảng 6B–13B. Các model có thể làm theo few-shot ví dụ mà không cần fine-tuning.
4. **RLHF.** Post-training về sở thích của con người đã chuyển đổi văn bản pretrained thô thành trợ lý trò chuyện.
5. **Định mức trước + RoPE + SwiGLU.** training ổn định trên quy mô lớn.

Kiến trúc cốt lõi không thay đổi nhiều kể từ năm GPT-2. Mọi thứ thú vị đã xảy ra trong dữ liệu, quy mô và hậu training.

```figure
causal-mask
```

## Tự xây dựng

### Bước 1: mặt nạ nhân quả

Xem `code/main.py`. Một dòng:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Thêm nó vào attention điểm trước khi softmax. Đó là toàn bộ cơ chế.

### Bước 2: GPT-ish model 2 lớp

Stack hai khối decoder (mặt nạ self-attention + FFN, không có cross-attention). Thêm token embedding, mã hóa vị trí và bỏ nhúng (gắn với ma trận token embedding - một thủ thuật tiêu chuẩn kể từ năm GPT-2).

### Bước 3: Dự đoán token tiếp theo, từ đầu đến cuối

Trên từ vựng đồ chơi 20 token, tạo ra logits ở mọi vị trí. Tính toán entropy chéo loss với mục tiêu dịch chuyển từng một. Không gradient - đây là một kiểm tra sự tỉnh táo chuyển tiếp.

### Bước 4: sampling

Thực hiện tham lam, temperature, top-k, top-p, min-p. Chạy từng trên một prompt cố định và so sánh đầu ra. Một hàm sampling là 10 dòng.

## Ứng dụng

PyTorch, 2026 thành ngữ:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

Dưới mui xe, `generate()` chạy forward pass, kéo logits vị trí cuối cùng, lấy mẫu token tiếp theo, nối nó và lặp lại. Mỗi production LLM inference stack (vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX) triển khai cùng một vòng lặp với tối ưu hóa nặng — điền trước hàng loạt, hàng loạt liên tục, phân trang KV cache, giải mã suy đoán.

**GPT vs BERT, mỗi dòng một dòng: **GPT dự đoán `P(x_t | x_{<t})`. BERT dự đoán `P(x_masked | x_unmasked)`. loss xác định xem model có thể tạo ra hay không.

## Sản phẩm bàn giao

Xem `outputs/skill-sampling-tuner.md`. skill chọn sampling parameters cho tác vụ thế hệ mới và gắn cờ khi cần giải mã xác định.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py` và xác minh ma trận attention nhân quả là hình tam giác thấp hơn sau khi softmax. Kiểm tra tại chỗ: hàng 3 chỉ nên có trọng số trong cột 0–3.
2. **Trung bình.** Thực hiện beam search cho chiều rộng 4. So sánh perplexity của chùm tia-4 và tham lam trên 10 prompts ngắn. Chùm tia có luôn chiến thắng không? (Gợi ý: thường để dịch, không phải để trò chuyện mở.)
3. **Khó.** Thực hiện giải mã suy đoán: sử dụng một model 2 lớp nhỏ làm bản nháp và một model 6 lớp làm trình xác minh. Đo tốc độ đồng hồ treo tường trên 100 lần hoàn thành độ dài 64. Xác nhận kết quả đầu ra phù hợp với lòng tham của người xác minh.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Mặt nạ nhân quả | "Tam giác" | Ma trận `-inf` hình tam giác trên được thêm vào điểm số attention để vị trí `i` chỉ nhìn thấy các vị trí `≤ i`. |
| Dự đoán token tiếp theo | "Người loss" | Entropy chéo của phân bố model so với token tiếp theo thực sự ở mọi vị trí. |
| Tự hồi quy | "Tạo từng cái một" | Đầu ra nguồn cấp dữ liệu trở lại dưới dạng đầu vào; song song chỉ trong training, không phải trong thế hệ. |
| Logits | "Điểm trước khi softmax" | Sản lượng thô của đầu LM trước khi softmax; sampling xảy ra trên những điều này. |
| Temperature | "Núm sáng tạo" | Chia logits cho T; T→0 = tham lam, T→∞ = đồng phục. |
| Top-p | "Hạt nhân sampling" | Cắt bớt phân phối thành tập nhỏ nhất tổng thành ≥p; mẫu từ những gì còn lại. |
| Tối thiểu-p | "Tốt hơn top-p" | Giữ tokens nơi `p ≥ min_p × max_p`; thích ứng với mức cắt theo độ sắc nét của phân phối. |
| Giải mã suy đoán | "Soạn thảo + xác minh" | model giá rẻ đề xuất N tokens; model lớn xác minh song song. |
| Giáo viên ép buộc | "Training mánh khóe" | Trong quá trình training, hãy cung cấp token trước đó thực sự, không phải dự đoán của model. Tiêu chuẩn cho mọi LM seq2seq. |

## Đọc thêm

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - học GPT-3 và trong ngữ cảnh.
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) - giấy giải mã thông số kỹ thuật.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — mã tham chiếu LM nhân quả chính tắc.
