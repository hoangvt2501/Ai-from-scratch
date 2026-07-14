# Luật tỷ lệ

> Bài báo Kaplan năm 2020 cho biết: model lớn hơn, loss thấp hơn. Bài báo Hoffmann năm 2022 cho biết: bạn chưa đủ training. Compute được chia thành hai nhóm - parameters và tokens - và sự phân chia không rõ ràng.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 05 (Transformer đầy đủ), Giai đoạn 7 · 07 (GPT)
**Thời lượng:** ~45 phút

## Vấn đề

Khi bạn có C FLOPs tính toán training và muốn có model tốt nhất, bạn phải đối mặt với hai núm:

1. **Có bao nhiêu parameters (N)?** model lớn hơn, dung lượng cao hơn.
2. **Có bao nhiêu training tokens (D)?** Nhiều dữ liệu hơn, sử dụng dung lượng tốt hơn.

FLOPs thang đo xấp xỉ `6 × N × D`. Bạn có thể đẩy N lên và D xuống, hoặc D lên và N xuống. Cái nào tốt hơn?

Trước năm 2022, câu trả lời là "đẩy mạnh N". GPT-3 (2020) là 175B parameters được huấn luyện trên ~300B tokens. Tỷ lệ khoảng 1,7 tokens trên parameter. Luật mở rộng quy mô Kaplan đã hỗ trợ điều này.

Hoffmann et al. (2022), training một gia đình nhỏ models có tên là Chinchilla, đã tìm thấy một điều khác biệt: tỷ lệ tối ưu gần với **20 tokens trên parameter**. GPT-3 chưa được huấn luyện 10×. Chinchilla (70B params, 1.4T tokens) đánh bại GPT-3 (175B, 300B tokens) trên mỗi benchmark với chi phí thấp hơn 2.5× inference.

Năm 2026 là thế giới của Chinchilla - với một bước ngoặt quan trọng. Llama 3 8B được huấn luyện trên 15 nghìn tỷ tokens, tỷ lệ 1.875 tokens mỗi parameter. Chín mươi bốn lần vượt qua Chinchilla-tối ưu. Inference chi phí quan trọng hơn chi phí training cho models sẽ được sử dụng trên quy mô lớn, vì vậy quá training (quá khứ Chinchilla) cho một dấu chân có thể triển khai nhỏ hơn là mặc định vào năm 2026.

## Khái niệm

![Chinchilla curves: loss vs compute at various N/D ratios](../assets/scaling-laws.svg)

### Định luật Hoffmann

Từ bài báo Chinchilla, loss như sau:

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = parameters (không embedding).
- `D` = training tokens.
- `α ≈ 0.34`, `β ≈ 0.28` (gần đối xứng).
- `E ≈ 1.69`, trần loss không thể rút gọn.
- `A ≈ 406`, `B ≈ 411`.

Hai thuật ngữ giao dịch với nhau khi bạn mở rộng quy mô. Lấy đạo hàm w.r.t. `N` ở tính toán cố định (C = 6ND) và giải:

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

Tối ưu điện toán: 20 tokens mỗi parameter.

### Tại sao lại training quá mức

Chinchilla-tối ưu giảm thiểu training loss trên training FLOP. Nhưng bạn phải trả chi phí training một lần; inference giá mãi mãi.

Đối với một chatbot phục vụ một nghìn tỷ tokens mỗi tháng, inference chiếm ưu thế trong tổng chi phí. Cách tiếp cận của Llama: huấn luyện nhỏ hơn, dài hơn. 8B ở 15T tokens được tối ưu hóa inference sâu:

- Phù hợp với GPUs tiêu dùng.
- Độ trễ là một phần nhỏ của 70B Chinchilla-optimal.
- Chất lượng đủ gần cho hầu hết các tác vụ.

Bài báo năm 2024 của DeepMind ("Over-training là tối ưu mới") đã chính thức hóa điều này. Đối với khối lượng công việc do inference thống trị, tỷ lệ phù hợp là gần 100–500 tokens mỗi parameter tùy thuộc vào volume phục vụ.

### Sự xuất hiện vs sự mượt mà

Tuyên bố: một số khả năng nhất định (số học, suy luận nhiều bước, chain-of-thought theo dõi) "xuất hiện" đột ngột ở một quy mô nào đó.

Schaeffer et al. (2023) lập luận rằng đây là một artifact đo lường: các chỉ số mới nổi sử dụng tính điểm không liên tục (khớp chính xác, accuracy ở ngưỡng) để che giấu sự cải thiện mượt mà trong logits cơ bản. Các chỉ số liên tục (entropy chéo) hiển thị các đường cong mượt mà.

Vào năm 2026, sự đồng thuận là: dự đoán thông qua loss liên tục là đáng tin cậy. Benchmark nhảy thường là artifacts ghi bàn. Lập kế hoạch ngân sách dựa trên các chỉ số liên tục.

### Bức tranh năm 2026

Luật mở rộng quy mô vẫn hoạt động, nhưng:

| Yếu tố | Thay đổi cách thức |
|--------|-------------|
| Chất lượng dữ liệu | Quản lý tokens "tốt" (kiểu Phi) dịch chuyển đường cong bằng >2× tính toán hiệu quả |
| MoE | Tổng số tham số tách rời khỏi FLOPs hoạt động; quy luật tỷ lệ cho mỗi hoạt động-FLOP |
| Sau training | Một số khả năng (theo hướng dẫn, mã) thay đổi với SFT+RLHF hơn pretraining |
| Đa phương thức | Hình ảnh + văn bản tokens tỷ lệ với nhau; Các đường cong riêng biệt cho mỗi phương thức |
| Dữ liệu tổng hợp | Models tạo dữ liệu training; Tính toán hiệu quả có thể kết hợp |

Muon optimizer (Kimi Moonlight, 2024) cho thấy mức tăng tính toán hiệu quả ~2× so với AdamW ở dữ liệu phù hợp. Một số lần chạy training năm 2026 sử dụng Muon theo mặc định. Thay đổi hằng số tuyệt đối trong định luật tỷ lệ, không phải hình dạng của nó.

```figure
scaling-laws
```

## Tự xây dựng

Xem `code/main.py`. Chúng ta triển khai phương trình loss Chinchilla và giải quyết `(N, D)` tối ưu tính toán ở mỗi ngân sách điện toán.

### Bước 1: Chinchilla loss

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

Vẽ `L` như một đường viền trên `(N, D)` ở `C = 6ND` cố định. Tìm mức tối thiểu.

### Bước 2: biên giới tối ưu điện toán

Đối với ngân sách tính toán từ `1e17` đến `1e25` FLOPs, hãy tìm `(N, D)` giảm thiểu loss phải `6ND = C`. Xác minh tỷ lệ `D/N ≈ 20`.

### Bước 3: chi phí quá training

Tính thêm loss bạn phải trả để huấn luyện một model nhỏ hơn 10× (1/10 N tối ưu, 10× D tối ưu). Báo cáo khoản tiết kiệm FLOP inference (tỷ lệ với N) để đổi lại.

### Bước 4: so sánh với models thực tế

Giảm các cặp `(N, D)` đã biết cho GPT-3, Chinchilla, Llama 3 8B, DeepSeek-V3 (tham số hoạt động) và so sánh loss dự đoán so với báo cáo.

## Ứng dụng

Bạn không có khả năng huấn luyện một biên giới model chính mình. Nhưng luật tỷ lệ cho bạn biết:

1. **Liệu fine-tune của bạn có đủ dữ liệu hay không.** Nếu dữ liệu cụ thể của nhiệm vụ của bạn dưới 20 tokens trên mỗi thông số của model cơ sở, hãy mong đợi độ bão hòa ở một số tầng loss.
2. **Có nên chọn một model cơ sở lớn hơn hay không.** Nếu bạn đang dành tất cả ngân sách của mình cho inference, hãy thích một model nhỏ hơn, được huấn luyện lâu hơn.
3. **Nơi lợi nhuận giảm dần.** Ngoài 1000× Chinchilla tối ưu, các thay đổi log-loss trở thành nhiễu.

**Quỹ đạo nghiên cứu vào năm 2026:**

- **Chế độ hạn chế dữ liệu.** Web có một số lượng hữu hạn tokens chất lượng cao (~5–10 nghìn tỷ tiếng Anh sau khi lọc). Frontier pretraining đang tiến gần đến mức trần này. Dữ liệu tổng hợp, đa ngôn ngữ, đa phương thức và fine-tuning quy mô RLHF là những đòn bẩy tiếp theo.
- **Thủ thuật tính toán.** Muon optimizer, MoE, quản lý dữ liệu tốt hơn - mỗi thủ thuật thay đổi các hằng số tuyệt đối, không phải tiệm cận.
- **Luật tỷ lệ cho RL.** Câu hỏi mở. Bằng chứng ban đầu cho thấy định luật lũy thừa trong các mẫu RL nhưng với số mũ rất khác so với pretraining.

## Sản phẩm bàn giao

Xem `outputs/skill-training-budget-estimator.md`. skill chọn `(N, D, hours, GPU)` cho một training chạy mới dựa trên ngân sách điện toán, hạn chế triển khai và loss mục tiêu.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. In `(N, D)` tối ưu Chinchilla cho ngân sách điện toán `1e20`, `1e22` `1e24`. So sánh với bảng model thực.
2. **Trung bình.** Triển khai đường cong Hoffmann loss-as-function-of-compute. Vẽ loss so với `log10(C)` cho biên giới tối ưu điện toán. Xác định khi nào luật dự đoán chúng ta sẽ cần `>10^28` FLOPs để giảm 0,1 entropy chéo tiếp theo.
3. **Khó.** Phù hợp với luật tỷ lệ của riêng bạn trên 5 models nhỏ (100K đến 10M tham số) được huấn luyện trên cùng một dataset. Ước tính `α` và `E`. Số mũ của bạn phù hợp với số mũ đã xuất bản như thế nào?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Parameters (N) | "Kích thước Model" | Số lượng trọng lượng không embedding; xác định công suất. |
| Tokens (D) | "Training dữ liệu" | Số training tokens được nhìn thấy; xác định mức độ sử dụng parameters. |
| Điện toán (C) | "FLOPs chi tiêu" | Khoảng `6 × N × D` cho một transformer tiêu chuẩn. |
| Chinchilla-tối ưu | "D/N ≈ 20" | Tỷ lệ giảm thiểu loss trên mỗi FLOP của pretraining. |
| Quá training | "Chinchilla trong quá khứ" | Chi thêm training FLOPs để tiết kiệm inference FLOPs; D/N >> 20. |
| loss không thể rút gọn | "Sàn nhà" | Thuật ngữ `E` trong luật tỷ lệ; entropy của chính dữ liệu. |
| Khả năng nổi lên | "Đột ngột nhảy vọt trên quy mô lớn" | Thường là một cầu thủ ghi bàn artifact; loss liên tục trơn tru. |
| Điện toán hiệu quả | "Hệ số hiệu quả Training" | Dữ liệu / optimizer / kiến trúc tốt hơn nhân lên mức độ FLOP đi được. |

## Đọc thêm

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - bài báo luật tỷ lệ đầu tiên; chưa được huấn luyện.
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Chinchilla.
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — sự xuất hiện như một artifact đo lường.
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) - tại sao training quá mức của Llama lại phù hợp với khối lượng công việc của nó.
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) - Hệ số điện toán 2×.
