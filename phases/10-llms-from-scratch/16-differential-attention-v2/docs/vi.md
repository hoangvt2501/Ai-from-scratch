# Attention vi sai (V2)

> Softmax attention trải một lượng nhỏ xác suất trên mọi token không khớp. Hơn 100k tokens nhiễu đó cộng lại và nhấn chìm tín hiệu. Transformer vi sai (Ye và cộng sự, ICLR 2025) khắc phục nó bằng cách tính attention dưới dạng chênh lệch của hai cực đại mềm, trừ đi sàn nhiễu được chia sẻ. DIFF V2 (Microsoft, tháng 1 năm 2026) là bản viết lại production stack: khớp độ trễ giải mã với Transformer cơ sở, không có hạt nhân tùy chỉnh, tương thích với FlashAttention. Bài học này từ đầu đến cuối từ V1 đến V2, với việc triển khai đồ chơi hoạt động của thao tác khác biệt mà bạn có thể chạy trong stdlib Python.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 7 · 02 (self-attention), Giai đoạn 7 · 15 (attention biến thể), Giai đoạn 10 · 14 (hướng dẫn kiến trúc)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Nêu chính xác lý do tại sao softmax attention có sàn nhiễu và tại sao nó phát triển theo độ dài ngữ cảnh.
- Suy ra công thức vi sai attention và giải thích lý do tại sao phép trừ loại bỏ thành phần nhiễu được chia sẻ trong khi vẫn giữ được tín hiệu.
- Đi bộ giữa V1-to-V2: cái gì nhanh hơn, cái gì đơn giản hơn, cái gì ổn định hơn và tại sao mỗi thay đổi là cần thiết cho production trước khi training.
- Triển khai attention vi sai từ đầu trong các Python thuần túy và xác minh theo kinh nghiệm thuộc tính khử nhiễu trên truy vấn tín hiệu cộng nhiễu tổng hợp.

## Vấn đề

softmax attention tiêu chuẩn có một thuộc tính toán học biến thành một vấn đề đau đầu trong hoạt động trên quy mô lớn. Đối với một `q` truy vấn, trọng số attention là `softmax(qK^T / sqrt(d))`. Softmax không bao giờ có thể tạo ra các số 0 chính xác - mọi token không khớp đều có một số khối lượng dương. Khối lượng dư đó là nhiễu và nó mở rộng theo độ dài ngữ cảnh. Ở tokens 128k, ngay cả khi mỗi token không khớp chỉ nhận được 0,001% xác suất, 127.999 trong số chúng cộng lại đóng góp khoảng 12% tổng số. model phải học cách định tuyến xung quanh một sàn nhiễu phát triển theo ngữ cảnh.

Về mặt kinh nghiệm, điều này thể hiện dưới dạng nhiễu attention đầu: trích dẫn ảo giác trong RAG ngữ cảnh dài, thất bại ở giữa các nhiệm vụ truy xuất 100k token và sự suy giảm accuracy tinh tế trên đống cỏ khô benchmarks quá 32k. Bài báo Differential Transformer (arXiv: 2410.05258, ICLR 2025) đã đo lường khoảng cách: DIFF Transformers đạt perplexity thấp hơn, accuracy ngữ cảnh dài cao hơn và ít ảo giác hơn so với đường cơ sở cùng kích thước.

DIFF V1 có ba vấn đề khiến nó không nằm ngoài biên giới trước khi training pipelines. Bộ nhớ đệm giá trị của nó phải được tải hai lần cho mỗi bước giải mã, nó yêu cầu các nhân CUDA tùy chỉnh phá vỡ khả năng tương thích FlashAttention và RMSNorm trên đầu của nó làm mất ổn định training dài hạn ở quy mô 70 tỷ trở lên. DIFF V2 (Microsoft unilm blog, ngày 20 tháng 1 năm 2026) đã khắc phục cả ba. Bài học này hướng dẫn cả hai phiên bản, xây dựng toán tử khác biệt và benchmarks khử nhiễu trên một truy vấn đồ chơi.

## Khái niệm

### Sàn ồn của softmax

Đối với `q` truy vấn và khóa `K = [k_1, ..., k_N]`, trọng số attention là:

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

Không có `w_i` nào bằng không. Nếu `k_i` hoàn toàn không liên quan đến `q`, điểm số `q . k_i` không phải là 0 - nó dao động xung quanh bằng không với variance `||q||^2 / d`. Sau khi chuẩn hóa softmax, mỗi token không liên quan vẫn đóng góp `O(1/N)` vào tổng trọng số. Tổng đóng góp của tokens không liên quan là `O((N-1)/N) = O(1)` - không phải là một số lượng nhỏ.

Những gì model muốn là một cái gì đó giống như một top-k cứng: trọng lượng cao trên tokens phù hợp, trọng lượng gần như bằng không ở những nơi khác. Softmax quá mượt mà để làm điều đó trực tiếp.

### Ý tưởng khác biệt

Chia phép chiếu Q và K của mỗi đầu thành hai: Q = (Q_1, Q_2) và K = (K_1, K_2). Tính hai bản đồ attention:

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

Đầu ra:

```
DiffAttn = (A_1 - lambda * A_2) V
```

Phép trừ sẽ loại bỏ bất kỳ sự phân bố nhiễu nào mà hai bản đồ chia sẻ. Nếu cả hai bản đồ đều có trọng số gần như đồng nhất trên tokens 127k không liên quan (mà chúng sẽ khởi tạo ngẫu nhiên), thì chúng sẽ hủy bỏ. Tín hiệu - trọng lượng đỉnh trên một số tokens thực sự có liên quan - chỉ hủy bỏ nếu nó xuất hiện trong cả hai bản đồ ở cùng cường độ, điều mà nó sẽ không xuất hiện khi model tàu hoạt động.

`lambda` là một vô hướng có thể học được trên mỗi đầu, được tham số hóa là `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`. Nó có thể là âm. `lambda_init` mặc định là một số dương nhỏ như 0.8.

### Tại sao điều này phù hợp với tính năng khử nhiễu có đầu

Hãy nghĩ về hai micrô ồn ào ghi lại cùng một giọng nói. Cả hai đều thu loa cộng với nhiễu xung quanh tương quan. Trừ cái này khỏi cái kia và nhiễu được chia sẻ sẽ giảm đi. Giọng nói tồn tại vì hai tín hiệu khác nhau về pha hoặc biên độ đủ để ngăn chặn việc hủy bỏ hoàn toàn. Mỗi đầu `lambda` học chính xác sự cân bằng này.

### V1 và V2: sự khác biệt

V1 giữ số lượng parameter bằng với số Transformer cơ sở. Để có được hai truy vấn trên mỗi đầu, nó đã giảm một nửa kích thước đầu. Điều đó tốn kém khả năng biểu đạt đầu và - đau đớn hơn - giảm một nửa bộ nhớ đệm giá trị trên mỗi đầu. Giải mã phải tải bộ nhớ đệm giá trị hai lần mỗi bước (một lần mỗi softmax branch). Kết quả: giải mã chậm hơn đường cơ sở mặc dù khớp với số lượng parameter.

V2 tăng gấp đôi số lượng đầu truy vấn và giữ nguyên các tiêu đề KV (mượn parameters từ phép chiếu lên). Kích thước đầu vẫn giữ nguyên như đường cơ sở. Sau khi trừ, chiều bổ sung được chiếu xuống để khớp với phép chiếu O_W của đường cơ sở Transformer. Ba điều xảy ra cùng một lúc:

1. Tốc độ giải mã khớp với đường cơ sở (KV cache được tải một lần).
2. FlashAttention chạy không thay đổi (không có hạt nhân tuỳ chỉnh).
3. Cường độ số học khi giải mã tăng lên (nhiều điện toán hơn trên mỗi byte được tải từ HBM).

V2 cũng loại bỏ RMSNorm trên đầu mà V1 đã sử dụng để ổn định phép trừ. Ở thang đo training trước 70B-class, điều đó RMSNorm làm mất ổn định vào cuối training. V2 thay thế nó bằng một sơ đồ khởi tạo đơn giản hơn giúp training ổn định mà không cần mô-đun bổ sung.

### Khi nào cần tiếp cận nó

| Khối lượng công việc | Lợi ích |
|----------|---------|
| RAG ngữ cảnh dài (64k+) | Bản đồ attention sạch hơn, ít trích dẫn ảo giác hơn |
| Kim-in-haystack benchmarks | Mức tăng đáng kể accuracy vượt qua 32k |
| QA đa tài liệu | Ít nhiễu tài liệu chéo hơn |
| Hoàn thành mã ở 8k | Cận biên, không đáng để thay đổi kiến trúc |
| Trò chuyện ngắn (< 4k) | Về cơ bản không thể phân biệt được với đường cơ sở |

Giá trị tăng lên theo độ dài ngữ cảnh. Ở mức 4k tokens sàn nhiễu đủ nhỏ để attention tiêu chuẩn là ổn. Ở mức 128k, nó đang làm tổn thương bạn.

### Nó stacks như thế nào với các núm 2026 khác

| Feature | Tương thích với DIFF V2? |
|---------|------------------------|
| GQA | Có (V2 tăng đầu Q, không phải đầu KV) |
| MLA (Tìm kiếm sâu) | Về nguyên tắc, không có bài báo nào kết hợp chúng |
| MoE | Có (attention độc lập với khối MLP) |
| Dây thừng | Có (không thay đổi) |
| YaRN / mở rộng ngữ cảnh dài | Có (chính xác là nơi DIFF giúp ích nhiều nhất) |
| FlashChú ý | Có trong V2 (không có trong V1) |
| Giải mã suy đoán | Có (attention thay đổi không hiển thị trong vòng lặp giải mã thông số kỹ thuật) |

```figure
differential-attention
```

## Tự xây dựng

`code/main.py` thực hiện attention vi sai trong Python thuần túy. Truy vấn đồ chơi với cấu trúc tín hiệu cộng nhiễu đã biết cho phép bạn đo trực tiếp tỷ lệ khử nhiễu.

### Bước 1: softmax attention tiêu chuẩn

Hoạt động ma trận Stdlib: danh sách danh sách, matmul thủ công softmax với phép trừ độ ổn định số của tối đa.

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Bước 2: chia Q, K thành hai nửa

Kiểu V1: giảm một nửa kích thước đầu. Kiểu V2: giữ kích thước đầu và tăng gấp đôi số lượng đầu. Việc triển khai đồ chơi sử dụng V1 để rõ ràng về phương pháp sư phạm - toán học giống hệt nhau, chỉ có sổ sách kế toán khác nhau.

### Bước 3: hai softmax branches + trừ

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

Lưu ý: trọng số đầu ra có thể là âm. Điều đó không sao - bộ nhớ đệm giá trị vẫn xử lý các đóng góp đã ký. Phép chiếu V tiếp theo hấp thụ dấu hiệu.

### Bước 4: Đo khử nhiễu

Xây dựng một chuỗi tổng hợp có độ dài 1024. Đặt token tín hiệu ở một vị trí đã biết, lấp đầy rest bằng nhiễu. Tính toán (a) trọng số softmax attention tiêu chuẩn trên vị trí tín hiệu và (b) trọng số attention vi sai. Đo tỷ lệ tín hiệu trên nhiễu trong mỗi loại. DIFF attention tạo ra tỷ lệ tín hiệu trên nhiễu cao hơn một cách đáng tin cậy theo hệ số 3x-10x tùy thuộc vào mức độ khác nhau của hai branches đã được huấn luyện.

### Bước 5: Kế toán parameter V1 vs V2

Cho một config (hidden=4096, heads=32, d_head=128), in:

- Transformer cơ sở: Q, K, V mỗi kích thước `hidden * hidden`, MLP ở 4 * ẩn.
- DIFF V1: Q, K mỗi kích thước `hidden * hidden`, kích thước V `hidden * hidden` (không thay đổi), đầu mờ đi một nửa bên trong. Thêm `lambda` parameters trên mỗi đầu (O (đầu * d_head)).
- DIFF V2: Kích thước Q `2 * hidden * hidden`, kích thước K `hidden * hidden`, kích thước V `hidden * hidden`. Thêm độ mờ được chiếu xuống trước khi O_W. Thêm `lambda` parameters tương tự.

Đồ chơi đo lường chi phí parameter bổ sung cho V2 (khoảng `hidden * hidden` thêm cho mỗi khối attention) và in nó.

## Ứng dụng

DIFF V2 vẫn chưa được shipping trong mọi production inference server kể từ tháng 4 năm 2026, nhưng việc tích hợp đang được tiến hành trong vLLM và SGLang. Trong khi đó, mô hình hiển thị trong:

- production models ngữ cảnh dài nội bộ của Microsoft.
- Các bản sao nghiên cứu trong một số model training mở nhắm mục tiêu vào bối cảnh hơn 256 nghìn.
- Kiến trúc lai kết hợp attention DIFF với attention cửa sổ trượt trên các lớp thay thế.

Khi nào bạn sẽ đạt được điều này vào năm 2026:

- Training một model mới từ đầu nhắm mục tiêu vào bối cảnh hiệu quả hơn 64k. Thêm attention khác biệt ngay từ đầu; huấn luyện lại sau này rất tốn kém.
- Fine-tuning một model ngữ cảnh dài nơi các thất bại thua lỗ ở giữa thống trị đánh giá của bạn. Một LoRA trên các phép chiếu Q có thể xấp xỉ cấu trúc DIFF.

Khi bạn không:

- Bạn đang phục vụ một model được huấn luyện trước với hiệu suất bối cảnh dài ổn định. Chi phí huấn luyện lại hiếm khi hoàn trả cho trọng lượng hiện có.
- Bối cảnh của bạn luôn dưới 16k. Sàn nhiễu không đáng kể.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-diff-attention-integrator.md`. Với kiến trúc model, độ dài ngữ cảnh mục tiêu, hồ sơ ảo giác và ngân sách training, nó tạo ra một kế hoạch tích hợp để thêm attention vi sai vào một lần chạy hoặc LoRA fine-tune trước khi training mới.

## Bài tập

1. Chạy `code/main.py`. Xác minh tỷ lệ tín hiệu trên nhiễu được báo cáo cho attention vi sai cao hơn softmax attention tiêu chuẩn trên truy vấn tổng hợp. Thay đổi biên độ nhiễu và hiển thị điểm phân tần trong đó attention tiêu chuẩn không sử dụng được.

2. Tính delta parameter số đếm từ đường cơ sở đến DIFF V1 và từ đường cơ sở đến DIFF V2 cho 7B-class model (ẩn=4096, đầu=32, d_head=128, 32 lớp). Hiển thị thành phần nào đạt được parameters và thành phần nào giữ nguyên.

3. Đọc Phần 3 của bài báo DIFF V1 (arXiv: 2410.05258) và Phần 2 của blog Hugging Face DIFF V2. Trong hai câu, giải thích lý do tại sao RMSNorm V1 trên đầu người là cần thiết và tại sao V2 có thể loại bỏ nó mà không gây ra sự phân kỳ training.

4. Thực hiện cắt bỏ: tính toán attention vi sai với `lambda = 0` (softmax đầu tiên thuần túy) và `lambda = 1` (trừ toàn bộ). Trên truy vấn tổng hợp, hãy đo lường mức độ thay đổi giữa tín hiệu thành nhiễu trong quá trình quét. Xác định `lambda` tối đa hóa tín hiệu thành nhiễu.

5. Mở rộng đồ chơi đến GQA + DIFF V2. Chọn 8 đầu KV và 32 đầu Q. Cho thấy kích thước KV cache khớp với model GQA cơ sở có cùng configuration (8, 32).

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| attention vi sai | "Hai softmax trừ nhau" | Chia Q, K thành hai nửa, tính hai bản đồ softmax, trừ phần thứ hai (tỷ lệ theo lambda) từ bản đồ đầu tiên, sau đó nhân với V |
| Sàn nhiễu | "Đuôi không không của softmax" | Trọng số O (1/N) softmax đặt trên mọi token không liên quan, tổng bằng O (1) trong các ngữ cảnh dài |
| lambda | "Thang trừ" | Vô hướng có thể học được trên mỗi đầu được tham số hóa là `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`; có thể âm |
| KHÁC BIỆT V1 | "Phiên bản ICLR 2025" | Transformer vi sai ban đầu; một nửa đầu mờ để bảo toàn số lượng parameter, cần nhân tùy chỉnh, giải mã chậm hơn |
| KHÁC BIỆT V2 | "Bản sửa lỗi tháng 1 năm 2026" | Tăng gấp đôi đầu Q giữ đầu KV; phù hợp với tốc độ giải mã cơ bản và hoạt động với FlashAttention |
| RMSNorm trên đầu | "Bộ ổn định V1" | Định mức bổ sung V1 được áp dụng sau khi chênh lệch; V2 đã loại bỏ nó để ngăn chặn sự mất ổn định training muộn |
| Tỷ lệ tín hiệu trên nhiễu | "Lãng phí bao nhiêu attention" | Tỷ lệ trọng số trên vị trí tín hiệu thực với trọng số trung bình trên các vị trí không liên quan |
| Lạc giữa | "Chế độ lỗi ngữ cảnh dài" | Hiện tượng thực nghiệm trong đó việc truy xuất accuracy các tài liệu giảm xuống giữa một bối cảnh dài - DIFF attention làm giảm điều này |
| Cường độ số học | "FLOPs mỗi byte được tải" | Tỷ lệ V2 tăng khi giải mã bằng cách tăng gấp đôi truy vấn trên mỗi tải KV; quan trọng đối với giải mã ràng buộc bộ nhớ |

## Đọc thêm

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — bài báo gốc với lý thuyết khử nhiễu và cắt bỏ ngữ cảnh dài
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) - viết lại production stack, giải mã đường cơ sở phù hợp, tương thích với FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) - phân tích lý thuyết về lý do tại sao phép trừ phục hồi cấu trúc pretrained attention
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — biến thể chia sẻ parameter
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — đường cơ sở Transformer DIFF trừ đi
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — bối cảnh dài hạn benchmark các mục tiêu attention DIFF
