# Jamba - SSM-Transformer lai

> models không gian trạng thái (SSM) và transformers muốn những thứ khác nhau. Transformers mua chất lượng qua attention với chi phí bậc hai. SSM mua inference thời gian tuyến tính và bộ nhớ không đổi thông qua chất lượng định kỳ nhưng độ trễ. Jamba của AI21 (tháng 3 năm 2024) và Jamba 1.5 (tháng 8 năm 2024) đặt chúng vào cùng một model: 1 lớp Transformer cho mỗi 7 lớp Mamba, MoE trên mọi khối khác và một context window 256k phù hợp với một GPU 80GB duy nhất. Mamba-3 (ICLR 2026) thắt chặt phía SSM với các không gian trạng thái có giá trị phức tạp và phép chiếu MIMO. Bài học này đọc cả hai kiến trúc từ đầu đến cuối và giải thích lý do tại sao công thức lai đã tồn tại sau ba năm mở rộng quy mô trong khi các nỗ lực thuần SSM và tinh túy Transformer ngữ cảnh dài thì không.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, layer-mix calculator)
**Kiến thức tiên quyết:** Giai đoạn 10 · 14 (kiến trúc model mở), Giai đoạn 10 · 17 (attention thưa thớt gốc)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Giải thích ba primitives trong khối Jamba - Transformer lớp, lớp Mamba, MoE - và công thức 1: 7: thậm chí xen kẽ.
- Nêu rõ sự lặp lại của SSM trông như thế nào ở cấp độ cao và lý do tại sao nó cho phép inference bộ nhớ liên tục.
- Tính toán dấu chân KV cache của một model Jamba ở ngữ cảnh 256k và so sánh với những gì một Transformer model thuần túy sẽ cần.
- Kể tên ba cải tiến Mamba-3 (rời rạc theo cấp số nhân-hình thang, cập nhật trạng thái giá trị phức tạp, MIMO) và vấn đề mà mỗi đổi mới nhắm đến.

## Vấn đề

Attention là bậc hai về độ dài trình tự. Không gian trạng thái models là tuyến tính. Sự khác biệt đó kết hợp: ở 256k tokens, một bản đồ Transformer attention là 65 tỷ mục nhập trên mỗi đầu; trạng thái lặp lại của SSM là kích thước cố định bất kể độ dài trình tự.

models SSM thuần túy (Mamba, Mamba-2) khớp với Transformer perplexity ở quy mô nhỏ nhưng chậm trễ trong các tác vụ theo dõi trạng thái và thất bại trong một số loại truy xuất trong ngữ cảnh. Trực giác: SSM nén lịch sử thành một trạng thái cố định và khi lịch sử dài, thông tin bị rò rỉ. Attention nhớ mọi thứ chính xác nhưng phải trả chi phí bậc hai.

Cách khắc phục rõ ràng: sử dụng cả hai. Đặt Transformer lớp ở những recall chính xác quan trọng. Sử dụng các lớp SSM ở nơi khác. Điều chỉnh tỷ lệ. Jamba là model cấp production đầu tiên ship công thức lai này trên quy mô lớn (tổng cộng 52B, 12B hoạt động, 256k ngữ cảnh, 80GB đơn GPU). Jamba 1.5 mở rộng gia đình lên tổng cộng 398 tỷ / 94 tỷ hoạt động. Mamba-3 (ICLR 2026) là đường cơ sở SSM thuần túy tốt nhất hiện nay mà các giống lai có thể được xây dựng lại.

Bài học này đọc cả ba bài báo và tạo ra model tinh thần để "chọn tỷ lệ phù hợp".

## Khái niệm

### SSM trong một trang

Một không gian trạng thái model processes một chuỗi `x_1, ..., x_N` thông qua một trạng thái có kích thước cố định `h`:

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

Ở mỗi bước, trạng thái phát triển thông qua `A` động lực học tuyến tính, lấy `B x_t` đầu vào và phát ra `C h_t` đầu ra. `A, B, C` có thể học được. Lưu ý thuộc tính quan trọng: tính toán `y_t` chỉ cần `h_{t-1}` và `x_t`, không cần bất kỳ `x` nào trước đó. Bộ nhớ là hằng số. Inference là O (1) trên token.

Bí quyết để mô hình hóa chất lượng là cấu trúc của `A`. S4 (Gu 2021) đã sử dụng ma trận có cấu trúc cao có thể được đánh giá hiệu quả dưới dạng tích chập dài trong training. Mamba (Gu, Dao 2023) đã thay thế `A, B, C` cố định bằng các  phụ thuộc vào dữ liệu (phần "chọn lọc"). Mamba-2 (2024) đã đơn giản hóa hơn nữa cấu trúc. Mamba-3 (2026) bổ sung lại độ phức tạp ở những vị trí cụ thể.

Thuộc tính chính: đối với decoder LLM, lớp SSM là sự thay thế thả vào cho lớp attention, với trạng thái kích thước cố định trên mỗi lớp thay vì KV cache đang phát triển.

### Khối Jamba

Một khối Jamba xen kẽ các lớp theo hai số:

- `l`: tỷ lệ attention-Mamba. Jamba sử dụng `l = 8`, có nghĩa là 1 lớp Transformer cho mỗi 7 lớp Mamba (7 Mamba + 1 Attention = 8 lớp mỗi nhóm).
- `e`: tần số MoE. Jamba sử dụng `e = 2`, có nghĩa là mọi lớp khác đều áp dụng MoE.

Trình tự lớp trong một khối:

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

Mỗi khối Jamba có 8 lớp. Ở độ sâu 4 khối (tổng cộng 32 lớp), bạn nhận được 28 Mamba và 4 lớp Attention. 16 trong số đó sử dụng MoE.

### Tại sao lại có tỷ lệ 1:7

AI21 đã chạy cắt bỏ: tỷ lệ attention trên Mamba nào cho recall perplexity trên parameter VÀ theo ngữ cảnh tốt nhất trên các đánh giá ngữ cảnh dài của chúng?

- Quá nhiều attention (1:1): chất lượng tăng lên nhưng bộ nhớ và tốc độ giảm sút.
- Quá ít attention (1:15): bộ nhớ rất tuyệt nhưng truy xuất trong ngữ cảnh không thành công.
- Điểm ngọt: 1:7 hoặc 1:8.

Trực giác: các lớp Transformer xử lý theo dõi recall và trạng thái chính xác. Các lớp Mamba xử lý số lượng lớn xử lý rẻ tiền.

### Mã hóa vị trí

Bản thân các lớp Mamba nhận biết vị trí (thông qua sự lặp lại). Attention lớp trong các lớp lai dựa trên Mamba ban đầu không sử dụng RoPE - các lớp SSM cung cấp thông tin vị trí. Jamba 1.5 thêm RoPE vào các lớp attention để khái quát hóa ngữ cảnh dài hơn, một sự tinh chỉnh sau hoc dựa trên đánh giá ngữ cảnh dài thực nghiệm.

### Ngân sách bộ nhớ

Đối với hình dạng Jamba-1 (32 lớp: 28 Mamba + 4 Attention, ẩn 4096, 32 attention đầu):

- KV cache (chỉ attention lớp): `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB` ở 256k BF16. Chỉ có 4 lớp attention đóng góp.
- Trạng thái SSM: `28 * hidden * state_size` trên token tiền tố, nhưng đây là kích thước cố định trên mỗi lớp, không thay đổi tỷ lệ với độ dài trình tự. Trạng thái Mamba điển hình là 16 trên feature, ẩn 4096: tổng cộng `28 * 4096 * 16 * 2 = 3.7 MB`.

So với Transformer thuần túy ở 32 lớp, cùng một MHA ẩn, đầy đủ ở 32 đầu: `2 * 32 * 32 * 128 * 256k * 2 = 128 GB` ở 256k BF16. Giảm 8 lần KV cache. Ngay cả so với đường cơ sở GQA (8) hầu hết models sử dụng năm 2024 (`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`), lai 1: 7 của Jamba ở 16 GB vẫn nhỏ hơn 2 lần.

Đó là ý nghĩa của AI21 bằng cách "ngữ cảnh 256k trên một GPU 80GB duy nhất". Sự KV cache của một Transformer thuần túy MHA đầy đủ sẽ không phù hợp; ngay cả đường cơ sở GQA cũng không có chỗ cho trọng lượng và kích hoạt; Jamba có.

### Mamba-3: đường cơ sở SSM thuần túy vào năm 2026

Mamba-3 (ICLR 2026, arXiv:2603.15569) giới thiệu ba cải tiến về mặt SSM thuần túy:

1. **Rời rạc hình thang hàm mũ.** Thay thế rời rạc phương thức Euler trong Mamba-2 bằng một lặp lại biểu cảm hơn. Hoạt động giống như tích chập được áp dụng trên đầu vào trạng thái trong lặp lại lõi, thay vì như một tích chập bên ngoài trên `x_t`.

2. **Cập nhật trạng thái có giá trị phức tạp.** Mambas trước đây đã giảm ma trận trạng thái từ phức tạp (S4) sang đường chéo thực (Mamba) thành nhận dạng theo tỷ lệ (Mamba-2). Mamba-3 thêm lại các giá trị phức — tương đương với embedding quay phụ thuộc vào dữ liệu trên trạng thái. Điều này khôi phục khả năng theo dõi trạng thái mà chi phí đơn giản hóa có giá trị thực trước đó.

3. **Phép chiếu đa đầu vào đa đầu ra (MIMO).** Thay vì phép chiếu vô hướng trên mỗi feature, hãy sử dụng phép chiếu có giá trị ma trận. Cải thiện sức mạnh mô hình hóa và sử dụng phần cứng inference thời gian mà không làm tăng độ trễ giải mã.

Ở mức 1,5 tỷ parameters, Mamba-3 cải thiện accuracy hạ lưu trung bình 0,6 điểm so với Gated DeltaNet; biến thể MIMO thêm 1,2 điểm nữa để tăng tổng cộng 1,8 điểm. Ở cùng kích thước tiểu bang, Mamba-3 phù hợp với Mamba-2 với một nửa tiểu bang.

Mamba-3 vẫn chưa shipping trong một production lai trên quy mô - nhưng nó là ứng cử viên rõ ràng cho phía SSM của Jamba-class model tiếp theo.

### Khi nào nên tiếp cận một con lai

Hybrid giành chiến thắng khi:

- Bối cảnh đủ dài để Transformer KV cache thuần túy trở nên đau đớn (64k +).
- Nhiệm vụ kết hợp cấu trúc tầm ngắn (tốt cho SSM) với recall tầm xa (nhu cầu Transformer).
- Bạn muốn triển khai trên ngân sách bộ nhớ một GPU mà chỉ riêng Transformer KV cache sẽ không phù hợp.

Giống lai thua khi:

- Ngữ cảnh ngắn (dưới 16k). Chi phí SSM bị lãng phí; Transformer thuần túy là tốt.
- Các nhiệm vụ cần ở mọi nơi đến mọi nơi attention (suy luận sâu sắc, tham khảo chéo nhiều tài liệu). Sự thưa thớt của các lớp attention trong hỗn hợp gây tổn thương.
- Bạn đang mở rộng quy mô lên models biên giới nghìn tỷ parameter. Pure-Transformer + MLA + MoE (phong cách DeepSeek-V3) hiện đang giành chiến thắng trong cuộc đua năng lực.

### Bối cảnh cạnh tranh

| Model | Gia đình | Quy mô | Xác nhận quyền sở hữu duy nhất |
|-------|--------|------|-------------|
| Mamba-2 (Mamba-2) | SSM thuần túy | 3 tỷ | Thời gian tuyến tính, bộ nhớ không đổi |
| Jamba | Lai | 52B/12B | 256k trên 80GB |
| Jamba 1.5 Lớn | Lai | 398B/94B | Bối cảnh dài cấp doanh nghiệp |
| Mamba-3 (Mamba-3) | SSM thuần túy | 1,5 tỷ (giấy) | Khôi phục theo dõi trạng thái |
| Tìm kiếm sâu-V3 | Transformer tinh khiết + MoE | 671B/37B | Năng lực biên giới |

Bối cảnh năm 2026: Transformer MoE thuần túy thống trị biên giới, nhưng các sản phẩm lai sở hữu thị trường ngách hơn 256 nghìn bối cảnh. Chiến thắng theo dõi trạng thái của Mamba-3 có thể đẩy tỷ lệ lai thấp hơn (SSM nhiều hơn, ít attention hơn) trong thế hệ tiếp theo.

```figure
swiglu-ffn
```

## Ứng dụng

`code/main.py` là một máy tính bộ nhớ cho các kiến trúc lai. Với tỷ lệ SSM-Transformer và config kích thước ẩn/số lớp, nó tính toán:

- KV cache ở bối cảnh mục tiêu.
- Bộ nhớ trạng thái SSM.
- Tổng bộ nhớ ở ngữ cảnh N cho một loạt các hình dạng model.

Máy tính hỗ trợ:

- Đường cơ sở Transformer thuần túy (KV cache phát triển cùng với N).
- Kết hợp 1: 7 theo phong cách Jamba.
- Pure-SSM (không có KV cache nào cả).

Các con số được thực hiện trực tiếp từ các bài báo Jamba-1 và Jamba-1.5 cho các hình dạng đã xuất bản và ngoại suy cho các biến thể giả định.

Cân nhắc tích hợp để triển khai thực tế:

- Hầu hết các production inference servers (vLLM, SGLang) đều hỗ trợ Jamba và Mamba. Kiểm tra phiên bản cụ thể.
- Ở ngữ cảnh 256k, lợi thế bộ nhớ của Jamba hiển thị trong thông lượng yêu cầu đồng thời. Trên cùng một VRAM, bạn phù hợp với nhiều chuỗi Jamba hơn Transformer chuỗi.
- Mamba-3 như một model độc lập vẫn chưa được shipping trong production - bản xem trước nghiên cứu ở mức 1,5B.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-hybrid-picker.md`. Với thông số kỹ thuật khối lượng công việc (hồ sơ độ dài ngữ cảnh, hỗn hợp tác vụ, ngân sách bộ nhớ), nó khuyến nghị giữa Transformer thuần túy, kết hợp kiểu Jamba và SSM thuần túy, với lý do rõ ràng về sự đánh đổi bộ nhớ và chất lượng.

## Bài tập

1. Chạy `code/main.py` để tính toán KV cache ở ngữ cảnh 256k cho Transformer thuần túy 32 lớp (ẩn 4096, 32 đầu) và cho hỗn hợp Jamba-1 có cùng hình dạng. Xác minh mức giảm bộ nhớ ~8 lần mà bài báo AI21 tuyên bố.

2. Sửa đổi máy tính để model lai 1: 3 (4 Mamba: 1 Attention) và lai 1:15 (14 Mamba: 1 Attention). Biểu đồ KV cache so với tỷ lệ. Tỷ lệ KV cache bằng bộ nhớ trạng thái SSM ở tỷ lệ nào?

3. Đọc Phần 3 của bài báo Jamba (arXiv: 2403.19887). Giải thích lý do tại sao AI21 sử dụng Mamba-1 thay vì Mamba-2 mặc dù Mamba-2 nhanh hơn. Gợi ý: phần cắt bỏ lai ghi lại điều này.

4. Tính toán chi phí parameter của MoE-every-every-other-layer trong Jamba 1.5 Large (tổng cộng 398B, 94B hoạt động). So sánh tỷ lệ hoạt động với DeepSeek-V3 (37B/671B) và giải thích lý do tại sao kiến trúc của Jamba đẩy tỷ lệ hoạt động cao hơn.

5. Đọc Phần 3 của bài báo Mamba-3 (arXiv: 2603.15569). Giải thích trong ba câu tại sao cập nhật trạng thái có giá trị phức tương đương với embedding quay phụ thuộc vào dữ liệu. Gắn câu trả lời với Giai đoạn 7 · Dẫn xuất RoPE của bài 04.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| model không gian nhà nước (SSM) | "Lặp lại với trạng thái cố định" | Một lớp có `h_t = A h_{t-1} + B x_t` lặp lại đã học; bộ nhớ không đổi trên mỗi token |
| SSM chọn lọc | "Thủ thuật của Mamba" | Các parameters A, B, C phụ thuộc vào dữ liệu cho model tính chọn lọc giống như cổng tại thời gian tuyến tính |
| Tỷ lệ Attention trên Mamba | "Có bao nhiêu lớp attention" | Ở Jamba, `l = 8` có nghĩa là 1 lớp attention trên 7 lớp Mamba |
| Khối Jamba | "Nhóm 8 lớp" | Một attention + bảy Mamba + MoE ở các vị trí thay thế |
| Trạng thái SSM | "Bộ đệm ẩn" | Trạng thái kích thước cố định trên mỗi lớp thay thế cho KV cache cho các lớp Mamba |
| Bối cảnh 256k | "Số hàng đầu của Jamba" | Độ dài trình tự Jamba-1 phù hợp với một GPU 80GB duy nhất; Transformer thuần túy không thể ở kích thước đó |
| Mamba-3 (Mamba-3) | "SSM thuần túy năm 2026" | Kiến trúc SSM thuần túy tốt nhất hiện nay với trạng thái phức tạp + MIMO; các lai cơ bản xây dựng lại xung quanh |
| MIMO | "Đa đầu vào đa đầu ra" | Đổi mới Mamba-3 sử dụng phép chiếu giá trị ma trận thay vì vô hướng trên mỗi feature |
| Rời rạc hình thang theo cấp số nhân | "Mamba-3 tái phát" | Sự lặp lại biểu cảm hơn bao gồm sự rời rạc phương pháp Euler của Mamba-2 |
| Kiến trúc kết hợp | "Kết hợp attention và SSM" | Bất kỳ model nào xen kẽ các lớp Transformer và SSM; Jamba là nguyên mẫu production |

## Đọc thêm

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — bài báo gốc của Jamba, cắt bỏ tỷ lệ, yêu cầu ngữ cảnh 256k
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) - gia đình mở rộng quy mô, 398B/94B và 12B/52B các bản phát hành công khai
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) - giấy SSM chọn lọc mà Jamba xây dựng trên
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — sự kế thừa không gian trạng thái có cấu trúc được đơn giản hóa
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — trạng thái có giá trị phức tạp, MIMO, biên giới SSM thuần túy năm 2026
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) - bài báo S4, điểm khởi đầu của phả hệ SSM cho LLMs
