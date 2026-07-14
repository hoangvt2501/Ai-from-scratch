# Hướng dẫn kiến trúc DeepSeek-V3

> Giai đoạn 10 · Bài 14 đặt tên cho sáu núm kiến trúc mỗi lần mở model lượt. DeepSeek-V3 (Tháng 12 năm 2024, tổng cộng 671B parameters, 37B đang hoạt động) xoay tất cả sáu và thêm bốn núm nữa: Attention tiềm ẩn nhiều đầu, cân bằng tải không có loss phụ, Dự đoán đa Token và training DualPipe. Bài học này đọc kiến trúc của DeepSeek-V3 từ trên xuống dưới và rút ra mọi parameter đếm từ config đã xuất bản. Cuối cùng, bạn có thể giải thích lý do tại sao tỷ lệ 671B/37B là lựa chọn phù hợp và tại sao MLA + MoE cùng nhau đánh bại một trong hai ở biên giới.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, parameter calculator)
**Kiến thức tiên quyết:** Giai đoạn 10 · 14 (hướng dẫn model mở), Giai đoạn 10 · 17 (NSA), Giai đoạn 10 · 18 (MTP), Giai đoạn 10 · 19 (Ống kép)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Đọc DeepSeek-V3 config từ trên xuống dưới và giải thích từng trường về sáu núm GPT-2 cộng với bốn bổ sung dành riêng cho DeepSeek.
- Lấy tổng số parameter (671B), số lượng parameter hoạt động (37B) và các thành phần đóng góp vào mỗi thành phần.
- Tính toán dấu chân KV cache của MLA ở ngữ cảnh 128k và so sánh với những gì một model mật độ tham số hoạt động giống nhau với GQA sẽ phải trả.
- Nêu bốn cải tiến dành riêng cho DeepSeek (MLA, MTP, định tuyến không loss phụ trợ, DualPipe) và nêu tên phần nào của architecture/training stack mỗi đổi mới nhắm mục tiêu.

## Vấn đề

DeepSeek-V3 là model mở biên giới đầu tiên có kiến trúc khác biệt đáng kể so với gia đình Llama. Llama 3 405B là "GPT-2 với sáu núm xoay". DeepSeek-V3 GPT-2 với tất cả sáu núm cộng với bốn núm nữa. Đọc Llama 3 config là một khởi động để đọc config DeepSeek, nhưng cấu trúc sâu - hình dạng của khối attention, logic định tuyến, mục tiêu training thời gian - đủ khác để bạn cần một hướng dẫn riêng.

Phần thưởng của việc học nó: Bản phát hành trọng lượng mở của DeepSeek-V3 đã thay đổi ý nghĩa của "khả năng biên giới" trong models mở. Kiến trúc là bản thiết kế mà nhiều lần chạy training năm 2026 đang sao chép. Hiểu được nó là tiền đặt cược cho bất kỳ vai trò nào liên quan đến LLM training hoặc inference biên giới.

## Khái niệm

### Lõi bất biến, một lần nữa

DeepSeek-V3 vẫn là tự hồi quy. Nó vẫn stacks decoder khối. Mỗi khối vẫn có attention cộng với MLP cộng với hai RMSNorms. Nó vẫn sử dụng SwiGLU trong MLP. Nó vẫn sử dụng RoPE. Định mức trước. embeddings ràng buộc trọng lượng. Đường cơ sở giống như mọi Llama hoặc Mistral.

### Bước ngoặt: MLA thay vì GQA

Từ Giai đoạn 10 · 14 bạn biết GQA thu nhỏ KV cache bằng cách chia sẻ K và V trên các nhóm đầu Q. Attention tiềm ẩn nhiều đầu (MLA) đi xa hơn: K và V được nén thành một biểu diễn tiềm ẩn cấp thấp được chia sẻ (`kv_lora_rank`), sau đó được giải nén trên mỗi đầu một cách nhanh chóng. KV cache chỉ lưu trữ tiềm ẩn - thường là 512 phao trên mỗi token mỗi lớp, không phải 8 x 128 = 1024 phao.

Ở ngữ cảnh 128k, DeepSeek-V3 với MLA (một `c^{KV}` tiềm ẩn được chia sẻ trên mỗi token trên mỗi lớp; K và V đều có nguồn gốc từ tiềm ẩn này thông qua các hình chiếu lên có thể được hấp thụ vào matmul tiếp theo):

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

Đường cơ sở GQA giả định (Llama 3 hình dạng 70B, đầu 8 KV, đầu mờ 128) sẽ trả:

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA nhỏ hơn 4 lần so với bộ nhớ đệm GQA kiểu Llama-3-70B ở ngữ cảnh 128k.

Sự đánh đổi: MLA thêm một bước giải nén cho mỗi attention tính toán (trên mỗi đầu). Điện toán bổ sung nhỏ so với băng thông được tiết kiệm. Chiến thắng ròng cho inference ngữ cảnh dài.

### Định tuyến: cân bằng tải không loss phụ trợ

MoE bộ định tuyến quyết định chuyên gia top-k nào process mỗi token. Một bộ định tuyến ngây thơ tập trung quá nhiều công việc vào một số chuyên gia, khiến những người khác nhàn rỗi. Sửa lỗi tiêu chuẩn: thêm một thuật ngữ loss phụ trợ để phạt sự mất cân bằng tải. Điều này hoạt động nhưng làm giảm hiệu suất tác vụ chính một chút.

DeepSeek-V3 giới thiệu một sơ đồ không có loss phụ trợ. Mỗi chuyên gia bias thuật ngữ được thêm vào logits bộ định tuyến, được điều chỉnh trong quá trình training theo một quy tắc đơn giản: nếu `e` chuyên gia bị quá tải, hãy giảm `bias_e`; nếu quá tải, hãy tăng nó. Không có thêm thời hạn loss. Training luôn sạch sẽ. Tải chuyên gia luôn cân bằng.

Ảnh hưởng đến loss chính: không thể đo lường được. Ảnh hưởng đến kiến trúc MoE: sạch hơn, không có loss hyperparameter phụ để điều chỉnh.

### MTP: training dày đặc hơn + bản nháp miễn phí

Từ Giai đoạn 10 · 18 bạn biết DeepSeek-V3 thêm mô-đun D = 1 MTP dự đoán token hai vị trí phía trước. Ở inference, mô-đun được huấn luyện được tái sử dụng như một bản nháp giải mã suy đoán với 80% + chấp nhận. Ở training, mỗi trạng thái ẩn được giám sát trên D + 1 = 2 mục tiêu, cung cấp tín hiệu dày đặc hơn.

Parameters: 14B trên đỉnh của 671B chính. Chi phí: 2,1%.

### Các training: DualPipe

Từ Giai đoạn 10 · 19 bạn biết đấy, DualPipe là một pipeline hai chiều chồng chéo các khối tiến và lùi với các nút chéo liên lạc tất cả với tất cả. Ở thang đo 3,048-H800 của DeepSeek-V800, nó phục hồi khoảng 245k GPU giờ mà 1F1B sẽ mất bởi bong bóng pipeline.

### config, từng trường

Đây là config DeepSeek-V3 (đơn giản hóa):

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

Phân tích cú pháp nó:

- `hidden_size=7168`: Kích thước embedding.
- `num_hidden_layers=61`: tổng độ sâu khối.
- `first_k_dense_layers=3`: 3 khối đầu tiên sử dụng MLP dày đặc có kích thước 18432. 58 khối còn lại sử dụng MoE.
- `num_attention_heads=128`: 128 tiêu đề truy vấn.
- `kv_lora_rank=512`: K và V được nén đến kích thước tiềm ẩn này và được giải nén trên mỗi đầu.
- `num_experts=256, num_experts_per_tok=8`: mỗi khối MoE có 256 chuyên gia, tuyến đường top-8.
- `shared_experts=1`: Ngoài 256 chuyên gia được định tuyến, 1 chuyên gia luôn hoạt động đóng góp vào mọi token. Hãy coi nó như một "sàn dày đặc" đảm bảo mọi token đều có được thứ gì đó đáng tin cậy.
- `moe_intermediate_size=2048`: kích thước ẩn MLP của mỗi chuyên gia. Nhỏ hơn MLP dày đặc vì có 256 trong số chúng.

### Kế toán Parameter

Tính toán đầy đủ nằm trong `code/main.py`. Tiêu đề:

- Embedding: `vocab * hidden = 129280 * 7168 = ~0.93B`.
- 3 khối dày đặc đầu tiên: attention với MLA (~144M mỗi khối) + MLP dày đặc (~260M mỗi khối) + định mức. Tổng cộng khoảng 1,2 tỷ.
- 58 khối MoE: attention với MLA (~144 triệu) + 256 chuyên gia mỗi khối (30 triệu khối) + 1 chuyên gia chia sẻ (30 triệu) + định mức. Tổng cộng ~7,95 tỷ mỗi khối, bao gồm tất cả các chuyên gia. Tổng cộng 461 tỷ cho 58 khối MoE.
- Mô-đun MTP: 14B.

Tổng cộng: ~476B cho kiến trúc cốt lõi + 14B MTP + rõ ràng là số 671B được công bố chiếm parameters cấu trúc bổ sung (bias tensors, các thành phần dành riêng cho chuyên gia, chia sẻ quy mô chuyên gia, v.v.). Con số chúng tôi tái tạo trong máy tính nằm trong khoảng 3-5% so với số lượng được công bố - delta đến từ các tài liệu báo cáo kế toán chi tiết của DeepSeek trong phụ lục Phần 2 của nó.

Số parameters hoạt động trên mỗi chuyển tiếp:

- Attention: 144M mỗi lớp * 61 = 8.8B (tất cả các lớp cháy).
- MLP hoạt động: 3 lớp đầu tiên dày đặc (3 * 260M = 780M), 58 MoE lớp mỗi lớp hoạt động với 8 lớp định tuyến + 1 chia sẻ + định tuyến trên cao. MLP hoạt động trên mỗi lớp: ~260M. Tổng: 3 * 260M + 58 * 260M = ~15.9B.
- Embedding + định mức: 1,2B.
- Tổng số hoạt động: khoảng 26B lõi + 14B MTP (được huấn luyện nhưng không phải lúc nào cũng chạy ở inference) ≈ 37B.

### Tỷ lệ 671B / 37B

Tỷ lệ thưa thớt 18x (tham số hoạt động là 5,5% tổng số). DeepSeek-V3 là MoE model biên giới thưa thớt nhất có trọng số mở shipped. Mixtral 8x7B với tỷ lệ 13/47 (28%) dày đặc hơn nhiều. Llama 4 Maverick ở tỷ lệ 17B/400B (4,25%) là tương đương. Đặt cược DeepSeek: ở quy mô biên giới, nhiều chuyên gia có tỷ lệ kích hoạt thấp hơn tạo ra chất lượng tốt hơn trên mỗi FLOP hoạt động.

### Vị trí của DeepSeek-V3

| Model | Tổng cộng | Hoạt động | Tỷ lệ | Attention | Ý tưởng mới lạ |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70 tỷ | 70 tỷ | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400 tỷ | 17 tỷ | 4.25% | GQA | — |
| Hỗn hợp 8x22B | 141 tỷ | 39 tỷ | 27% | GQA | — |
| Tìm sâu V3 | 671 tỷ | 37 tỷ | 5.5% | MLA 512 | MLA + MTP + không chứa aux + DualPipe |
| Qwen 2.5 72B | 72 tỷ | 72 tỷ | 100% | GQA 64/8 | Tiện ích mở rộng YaRN |

### Tiếp theo: R1, V4

DeepSeek-R1 (2025) là một training lý luận chạy trên đường trục V3. R1 sử dụng cùng một kiến trúc. Điều đã thay đổi là công thức sau training (RL quy mô lớn về các tác vụ có thể xác minh), không phải kiến trúc pretraining.

DeepSeek-V4 (nếu ships) dự kiến sẽ giữ MLA + MoE + MTP và thêm DSA (DeepSeek Sparse Attention), người kế nhiệm NSA từ Giai đoạn 10 · 17. Dòng dõi ổn định: những đổi mới ở cấp độ kiến trúc tích lũy; mỗi phiên bản xoay các núm bổ sung.

```figure
moe-routing
```

## Ứng dụng

`code/main.py` là máy tính parameter chuyên về hình dạng của DeepSeek-V3. Chạy nó, so sánh đầu ra của nó với các con số của bài báo và sử dụng nó trên các biến thể giả định (256 chuyên gia so với 512, top 8 so với top-16, xếp hạng MLA 512 so với 1024).

Những gì cần xem:

- Tổng số parameter so với 671 tỷ được công bố.
- Số lượng parameter đang hoạt động so với 37 tỷ được công bố.
- KV cache ở ngữ cảnh 128k - so sánh MLA và GQA.
- Phân tích mỗi lớp để xem ngân sách parameter thực sự đi đến đâu.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-deepseek-v3-reader.md`. Với một model họ DeepSeek (V3, R1 hoặc bất kỳ biến thể nào trong tương lai), nó tạo ra một cách đọc kiến trúc từng thành phần đặt tên cho từng trường của config, lấy parameter đếm theo thành phần và xác định model sử dụng cái nào trong số bốn cải tiến cụ thể của DeepSeek.

## Bài tập

1. Chạy `code/main.py`. So sánh ước tính tổng parameter của máy tính với 671B đã xuất bản và xác định delta đến từ đâu. Phần 2 của bài báo có liệt kê đầy đủ.

2. Sửa đổi config để sử dụng MLA xếp hạng 256 thay vì 512. Tính toán kích thước KV cache kết quả ở ngữ cảnh 128k. Nó giảm bao nhiêu phần trăm và chi phí biểu cảm trên mỗi đầu là bao nhiêu?

3. So sánh định tuyến của DeepSeek-V3 (256 chuyên gia, top 8) với biến thể giả định (512 chuyên gia, top 8). Tổng số parameters tăng trưởng; hoạt động parameters giữ nguyên. Năng lực chuyên gia bổ sung mua được gì về mặt lý thuyết và chi phí inference là bao nhiêu?

4. Đọc Phần 2.1 của báo cáo kỹ thuật DeepSeek-V3 (arXiv: 2412.19437) về MLA. Giải thích trong ba câu tại sao ma trận giải nén K và V có thể được "hấp thụ" vào matmul tiếp theo để đạt hiệu quả inference thời gian.

5. DeepSeek-V3 sử dụng training FP8 cho hầu hết các hoạt động. Tính toán mức tiết kiệm bộ nhớ của FP8 so với BF16 để lưu trữ trọng lượng 671B. Điều này giao nhau như thế nào với ngân sách 14.8T-token training?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| MLA | "Attention tiềm ẩn nhiều đầu" | Nén K và V thành một tiềm ẩn cấp thấp được chia sẻ (kv_lora_rank, thường là 512), giải nén cho mỗi đầu một cách nhanh chóng; KV cache chỉ lưu trữ tiềm ẩn |
| kv_lora_rank | "Nén MLA mờ" | Kích thước của tiềm ẩn được chia sẻ cho K và V; DeepSeek-V3 sử dụng 512 |
| K lớp dày đặc đầu tiên | "Các lớp ban đầu vẫn dày đặc" | Một vài lớp model MoE đầu tiên bỏ qua bộ định tuyến MoE và chạy MLP dày đặc để ổn định |
| num_experts_per_tok | "Định tuyến Top-k" | Có bao nhiêu chuyên gia được định tuyến bắn mỗi token; DeepSeek-V3 sử dụng 8 |
| Chuyên gia chia sẻ | "Chuyên gia luôn hoạt động" | Các chuyên gia process mọi token bất kể định tuyến; DeepSeek-V3 sử dụng 1 |
| Định tuyến không loss phụ trợ | "Cân bằng tải được điều chỉnh Bias" | Mỗi chuyên gia bias thuật ngữ được điều chỉnh trong training để giữ cho chuyên gia cân bằng tải mà không cần thêm thuật ngữ loss |
| Mô-đun MTP | "Đầu dự đoán bổ sung" | Transformer khối dự đoán t+2 từ h^(1) và E(t+1); training dày đặc hơn, bản nháp giải mã suy đoán tự do |
| Ống kép | "pipeline hai chiều" | Training lịch trình trùng lặp forward/backward điện toán với tất cả các nút chéo |
| Tỷ lệ parameter hoạt động | "Thưa thớt" | active_params / total_params; DeepSeek-V3 đạt 5.5% |
| FP8 training | "training 8 bit" | Training lưu trữ và nhiều hoạt động tính toán trong FP8; giảm khoảng một nửa bộ nhớ so với BF16 với chi phí chất lượng nhỏ |

## Đọc thêm

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — tài liệu kiến trúc, training và kết quả đầy đủ
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — config tệp và ghi chú triển khai
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) - người tiền nhiệm giới thiệu MLA
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — người kế thừa training lý luận trên kiến trúc của V3
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) - hướng đi tương lai cho attention gia đình DeepSeek
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — tham chiếu lịch trình training
