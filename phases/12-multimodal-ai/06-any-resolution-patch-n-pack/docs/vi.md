# Tầm nhìn bất kỳ độ phân giải nào: Patch-n'-Pack và NaFlex

> Hình ảnh thực không phải là hình vuông 224x224. Biên lai là 9:16, biểu đồ là 16:9, quét y tế có thể là 4096x4096, ảnh chụp màn hình di động là 9:19.5. Câu trả lời VLM trước năm 2024 - thay đổi kích thước mọi thứ thành một hình vuông cố định - đã loại bỏ tín hiệu giúp OCR, hiểu tài liệu và phân tích cú pháp cảnh có độ phân giải cao hoạt động. NaViT (Google, 2023) cho thấy bạn có thể đóng gói các bản vá có độ phân giải thay đổi vào một transformer batch duy nhất với mặt nạ đường chéo khối. M-RoPE (2024) của Qwen2-VL đã loại bỏ hoàn toàn các bảng vị trí tuyệt đối. AnyRes của LLaVA-NeXT đã xếp hình ảnh có độ phân giải cao thành hình ảnh cơ sở + phụ. Biến thể NaFlex của SigLIP 2 (2025) hiện là encoder mặc định cho VLMs mở muốn có một checkpoint duy nhất để phục vụ mọi tỷ lệ khung hình. Bài học này thực hiện patch-n'-pack từ đầu đến cuối.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, patch packer + block-diagonal mask)
**Kiến thức tiên quyết:** Giai đoạn 12 · 01 (bản vá ViT), Giai đoạn 12 · 05 (LLaVA)
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Đóng gói các bản vá từ một batch hình ảnh có độ phân giải thay đổi thành một trình tự và xây dựng mặt nạ attention theo đường chéo khối.
- Chọn giữa lát gạch AnyRes (LLaVA-NeXT), NaFlex (SigLIP 2) và M-RoPE (Qwen2-VL) cho một tác vụ nhất định.
- Tính toán ngân sách token cho OCR, biểu đồ và nhiếp ảnh mà không cần thay đổi kích thước.
- Kể tên ba chế độ thất bại của thay đổi kích thước vuông: văn bản bị bóp, nội dung bị cắt, lãng phí tokens trên đệm.

## Vấn đề

Transformers mong đợi một trình tự. Một batch là một stack các chuỗi có cùng độ dài. Nếu hình ảnh của bạn có kích thước 224x224, bạn sẽ nhận được 196 bản vá tokens mỗi lần, không cần đệm, công việc đã hoàn thành. Đào trên 224, suy ra trên 224, không bao giờ nghĩ về độ phân giải nữa.

Thế giới không hợp tác. Tài liệu là dọc (8,5x11 inch, 2:3-ish). Ảnh chụp màn hình biểu đồ là ngang (16:9). Hóa đơn cao và mỏng (1:3). Hình ảnh y tế ships ở 2048x2048 hoặc lớn hơn. Ảnh chụp màn hình thiết bị di động là 1170x2532 (0,46:1).

Ba lựa chọn trước năm 2024 và lý do tại sao mỗi lựa chọn không thành công:

1. Thay đổi kích thước thành hình vuông cố định (224x224 hoặc 336x336). Squish bóp méo văn bản và khuôn mặt. Việc giảm tỷ lệ sẽ phá hủy nhãn biểu đồ và nội dung OCR. Thực hành tiêu chuẩn cho đến LLaVA-1.5.
2. Cắt theo tỷ lệ khung hình cố định. Bạn vứt bỏ hầu hết hình ảnh và chọn vị trí cắt là vấn đề về thị lực của chính nó.
3. Đệm sang cạnh dài nhất. Sửa biến dạng nhưng lãng phí 50% + tokens cho phần đệm cho ảnh chân dung. Chi phí attention bậc hai trên tất cả các tokens đệm đó.

Câu trả lời 2024-2025: hãy để transformer ăn các bản vá ở độ phân giải gốc của hình ảnh và tìm ra cách đóng gói một batch không đồng nhất vào một chuỗi mà không lãng phí điện toán.

## Khái niệm

### NaViT và patch-n'-pack

NaViT (Dehghani et al., 2023) là bài báo cho thấy công việc này trên quy mô lớn. Ý tưởng là máy móc:

1. Đối với mỗi hình ảnh trong batch, hãy tính toán lưới bản vá gốc của nó ở kích thước bản vá đã chọn (giả sử 14).
2. Làm phẳng các bản vá của mỗi hình ảnh thành chuỗi độ dài thay đổi của riêng nó.
3. Nối tất cả các bản vá của hình ảnh thành một chuỗi dài cho các batch.
4. Xây dựng mặt nạ attention theo đường chéo khối để các bản vá của hình ảnh A chỉ xuất hiện trong hình ảnh A.
5. Mang thông tin vị trí trên mỗi bản vá (RoPE 2D hoặc vị trí phân số embeddings).

Một batch của ba hình ảnh ở 336x336 (576 tokens), 224x224 (256 tokens) và 448x336 (768 tokens) trở thành một chuỗi 1600 token với mặt nạ đường chéo khối 1600x1600. Không có đệm. Không lãng phí điện toán. transformer xử lý tỷ lệ khung hình tùy ý.

NaViT cũng giới thiệu tính năng thả bản vá phân đoạn trong training - thả ngẫu nhiên 50% bản vá trên batch - vừa đều điều chỉnh vừa tăng tốc độ training. SigLIP 2 kế thừa điều này.

### AnyRes (LLaVA-NeXT)

AnyRes của LLaVA-NeXT là giải pháp thay thế thực tế. Với hình ảnh có độ phân giải cao và encoder cố định (CLIP hoặc SigLIP ở 336), hãy xếp hình ảnh:

1. Chọn bố cục lưới từ một tập hợp được xác định trước — (1x1), (1x2), (2x1), (1x3), (3x1), (2x2), v.v. — phù hợp nhất với tỷ lệ khung hình của hình ảnh.
2. Xếp hình ảnh đầy đủ vào lưới; Mỗi ô trở thành một cây trồng 336x336.
3. Đồng thời tạo hình thu nhỏ: toàn bộ hình ảnh được thay đổi kích thước thành 336x336 dưới dạng token ngữ cảnh toàn cầu.
4. Mã hóa mọi ô thông qua 336-encoder bị đóng băng. Nối tokens gạch + tokens hình thu nhỏ.

Đối với hình ảnh 672x672 ở lưới 2x2 cộng với hình thu nhỏ: 4 * 576 + 576 = 2880 tokens hình ảnh. Tốn kém nhưng hiệu quả - LLM nhìn thấy cả chi tiết địa phương và bối cảnh toàn cầu.

AnyRes là tuyến đường được lựa chọn khi encoder của bạn bị đóng băng và chỉ hỗ trợ một độ phân giải. Nó bùng nổ số lượng token cho các hình ảnh lớn (hình ảnh 1344x1344 ở lưới 4x4 là 9216 + 576 ≈ 9800 tokens, lấp đầy hầu hết ngữ cảnh 8k LLM).

### M-RoPE (Qwen2-VL)

Qwen2-VL giới thiệu Embedding vị trí quay đa phương thức. Thay vì các vị trí phân số của NaViT hoặc ô và hình thu nhỏ của AnyRes, mỗi bản vá mang một vị trí 3D (thời gian, chiều cao, chiều rộng). Các vòng quay query/key xử lý H, W và chiều dài thời gian tùy ý.

M-RoPE ships độ phân giải động gốc mà không cần huấn luyện lại. Tại inference bạn cung cấp bất kỳ hình ảnh HxW nào, bộ nhúng bản vá tạo ra H/14 x W/14 tokens, mỗi token có vị trí (t = 0, r = row, c = col), RoPE xoay attention với tần số phù hợp, hoàn tất. Qwen2.5-VL và Qwen3-VL tiếp tục điều này. V3PE của InternVL2 cũng là ý tưởng tương tự với mã hóa biến đổi cho mỗi phương thức.

Không giống như AnyRes, M-RoPE là O (H x W / P ^ 2) tokens ở độ phân giải gốc - không có chi phí ô nhân. Không giống như NaViT, nó vẫn mong đợi một hình ảnh duy nhất cho mỗi chuyển tiếp. Việc phân lô trên các độ phân giải vẫn cần vá và đóng gói ở trên cùng.

### NaFlex (SigLIP 2)

NaFlex là chế độ uốn cong gốc của SigLIP 2 checkpoint. Một model duy nhất phục vụ nhiều độ dài trình tự (256, 729, 1024 tokens) ở inference. Bên trong, nó sử dụng patch-n'-pack kiểu NaViT trong các vị trí phân số training và tuyệt đối trên mỗi bản vá. Điểm bán hàng: một checkpoint, chọn ngân sách token của bạn ở mức inference dựa trên nhiệm vụ.

Đối với một nhiệm vụ ngữ nghĩa (phân loại, truy xuất), 256 tokens. Để hiểu OCR hoặc biểu đồ, 1024 tokens. Không huấn luyện lại.

### Mặt nạ đóng gói

Mặt nạ đường chéo khối là nơi hầu hết các triển khai vấp ngã. Đối với một chuỗi độ dài đóng gói `N_total` bao phủ hình ảnh `i=0..B-1` có độ dài `n_i`, `M` mặt nạ có hình dạng `(N_total, N_total)` là 1 nếu cả hai chỉ số nằm trong cùng một khối của hình ảnh, nếu không thì 0. Bạn có thể tạo nó từ danh sách độ dài tích lũy:

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

Đây là một dòng PyTorch với `torch.block_diag` hoặc một tập hợp rõ ràng. Đường dẫn có độ dài thay đổi (`cu_seqlens`) của FlashAttention bỏ qua hoàn toàn mặt nạ và tham dự trong các trình tự bằng cách sử dụng tensor độ dài tích lũy trực tiếp - nhanh hơn ~10 lần so với mặt nạ dày đặc cho batches thông thường.

### Token ngân sách

Chọn chiến lược của bạn theo nhiệm vụ:

- OCR / tài liệu: 1024-4096 tokens. Hình thu nhỏ SigLIP 2 NaFlex ở 1024 hoặc AnyRes 3x3 +.
- Biểu đồ và giao diện người dùng: 729-1024 tokens ở 384-448 gốc. Độ phân giải động Qwen2.5-VL với giới hạn pixel tối đa.
- Ảnh tự nhiên: 256-576 tokens là được. Hạ lưu LLM thấy đủ. Trả tiền cho tokens có mật độ nội dung cao.
- Video: 64-128 tokens mỗi khung hình sau khi gộp không gian, 2-8 FPS. Bài 12.17 bao gồm điều này.

Quy tắc production năm 2026: chọn giới hạn pixel tối đa cho mỗi tác vụ, mã hóa ở tỷ lệ khung hình gốc lên đến giới hạn đó, đóng gói batch và bỏ qua phần đệm. Qwen2.5-VL hiển thị `min_pixels` và `max_pixels` cho chính xác núm này.

## Ứng dụng

`code/main.py` triển khai patch-n'-pack cho một batch hình ảnh không đồng nhất với tọa độ pixel số nguyên. Nó:

- Lấy danh sách các kích thước hình ảnh (H, W).
- Tính toán độ dài chuỗi bản vá của mỗi hình ảnh ở kích thước bản vá 14.
- Đóng gói chúng thành một chuỗi tổng chiều dài `sum(n_i)`.
- Xây dựng mặt nạ attention đường chéo khối (dày đặc, để rõ ràng).
- So sánh chi phí đóng gói so với thay đổi kích thước vuông và xếp gạch AnyRes.
- In bảng ngân sách token cho một batch hỗn hợp (biên lai, biểu đồ, ảnh chụp màn hình, ảnh).

Chạy nó. Những con số bỏ ra là lý do mỗi năm 2026 mở VLM sử dụng patch-n'-pack.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-resolution-budget-planner.md`. Với khối lượng công việc tỷ lệ khung hình hỗn hợp (OCR, biểu đồ, ảnh, khung video) và tổng ngân sách token, nó chọn chiến lược phù hợp (NaFlex, AnyRes, M-RoPE hoặc hình vuông cố định) và phát ra configuration cho mỗi yêu cầu. Sử dụng skill này khi bạn định cỡ VLM cho một sản phẩm - nó ngăn chặn sự bùng nổ token 10 lần im lặng giết chết ngân sách độ trễ.

## Bài tập

1. Biên lai là 600x1500 (1:2.5). Ở kích thước bản vá 14, có bao nhiêu tokens độ phân giải gốc? Có bao nhiêu sau khi thay đổi kích thước hình vuông thành 336? Cái nào mất nhiều OCR accuracy hơn trong thực tế?

2. Xây dựng mặt nạ đường chéo khối cho batch bốn hình ảnh có độ dài 256, 576, 729, 1024. Xác minh ma trận attention là 2585x2585 và có chính xác `256^2 + 576^2 + 729^2 + 1024^2` mục nhập không phải không.

3. Đối với hình ảnh 1792x896 ở bản vá 14, hãy so sánh: (a) thay đổi kích thước vuông thành 336 rồi mã hóa, (b) AnyRes 2x1 + hình thu nhỏ, (c) M-RoPE ở gốc. Cái nào sử dụng ít tokens nhất? Cái nào giữ được nhiều chi tiết nhất?

4. Thực hiện thả bản vá phân đoạn: đưa ra một trình tự đóng gói, thả ngẫu nhiên 50% tokens đồng đều và cập nhật mặt nạ đường chéo khối cho phù hợp. Đo sự thay đổi thưa thớt của mặt nạ.

5. Đọc Phần 3.2 của bài báo Qwen2-VL (arXiv: 2409.12191). Mô tả trong hai câu những gì `min_pixels` và `max_pixels` kiểm soát và tại sao cả hai giới hạn đều quan trọng.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Patch-n'-pack | "Đóng gói kiểu NaViT" | Nối chuỗi bản vá có độ dài thay đổi từ các hình ảnh khác nhau thành một chiều batch |
| Mặt nạ đường chéo khối | "Mặt nạ đóng gói" | Attention mặt nạ giới hạn các bản vá của mỗi hình ảnh để chỉ chú ý đến chính họ, không phải những người hàng xóm trong nhóm |
| AnyRes | "Lát gạch LLaVA-NeXT" | Chia hình ảnh có độ phân giải cao thành lưới các ô có kích thước cố định cộng với hình thu nhỏ toàn cục; Mã hóa mọi ô bằng một encoder cố định |
| NaFlex | "SigLIP 2 gốc linh hoạt" | checkpoint SigLIP 2 đơn phục vụ ngân sách 256/729/1024-token inference mà không cần huấn luyện lại |
| M-RoPE | "Dây thừng đa phương thức" | Mã hóa vị trí quay 3D (thời gian, hàng, cột) xử lý H, W, T tùy ý mà không cần bảng vị trí |
| cu_seqlens | "Đóng gói FlashAttention" | Độ dài tích lũy tensor đường dẫn varlen FlashAttention sử dụng thay vì mặt nạ đường chéo khối dày đặc |
| min_pixels / max_pixels | "Giới hạn độ phân giải" | Các nút Qwen2.5-VL cho mỗi yêu cầu giới hạn token đếm trên các đầu vào rất nhỏ hoặc rất lớn |
| Ngân sách token trực quan | "Có bao nhiêu tokens cho mỗi hình ảnh" | Số lượng bản vá tokens phát ra trên mỗi hình ảnh; Đặt ngân sách prompt của LLM và chi phí attention |

## Đọc thêm

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
