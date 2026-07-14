# Chim hồng hạc và Cross-Attention có cổng cho Few-Shot VLMs

> Flamingo (2022) của DeepMind đã làm hai điều trước bất kỳ ai khác. Nó cho thấy một model duy nhất có thể process các chuỗi hình ảnh, video và văn bản xen kẽ tùy ý. Và nó cho thấy VLMs có thể học theo ngữ cảnh - đưa ra một few-shot prompt với ba cặp ví dụ (hình ảnh, chú thích) và chú thích model một hình ảnh mới mà không cần bất kỳ bước gradient nào. Cơ chế: kiểm soát cross-attention lớp, được chèn vào giữa các lớp hiện có của LLM đóng băng, với cổng tanh đã học bắt đầu từ số không để khả năng văn bản của LLM được giữ nguyên khi khởi tạo. Bài học này hướng dẫn bộ lấy mẫu lại Perceiver của Flamingo và kiến trúc cross-attention được kiểm soát - tổ tiên của các đầu vào xen kẽ của Gemini và tokens hình ảnh của Idefics2.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Kiến thức tiên quyết:** Giai đoạn 12 · 03 (BLIP-2 Q-Former)
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Giải thích cách gated cross-attention bảo toàn khả năng văn bản của LLM bị đóng băng khi khởi tạo thông qua tanh(gate) = 0.
- Đi qua bộ lấy mẫu lại Perceiver: N bản vá hình ảnh → K sửa các truy vấn "tiềm ẩn" thông qua cross-attention.
- Mô tả cách Flamingo xử lý các chuỗi văn bản hình ảnh xen kẽ với mặt nạ nhân quả tôn trọng vị trí hình ảnh.
- Tái tạo cấu trúc prompt đa phương thức few-shot (3 ví dụ chú thích hình ảnh sau đó là hình ảnh truy vấn).

## Vấn đề

BLIP-2 nạp 32 tokens hình ảnh vào lớp đầu vào của LLM bị đóng băng. Hoạt động cho một hình ảnh mỗi prompt. Nhưng điều gì sẽ xảy ra nếu bạn muốn cung cấp *nhiều* hình ảnh xen kẽ với văn bản, như trong "đây là hình ảnh A, chú thích nó; đây là hình ảnh B, chú thích nó; bây giờ đây là hình ảnh C, chú thích nó"? self-attention của LLM sẽ cần xử lý tokens hình ảnh và tokens văn bản trong một luồng duy nhất và câu hỏi vị trí nào có thể tham gia vào hình ảnh nào trở nên cầu kỳ.

Câu trả lời của Flamingo: hoàn toàn không thay đổi luồng đầu vào của LLM. Chèn thêm các lớp cross-attention giữa các khối LLM hiện có. Văn bản vẫn tokens chảy qua self-attention nhân quả của LLM như mọi khi. Giữa mỗi vài khối LLM, văn bản cũng tokens tham gia chéo vào features hình ảnh thông qua một lớp được kiểm soát mới. Cổng (được khởi tạo về không) có nghĩa là ở bước không, các lớp mới không hoạt động - model hoạt động giống hệt như pretrained LLM. Khi training tiến triển, cổng mở ra và thông tin trực quan bắt đầu chảy.

Câu hỏi thứ hai mà Flamingo trả lời: làm thế nào để bạn xử lý một số lượng hình ảnh thay đổi (0, 1 hoặc nhiều) trên mỗi prompt? Bộ lấy mẫu lại Perceiver - một mô-đun cross-attention nhỏ lấy bất kỳ số lượng bản vá nào bạn có và tạo ra một số lượng tokens tiềm ẩn trực quan cố định. Lớp LLM cross-attention nhìn thấy cùng một hình dạng bất kể có bao nhiêu hình ảnh trong prompt.

## Khái niệm

### Các LLM đóng băng

Flamingo bắt đầu với một LLM Chinchilla 70B đông lạnh. Tất cả các trọng lượng 70B không bị ảnh hưởng. Văn bản self-attention và FFN hiện có hoạt động bình thường.

### Bộ lấy mẫu lại bộ nhận thức

Đối với mỗi hình ảnh trong prompt, ViT tạo ra N bản vá tokens. Bộ lấy mẫu lại Perceiver có K tiềm ẩn cố định có thể học được (Flamingo sử dụng K = 64). Mỗi khối bộ lấy mẫu lại là hai bước phụ:

1. Cross-attention: K tiềm ẩn tham dự trên tokens mảng N (Q từ tiềm ẩn, K/V từ các bản vá).
2. Self-attention + FFN trong tiềm ẩn.

Sau 6 khối bộ lấy mẫu lại, đầu ra là tokens hình ảnh K = 64 của 1024 mờ, bất kể ViT đã tạo ra bao nhiêu bản vá. Hình ảnh 224x224 (196 bản vá) và hình ảnh 480x480 (900 bản vá) đều thoát ra dưới dạng 64 tokens bộ lấy mẫu lại.

Đối với video, bộ lấy mẫu lại được áp dụng theo thời gian: các bản vá của mỗi khung hình tạo ra 64 tiềm ẩn và mã hóa vị trí thời gian cho phép model phân biệt t = 0 với t = N. Video đầy đủ trở thành tokens trực quan T * 64.

### cross-attention có cổng

Giữa mỗi M lớp LLM đông lạnh (Flamingo sử dụng M = 4), chèn một khối cross-attention có cổng mới:

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` là một vô hướng có thể học được được khởi tạo về không.
- `tanh(0) = 0`, vì vậy khi bắt đầu, branch có cổng đóng góp bằng không.
- Khi `alpha` rời khỏi con số không, sự đóng góp cross-attention tăng lên một cách suôn sẻ.
- Kết nối còn lại có nghĩa là ngay cả một cổng mở hoàn toàn cũng không ghi đè lên biểu diễn văn bản của LLM; nó chỉ thêm thông tin trực quan lên trên.

Đây là lựa chọn thiết kế quan trọng nhất duy nhất trong Flamingo: điều hòa trực quan là bổ sung, được kiểm soát và không khi khởi tạo. Flamingo ở bước 0 là một Chinchilla 70B hoàn hảo trên đầu vào chỉ có văn bản.

### cross-attention mặt nạ cho đầu vào xen kẽ

Trong một prompt như "<hình ảnh A> chú thích A <hình ảnh B> chú thích B <hình ảnh C> ?", mỗi token văn bản chỉ nên xem hình ảnh xuất hiện trước nó trong trình tự. Mặt nạ cross-attention thực thi: văn bản token ở vị trí `t` chỉ tham gia vào bộ lấy mẫu lại hình ảnh tokens có chỉ mục hình ảnh `i < i_t` trong đó `i_t` là hình ảnh gần đây nhất trước vị trí `t`. "Chỉ nhìn thấy hình ảnh cuối cùng trước đó" hoặc "nhìn thấy tất cả hình ảnh trước đó" đều là những lựa chọn hợp lệ; Flamingo đã chọn cái trước.

### Học few-shot trong ngữ cảnh

Một prompt Flamingo trông giống như:

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

model nhìn thấy mô hình hoàn thành và xuất ra "chim" (hoặc bất kỳ hình ảnh nào3 hiển thị). Không có gradient bước. Khả năng học tập trong ngữ cảnh của LLM đóng băng được thực hiện thông qua cross-attention được kiểm soát - đây là điểm nhấn của bài báo và tại sao nó lại quan trọng.

### Training dữ liệu

Flamingo được huấn luyện trên ba datasets:

1. MultiModal MassiveWeb (M3W): 43 triệu trang web với hình ảnh và văn bản xen kẽ, tái tạo thứ tự đọc.
2. Cặp hình ảnh-văn bản (ALIGN + LTIP): 4.4B cặp.
3. Cặp video-văn bản (VTP): 27 triệu video clip ngắn.

OBELICS (2023) là bản sao mở của kho dữ liệu web xen kẽ, mà Idefics, Idefics2 và hầu hết các models huấn luyện "giống chim hồng hạc" mở.

### OpenFlamingo và Rái cá

OpenFlamingo (2023) là bản sao mở. Kiến trúc giống hệt nhau (Bộ lấy mẫu lại Perceiver + cross-attention có cổng trên LLaMA đông lạnh hoặc MPT). Checkpoints ở 3B, 4B, 9B. Chất lượng tụt hậu Flamingo do LLM cơ sở nhỏ hơn và ít dữ liệu hơn.

Otter (2023) được xây dựng trên OpenFlamingo với khả năng điều chỉnh hướng dẫn trên MIMIC-IT (một dataset các hướng dẫn đa phương thức), hiển thị các cross-attention có cổng cũng hoạt động để theo dõi hướng dẫn.

### Con cháu

- Idefics / Idefics2 / Idefics3: Dòng cross-attention được kiểm soát của Hugging Face, dần dần đơn giản hơn (Idefics2 đã bỏ bộ lấy mẫu lại để ủng hộ tokens bản vá trực tiếp với gộp thích ứng).
- Quá trình chuyển đổi từ chim hồng hạc sang tắc kè hoa: đến năm 2024, nhiều đội chuyển sang hợp nhất sớm (Bài 12.11); cross-attention có cổng kiểu chim hồng hạc vẫn ở production nơi cần đóng băng xương sống.
- Đầu vào xen kẽ của Gemini: về mặt khái niệm kế thừa tính linh hoạt định dạng xen kẽ của Flamingo, mặc dù cơ chế chính xác là độc quyền.

### So sánh với BLIP-2

|| BLIP-2 · | Chim hồng hạc |
|---|---|---|
| Cầu trực quan | Q-Former một lần ở đầu vào | cross-attention có cổng ở mỗi M lớp |
| tokens trực quan | 32 mỗi hình ảnh | 64 cho mỗi hình ảnh trên mỗi lớp attn chéo |
| LLM đông lạnh | Có | Có |
| Few-shot trong ngữ cảnh | Yếu | Mạnh mẽ - trung tâm của tờ báo |
| Đầu vào xen kẽ | Không có hỗ trợ gốc | Vâng, mục tiêu thiết kế |
| Training dữ liệu | 130 triệu cặp | 1,3 tỷ cặp + 43 triệu trang xen kẽ |
| Số lượng Parameter | 188 triệu người được huấn luyện | ~10B được huấn luyện (các lớp attn chéo) |
| Điện toán | Số ngày trên 8 chiếc A100 | Tuần trên hàng nghìn TPUv4 |

Chọn BLIP-2 cho VQA một hình ảnh với ngân sách hạn hẹp. Chọn Flamingo/Idefics2 để suy luận xen kẽ, few-shot hoặc nhiều hình ảnh.

## Ứng dụng

`code/main.py` thể hiện:

1. Một bộ lấy mẫu lại Perceiver trên 36 bản vá giả tokens với 8 tiềm ẩn có thể học được (Python cross-attention thuần túy).
2. Một bước cross-attention được kiểm soát với `alpha = 0` → đầu ra bằng đầu vào (LLM không thay đổi), sau đó `alpha = 2.0` → đóng góp trực quan được trộn lẫn vào.
3. Trình tạo mặt nạ xen kẽ tạo mặt nạ attention 2D cho chuỗi "(hình ảnh 1) (văn bản 1) (hình ảnh 2) (văn bản 2)".

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-gated-bridge-diagnostic.md`. Với config của một VLM mở (Y/N lấy mẫu lại, tần số chéo, sơ đồ cổng), nó xác định các yếu tố dòng Flamingo và giải thích chiến lược đóng băng. Hữu ích để gỡ lỗi lý do tại sao fine-tune làm giảm hiệu suất văn bản (câu trả lời: cổng quá rộng quá nhanh).

## Bài tập

1. Tính toán số lượng parameter trực quan của Flamingo-9B: 9B LLM + 1.4B lớp cross-attention có cổng + bộ lấy mẫu lại 64M. Phần nào trong tổng số tham số được huấn luyện?

2. Thực hiện `y = tanh(alpha) * cross + x` dư được kiểm soát trong PyTorch. Cho thấy một cách thực nghiệm rằng với `alpha=0`, `y==x` chính xác tại điểm bắt đầu.

3. Đọc Phần 3.2 của OpenFlamingo (arXiv:2308.01390) về cách họ xử lý nhiều hình ảnh trong một batch khi mỗi prompt có số lượng hình ảnh khác nhau. Mô tả chiến lược khoảng đệm.

4. Tại sao mặt nạ cross-attention của Flamingo cho phép một văn bản token chỉ tham gia vào *hình ảnh gần đây nhất* trước đó thay vì tất cả các hình ảnh trước đó? Đọc bài báo Flamingo Phần 2.4 và giải thích sự đánh đổi.

5. few-shot theo ngữ cảnh: tạo một prompt với 4 ví dụ về "hình ảnh → màu của đối tượng chính" cho một biến thể Flamingo mới. Mô tả mẫu accuracy dự kiến khi bạn thay đổi số lượng ví dụ từ 0 đến 8.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Bộ lấy mẫu lại bộ nhận thức | "cross-attention tiềm ẩn cố định" | Mô-đun tạo ra K tokens cố định từ một số bản vá đầu vào thay đổi |
| cross-attention có cổng | "Cầu cổng Tanh" | Lớp còn lại `y = tanh(alpha)*cross + x`, alpha có thể học được, khởi tạo 0 |
| Đầu vào xen kẽ | "Trình tự hỗn hợp" | Định dạng Prompt với hình ảnh và văn bản được trộn tự do theo thứ tự đọc |
| LLM đông lạnh | "Không LLM gradients" | Trọng số của LLM văn bản không cập nhật; chỉ các lớp resampler + cross-attn mới huấn luyện |
| Few-shot | "Ví dụ trong ngữ cảnh" | Đưa ra một vài cặp (hình ảnh, câu trả lời) trong prompt; model khái quát hóa mà không cần tinh chỉnh |
| OBELICS | "Kho dữ liệu web xen kẽ" | Mở dataset của 141 triệu trang web với hình ảnh và văn bản theo thứ tự đọc |
| Chinchilla | "Cơ sở đông lạnh 70B" | Văn bản đóng băng của Flamingo LLM, từ bài báo Chinchilla của DeepMind |
| Lịch trình cổng | "Alpha di chuyển như thế nào" | Tốc độ mở cổng cross-attention trong training |
| Tần số chéo | "Mỗi M lớp" | Tần suất chèn một khối cross-attention có cổng; Chim hồng hạc sử dụng M = 4 |
| Chim hồng hạc mở | "Sao chép mở" | MosaicML/LAION mở checkpoint tại 3-9B; kiến trúc giống hệt với Flamingo |

## Đọc thêm

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — bài báo gốc.
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) - sinh sản mở.
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — kho dữ liệu web xen kẽ.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — kiến trúc Perceiver chung.
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) - hậu duệ của Flamingo được điều chỉnh theo hướng dẫn.
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) - đơn giản hóa hiện đại của cách tiếp cận Flamingo.
