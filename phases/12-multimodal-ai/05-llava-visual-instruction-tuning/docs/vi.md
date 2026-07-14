# LLaVA và điều chỉnh hướng dẫn trực quan

> LLaVA (Tháng Tư 2023) là kiến trúc đa phương thức được sao chép nhiều nhất trên hành tinh. Nó thay thế Q-Former của BLIP-2 bằng MLP 2 lớp, thay thế cross-attention có cổng của Flamingo bằng sự nối token ngây thơ và được huấn luyện trên 158k lượt hướng dẫn trực quan do GPT-4 tạo ra từ chú thích chỉ có văn bản. Bất kỳ học viên nào đã xây dựng một VLM từ năm 2023 đến năm 2026 đều đã xây dựng một số biến thể của LLaVA. LLaVA-1.5 đã thêm AnyRes. LLaVA-NeXT tăng độ phân giải. LLaVA-OneVision thống nhất hình ảnh, đa hình ảnh và video trong một công thức. Bài học này đọc công thức, triển khai máy chiếu và giải thích lý do tại sao "đơn giản hơn chiến thắng".

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, projector + instruction-template builder)
**Kiến thức tiên quyết:** Giai đoạn 12 · 02 (CLIP), Giai đoạn 11 (LLM Kỹ thuật - điều chỉnh hướng dẫn)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Xây dựng máy chiếu MLP 2 lớp ánh xạ bản vá ViT embeddings (mờ 1024) với độ mờ embedding của LLM (mờ 4096).
- Đi bộ LLaVA hai giâytagcông thức: (1) máy chiếu alignment trên các cặp phụ đề 558k, (2) điều chỉnh hướng dẫn trực quan trên 158k lượt do GPT-4 tạo ra.
- Xây dựng một prompt định dạng LLaVA với hình ảnh token trình giữ chỗ, system prompt và user/assistant lượt.
- Giải thích lý do tại sao cộng đồng chuyển từ Q-Former sang MLP bất chấp chiến thắng token ngân sách của Q-Former.

## Vấn đề

Q-Former của BLIP-2 (Bài 12.03) nén một hình ảnh thành 32 tokens. Sạch sẽ, hiệu quả, tốt cho benchmarks. Nhưng nó có hai vấn đề.

Đầu tiên, Q-Former có thể huấn luyện được nhưng loss của nó không phải là nhiệm vụ cuối cùng. Giai đoạn 1 huấn luyện ITC + ITM + ITG. Giai đoạn 2 huấn luyện LM loss. Các truy vấn học một số biểu diễn trung gian mà LLM sau đó phải giải mã. Thông tin bị mất trong nút thắt cổ chai.

Thứ hai, Q-Former có 188 triệu tham số và ở thang đo năm 2023 của LLaVA, bạn phải đồng thiết kế nó với LLM mục tiêu của mình. Thay đổi LLM, huấn luyện lại Q-Former. Thay đổi tầm nhìn encoder, huấn luyện lại. Mỗi sự kết hợp là một dự án R&D riêng biệt.

Câu trả lời của LLaVA thật đáng xấu hổ vì sự đơn giản của nó: lấy bản vá 576 của ViT tokens, chuyển từng bản qua MLP (`1024 → 4096 → 4096`) 2 lớp và kết xuất tất cả 576 vào trình tự đầu vào của LLM. Không có nút thắt cổ chai. Không có giai đoạn 1 pretraining về các mục tiêu kỳ lạ. Chỉ cần huấn luyện MLP trên loss LM trực tiếp.

Dữ liệu đến từ đâu? Thông tin chi tiết thứ hai của LLaVA: sử dụng GPT-4 (chỉ văn bản) để tạo dữ liệu hướng dẫn. Cung cấp GPT-4 dữ liệu chú thích COCO và hộp giới hạn cho một hình ảnh, yêu cầu nó tạo ra các cuộc trò chuyện, mô tả và các câu hỏi suy luận phức tạp. 158k phản hồi lệnh quay miễn phí. Không có chú thích của con người.

Kết quả: một VLM chạy trên 8 chiếc A100 trong một ngày, đánh bại Flamingo trên MMMU và shipped một checkpoint mở mà cộng đồng có thể mở rộng. Đến cuối năm 2023, nó đã tạo ra 50+ nhánh.

## Khái niệm

### Kiến trúc

LLaVA-1.5 ở 13B:
- Tầm nhìn encoder: CLIP ViT-L/14 @ 336 (bị đóng băng trong stage 1, tùy chọn không đóng băng stage 2).
- Máy chiếu: MLP 2 lớp với kích hoạt GELU, `1024 → 4096 → 4096`.
- LLM: Vicuna-13B (sau này Llama-3.1-8B).

Forward pass trên prompt hình ảnh + văn bản:

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

Hình ảnh chiếm 576 tokens ngữ cảnh LLM. Ở ngữ cảnh 2048, điều đó còn lại 1472 tokens cho văn bản. Ở ngữ cảnh 32k, đó là lỗi làm tròn.

### Giai đoạn 1: alignment máy chiếu

Đóng băng ViT. Đóng băng LLM. Chỉ huấn luyện MLP 2 lớp. Dataset: 558k cặp chú thích hình ảnh (LAION-CC-SBU). Loss: mô hình ngôn ngữ trên chú thích, điều kiện trên tokens hình ảnh được chiếu.

Trong một epoch duy nhất ở batch 128, điều này được thực hiện trong vài giờ. Máy chiếu học cách ánh xạ ViT-space với LLM-space. Không có giám sát nhiệm vụ cụ thể.

### Giai đoạn 2: điều chỉnh hướng dẫn trực quan

Hủy đóng băng máy chiếu (vẫn có thể huấn luyện được). Giải phóng LLM (thường là hoàn toàn, đôi khi LoRA). Huấn luyện trên 158k lượt hướng dẫn trực quan.

Dữ liệu hướng dẫn là mẹo. Liu và cộng sự đã tạo ra nó bởi:
1. Chụp ảnh COCO.
2. Trích xuất mô tả văn bản (5 chú thích của con người + danh sách hộp giới hạn).
3. Gửi đến GPT-4 với ba mẫu prompt:
   - Cuộc trò chuyện: "Tạo cuộc đối thoại qua lại giữa người dùng và trợ lý về hình ảnh này."
   - Mô tả chi tiết: "Đưa ra mô tả phong phú, chi tiết về hình ảnh."
   - Lý luận phức tạp: "Đặt một câu hỏi đòi hỏi suy luận về hình ảnh, sau đó trả lời nó."
4. Phân tích cú pháp đầu ra của GPT-4 thành các cặp (lệnh, phản hồi).

Không có điều nào trong số này chạm trực tiếp vào hình ảnh - chỉ có mô tả văn bản. GPT-4 ảo giác nội dung hình ảnh hợp lý. Một số nhiễu, nhưng nó hoạt động: 158k lượt là đủ để mở khóa cuộc đối thoại.

### Tại sao cộng đồng sao chép điều này

- Không có tổn thất cụ thể ở giai đoạn 1 để điều chỉnh. LM loss xuyên suốt.
- Máy chiếu hoạt động trong vài giờ, không phải vài ngày.
- LLM có thể được hoán đổi (LLaVA-Llama2, LLaVA-Mistral, LLaVA-Llama3) chỉ bằng cách huấn luyện lại máy chiếu.
- Dữ liệu hướng dẫn trực quan pipeline sử dụng GPT-4 và rẻ để tạo lại cho một miền mới.

### LLaVA-1.5 và LLaVA-NeXT

LLaVA-1.5 (Tháng 10 năm 2023) được bổ sung:
- Dữ liệu nhiệm vụ học thuật (VQA, OKVQA, RefCOCO) được trộn lẫn vào điều chỉnh hướng dẫn.
- Tốt hơn system prompt.
- 2048 → bối cảnh 32k.

LLaVA-NeXT (tháng 1 năm 2024) cho biết thêm:
- AnyRes: chia hình ảnh có độ phân giải cao thành lưới 2x2 hoặc 1x3 gồm 336x336 cắt, cộng với một hình thu nhỏ có độ phân giải thấp toàn cầu. Mỗi lần cắt trở thành 576 tokens; tổng cộng khoảng 2880 tokens hình ảnh cho mỗi hình ảnh. Các tác vụ OCR và biểu đồ đã nhảy.
- Kết hợp dữ liệu hướng dẫn tốt hơn với ShareGPT4V (phụ đề GPT-4V chất lượng cao).
- LLMs cơ sở mạnh hơn (Mistral-7B, Yi-34B).

### LLaVA-OneVision

Bài học 12.08 trình bày chuyên sâu về OneVision. Phiên bản ngắn: cùng một máy chiếu, nhưng được huấn luyện với chương trình giảng dạy bao gồm một hình ảnh, nhiều hình ảnh và video trong một model với ngân sách token hình ảnh được chia sẻ.

### So sánh với Q-Former

|| Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| tokens hình ảnh trên mỗi hình ảnh | 32 | 576 (cơ sở) hoặc 2880 (AnyRes) |
| Tham số có thể huấn luyện | 188 triệu + LM | 40 triệu + LM |
| Giai đoạn 1 loss | ITC + ITM + ITG | Chỉ LM |
| LLM thả vào | Yêu cầu huấn luyện lại | Hoán đổi với huấn luyện lại tối thiểu |
| Đa hình ảnh | Khó xử | Tự nhiên (concat) |
| Băng hình | Khó xử | Tự nhiên (concat trên mỗi khung hình) |
| Token ngân sách | Nhỏ | Lớn |

MLP giành chiến thắng nhờ sự đơn giản và token linh hoạt. Q-Former giành chiến thắng trên ngân sách token. Đến cuối năm 2023, ngân sách token không còn là ràng buộc ràng buộc (LLM bối cảnh tăng lên 32k-128k+) và sự đơn giản chiếm ưu thế.

### Định dạng prompt

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` là một trình giữ chỗ token. Trước khi tokenization, nó được thay thế bằng tokens trực quan 576 (hoặc 2880 với AnyRes). Tokenizer thấy một trình tự dài hơn một chút so với nó đã được huấn luyện, nhưng LLM xử lý đầu vào mới vì giai đoạn 1 đã dạy nó.

### Parameter kinh tế

Sự cố LLaVA-1.5-7B:
- CLIP ViT-L/14 @ 336: 303M (đóng băng stage 1, thường không đóng băng stage 2).
- Máy chiếu (tuyến tính 2x): ~22M có thể huấn luyện.
- Llama-7B: 7B.
- Tổng cộng: 7,3 tỷ tham số. Có thể huấn luyện trong giai đoạn 2: máy chiếu 7B + 22M đầy đủ.

Chi phí Training cho giai đoạn 2: ~20 giờ trên 8xA100. Đây là con số quan trọng - một ngày, một nút, có thể tái tạo. Đó là lý do tại sao LLaVA lan rộng.

## Ứng dụng

`code/main.py` thực hiện:

1. Máy chiếu MLP 2 lớp (mờ 16 → 32 → 32 cho cân đồ chơi) ở Python nguyên chất.
2. pipeline xây dựng prompt: system prompt + `<image>` được thay thế bằng N tokens dự kiến + lượt người dùng + trình giữ chỗ tạo trợ lý.
3. Trình hiển thị cho khối hình ảnh 576 token trông như thế nào trong ngữ cảnh LLM (tỷ lệ phần trăm ngữ cảnh 2k / 32k / 128k được sử dụng).

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-llava-vibes-eval.md`. Với một checkpoint gia đình LLaVA, nó chạy một bộ 10 prompt vibes-eval (3 chú thích, 3 VQA, 2 lý luận, 2 từ chối) và báo cáo thẻ điểm mà con người có thể đọc được. Không phải là một benchmark; một bài kiểm tra khói để xác nhận máy chiếu và LLM đang kết nối tốt.

## Bài tập

1. Tính toán số lượng parameter có thể huấn luyện cho máy chiếu MLP 2 lớp ở `1024 → 4096 → 4096`. Với GELU và bias, nó đại diện cho phần nào của LLaVA-13B?

2. Xây dựng một prompt LLaVA cho trường hợp "từ chối" - hình ảnh chứa một cá nhân. Viết phản hồi của trợ lý dự kiến. Tại sao LLaVA nên từ chối zero-shot này và dữ liệu training nào sẽ cần thiết để củng cố việc từ chối?

3. Đọc phần AnyRes của blog LLaVA-NeXT. Tính toán số token hình ảnh cho hình ảnh 1344x672 tại AnyRes. So sánh với tokens cơ sở 576 ở 336x336.

4. Máy chiếu LLaVA stage-1 được huấn luyện với LM loss về phụ đề. Điều gì xảy ra nếu bạn bỏ qua giai đoạn 1 và chuyển thẳng sang giai đoạn 2 (điều chỉnh hướng dẫn trực quan)? Trích dẫn cắt bỏ VLMs lăng trụ (arXiv: 2402.07865) để có câu trả lời.

5. LLaVA-Instruct-150k sử dụng GPT-4 có phụ đề COCO để tạo hướng dẫn. Đối với miền mới (tia X y tế, hình ảnh vệ tinh), hãy mô tả dữ liệu bốn bước pipeline để tạo hướng dẫn miền. Điều gì có thể xảy ra ở mỗi bước?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Máy chiếu | "Cầu MLP" | MLP 2 lớp với ánh xạ GELU ViT mờ đến LLM mờ |
| Hình ảnh token | "<image> Trình giữ chỗ" | Prompt điểm đánh dấu được thay thế bằng N tokens thị giác chiếu trước khi inference |
| Điều chỉnh hướng dẫn trực quan | "LLaVA giai đoạn 2" | Training trên bộ ba do GPT-4 tạo (hình ảnh, hướng dẫn, phản hồi) |
| Giai đoạn 1 alignment | "Máy chiếu pretraining" | Đóng băng ViT và LLM, máy chiếu tàu hỏa với loss LM trên phụ đề |
| AnyRes | "Lát gạch đa cắt" | Chia hình ảnh có độ phân giải cao thành lưới ô và nối tokens hình ảnh của từng ô |
| LLaVA-Hướng dẫn | "GPT-4 tạo" | 158k cặp lệnh-phản hồi được tổng hợp từ phụ đề COCO + GPT-4 |
| Tầm nhìn encoder đóng băng | "Xương sống bị khóa" | Trọng số CLIP không cập nhật trong stage 1, đôi khi cũng không cập nhật trong stage 2 |
| Chia sẻGPT4V | "Phụ đề tốt hơn" | 1 triệu phụ đề dày đặc do GPT-4V tạo ra, được sử dụng cho alignment chất lượng cao hơn |
| VQA | "Trả lời câu hỏi trực quan" | Nhiệm vụ trả lời câu hỏi dạng tự do về hình ảnh |
| VLMs lăng trụ | "Giấy không gian thiết kế" | Karamcheti 2024 kiểm tra một cách có hệ thống các lựa chọn dữ liệu và máy chiếu |

## Đọc thêm

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) - bài báo LLaVA.
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) - LLaVA-1.5.
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) - chú thích dày đặc dataset.
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) - cắt bỏ không gian thiết kế.
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) - video đơn ảnh, đa hình ảnh thống nhất.
