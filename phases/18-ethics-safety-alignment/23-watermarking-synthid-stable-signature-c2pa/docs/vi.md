# Hình mờ — SynthID, Chữ ký ổn định, C2PA

> Ba công nghệ cấu trúc 2026 nguồn gốc nội dung do AI tạo ra. SynthID (Google DeepMind) — hình mờ hình ảnh ra mắt Tháng Tám 2023, văn bản + video Tháng Năm 2024 (Gemini + Veo), văn bản mã nguồn mở Tháng Mười 2024 thông qua Bộ công cụ GenAI có trách nhiệm, máy dò đa phương tiện hợp nhất Tháng Mười Một 2025 cùng với Gemini 3 Pro. Hình mờ văn bản điều chỉnh xác suất token sampling tiếp theo một cách không thể nhận thấy; image/video hình mờ tồn tại sau khi nén, cắt xén, bộ lọc, thay đổi tốc độ khung hình. Chữ ký ổn định (Fernandez và cộng sự, ICCV 2023, arXiv:2303.15435) — tinh chỉnh decoder khuếch tán tiềm ẩn để mọi đầu ra đều chứa một thông báo cố định; được cắt (10% nội dung) được phát hiện hình ảnh tạo >90% ở FPR<1e-6. Tiếp theo "Chữ ký ổn định không ổn định" (arXiv:2405.07145, Tháng Năm 2024) — fine-tuning xóa hình mờ trong khi vẫn giữ nguyên chất lượng. C2PA — tiêu chuẩn siêu dữ liệu rõ ràng giả mạo, được ký bằng mật mã (C2PA 2.2 Explainer 2025). Watermarking và C2PA bổ sung cho nhau: siêu dữ liệu có thể bị loại bỏ nhưng có nguồn gốc phong phú hơn; Hình mờ tồn tại thông qua quá trình chuyển mã nhưng mang ít thông tin hơn.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, token-watermark embed + detect)
**Kiến thức tiên quyết:** Giai đoạn 10 · 04 (sampling), Giai đoạn 01 · 09 (lý thuyết thông tin)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Mô tả hình mờ cấp token (kiểu văn bản SynthID) và cơ chế mà nó có thể phát hiện được.
- Mô tả Chữ ký ổn định và cuộc tấn công xóa năm 2024 đã phá vỡ nó.
- Nêu vai trò của C2PA và lý do tại sao nó bổ sung cho watermarking.
- Mô tả những hạn chế chính: tín hiệu model cụ thể, tính mạnh mẽ khi diễn giải và các cuộc tấn công bảo toàn ý nghĩa (arXiv: 2508.20228).

## Vấn đề

Giai đoạn 2023-2024 chứng kiến deepfake và nội dung do AI tạo ra tham gia vào bối cảnh chính trị và người tiêu dùng trên quy mô lớn. Watermarking là tín hiệu xuất xứ kỹ thuật được đề xuất: đánh dấu các thế hệ tại thời điểm tạo, phát hiện chúng sau. Bằng chứng năm 2025: không có hình mờ nào là mạnh mẽ vô điều kiện, nhưng được phân lớp với siêu dữ liệu C2PA, sự kết hợp này cung cấp một câu chuyện xuất xứ có thể sử dụng được.

## Khái niệm

### Hình mờ văn bản (kiểu văn bản SynthID)

Cơ chế Kirchenbauer et al. 2023, do Google sản xuất:

1. Ở mỗi bước giải mã, băm K tokens trước đó để tạo ra một phân vùng giả ngẫu nhiên của từ vựng thành các tập hợp "xanh lá cây" và "đỏ".
2. Bias sampling về phía bộ màu xanh lá cây bằng cách thêm δ vào logits màu xanh lá cây.
3. Thế hệ chứa nhiều tokens xanh hơn là cơ hội tạo ra.

Phát hiện: băm lại từng tiền tố, đếm tokens xanh trong thế hệ, tính điểm z. Điểm z là >0 đối với văn bản có hình mờ, ~0 đối với văn bản của con người.

Tính chất:
- Người đọc không thể nhận thấy (δ đủ nhỏ để chất lượng loss là nhỏ).
- Có thể phát hiện với quyền truy cập vào chức năng phân vùng từ vựng.
- Không mạnh mẽ để diễn giải - viết lại văn bản sẽ phá hủy tín hiệu.

SynthID-text là mã nguồn mở Tháng Mười 2024 thông qua Bộ công cụ GenAI có trách nhiệm của Google.

### Chữ ký ổn định (hình ảnh)

Fernandez và cộng sự. ICCV 2023. Fine-tune decoder khuếch tán tiềm ẩn để mọi hình ảnh được tạo đều chứa một thông điệp nhị phân cố định được nhúng trong biểu diễn tiềm ẩn. Phát hiện được giải mã từ tiềm ẩn bằng decoder thần kinh. Hình ảnh bị cắt (đến 10% nội dung) được phát hiện >90% ở FPR<1e-6.

Tháng 5 năm 2024 "Chữ ký ổn định không ổn định" (arXiv:2405.07145): fine-tuning decoder xóa hình mờ trong khi vẫn giữ được chất lượng hình ảnh. fine-tuning sau thế hệ đối nghịch là rẻ; Độ bền đối nghịch của hình mờ bị hạn chế.

### Máy dò hợp nhất SynthID (Tháng Mười Một 2025)

Cùng với Gemini 3 Pro: một máy dò đa phương tiện đọc tín hiệu SynthID từ văn bản, hình ảnh, âm thanh và video trong một API. Thống nhất stack xuất xứ của Google.

### C2PA

Liên minh về nguồn gốc nội dung và tính xác thực. Tiêu chuẩn siêu dữ liệu giả mạo được ký bằng mật mã. Giải thích C2PA 2.2 (2025). Bản kê khai C2PA ghi lại các tuyên bố xuất xứ (ai đã tạo, khi nào, biến đổi nào) được ký bởi khóa của người tạo.

Bổ sung cho watermarking:
- Siêu dữ liệu có thể bị tước bỏ; hình mờ không thể (dễ dàng).
- Siêu dữ liệu phong phú (chuỗi nguồn gốc đầy đủ); hình mờ mang các bit.
- C2PA phụ thuộc vào việc áp dụng nền tảng; hình mờ được nhúng tự động.

Google tích hợp cả trong Tìm kiếm, Quảng cáo và "Giới thiệu về hình ảnh này".

### Hạn chế

- **Model cụ thể.** SynthID tạo hình mờ các thế hệ từ models hỗ trợ SynthID. Một thế hệ từ một model không có SynthID không được đánh dấu mờ, vì vậy "không có tín hiệu SynthID" không phải là bằng chứng xác thực.
- **Diễn giải.** Hình mờ văn bản không tồn tại sau khi diễn giải giữ gìn ý nghĩa.
- **Các cuộc tấn công biến đổi.** arXiv:2508.20228 (2025) cho thấy các cuộc tấn công bảo toàn ý nghĩa phá hủy cả hình mờ văn bản và nhiều hình mờ hình ảnh.
- **Fine-tune xóa.** Theo "Chữ ký ổn định không ổn định", fine-tuning sau tạo sẽ xóa hình mờ được nhúng.

### Đạo luật AI EU Điều 50

Quy tắc minh bạch cho việc gắn nhãn nội dung do AI tạo (bản nháp đầu tiên tháng 12 năm 2025, bản nháp thứ hai tháng 3 năm 2026, dự kiến cuối cùng tháng 6 năm 2026 theo [European Commission status page](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)). Bộ quy tắc vẫn còn trong dự thảo kể từ tháng 4 năm 2026 và thời gian có thể thay đổi. Lớp quy định yêu cầu lớp kỹ thuật. Deepfake phải được dán nhãn.

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài 22-23 nói về những gì model phát ra (dữ liệu cá nhân, tín hiệu xuất xứ). Bài 27 bao gồm quản trị dữ liệu training. Bài 24 là framework quy định yêu cầu các biện pháp kỹ thuật này.

## Ứng dụng

`code/main.py` xây dựng hình mờ văn bản đồ chơi. Tokens là số nguyên 0..N-1; có hình mờ sampling thiên vị đối với bộ màu xanh lá cây được xác định bằng hàm băm. Một máy dò tính toán điểm z token xanh lá cây. Bạn có thể quan sát phát hiện ở thế hệ 1000-token, xem diễn giải phá hủy tín hiệu và đo tỷ lệ dương tính giả trên văn bản của con người.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-provenance-audit.md`. Với việc triển khai nội dung với tuyên bố xuất xứ, nó sẽ kiểm tra: cơ chế hình mờ (nếu có), chuỗi ký C2PA (nếu có), tính mạnh mẽ đối nghịch của từng loại và phạm vi theo phương thức.

## Bài tập

1. Chạy `code/main.py`. Báo cáo điểm z cho thế hệ 1000 token có hình mờ so với văn bản do con người soạn thảo. Xác định tỷ lệ dương tính giả ở ngưỡng tin cậy 95%.

2. Thực hiện một cuộc tấn công diễn giải thay thế 30% tokens bằng các từ đồng nghĩa. Đo lại điểm z.

3. Đọc Kirchenbauer et al. 2023 Phần 6 về độ bền bỉ. Tại sao hình mờ văn bản không thành công khi diễn giải nhưng hình mờ hình ảnh vẫn tồn tại sau khi cắt?

4. Thiết kế triển khai sử dụng siêu dữ liệu SynthID-text + C2PA. Mô tả chuỗi xuất xứ mà người tiêu dùng nhìn thấy. Xác định một chế độ lỗi của mỗi thành phần.

5. Kết quả "Chữ ký ổn định không ổn định" năm 2024 cho thấy fine-tuning xóa hình mờ hình ảnh. Thiết kế kiểm soát triển khai để hạn chế cuộc tấn công này - ví dụ: yêu cầu các bản phát hành fine-tuned checkpoints có chữ ký.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| SynthID | "Hình mờ của Google" | Tín hiệu xuất xứ đa phương thức; văn bản, hình ảnh, âm thanh, video |
| Hình mờ Token | "Phong cách Kirchenbauer" | Hình mờ văn bản sampling thiên vị có thể phát hiện được thông qua điểm z token xanh lá cây |
| Chữ ký ổn định | "Hình mờ hình ảnh" | Fine-tuned-decoder hình mờ; ICCV 2023 |
| C2PA | "Tiêu chuẩn siêu dữ liệu" | Siêu dữ liệu xuất xứ rõ ràng giả mạo được ký bằng mật mã |
| Diễn giải độ mạnh mẽ | "Viết lại có phá vỡ nó không" | Thuộc tính hình mờ văn bản; hiện đang giới hạn |
| Loại bỏ Fine-tune | "đối nghịch không có hình mờ" | Tấn công xóa hình mờ hình ảnh thông qua decoder fine-tuning |
| Máy dò đa phương thức | "SynthID thống nhất" | Tháng Mười Một 2025 API thống nhất trên các phương thức |

## Đọc thêm

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — cơ chế hình mờ token
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — giấy hình mờ hình ảnh
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — cuộc tấn công loại bỏ
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — hình mờ phương thức chéo
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — tiêu chuẩn siêu dữ liệu
