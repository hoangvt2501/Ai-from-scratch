# OpenAI Chuẩn bị Framework và Framework An toàn DeepMind Frontier

> OpenAI Chuẩn bị Framework phiên bản 2 (Tháng Tư 2025) giới thiệu các Danh mục Nghiên cứu - Tự chủ tầm xa, Bao cát, Sao chép và Thích ứng Tự động, Các biện pháp bảo vệ phá hoại - khác với Danh mục được theo dõi. Danh mục được theo dõi trigger Báo cáo Khả năng cộng với Báo cáo Bảo vệ được Nhóm Tư vấn An toàn xem xét. FSF v3 của DeepMind (tháng 9 năm 2025, với các cấp độ khả năng được theo dõi được thêm vào ngày 17 tháng 4 năm 2026) gập quyền tự chủ thành ML lĩnh vực R&D và Cyber (ML quyền tự chủ R&D cấp 1 = tự động hóa hoàn toàn pipeline R&D AI với chi phí cạnh tranh so với con người + AI công cụ). FSF v3 giải quyết rõ ràng các alignment lừa đảo thông qua giám sát tự động đối với việc lạm dụng lý luận công cụ. Lưu ý trung thực: Các Danh mục Nghiên cứu trong PF v2 (bao gồm cả Quyền tự chủ tầm xa) không tự động trigger các biện pháp giảm thiểu; ngôn ngữ policy là "tiềm năng". Bản thân DeepMind cho biết giám sát tự động "sẽ không đủ lâu dài" nếu lý luận công cụ được củng cố.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, three-framework decision-table diff tool)
**Kiến thức tiên quyết:** Giai đoạn 15 · 19 (Anthropic RSP)
**Thời lượng:** ~45 phút

## Vấn đề

Bài 19 đọc kỹ policy tỷ lệ của Anthropic. Bài học này hoàn thành bức tranh bằng cách đọc OpenAI và DeepMind. Ba tài liệu này là anh em họ artifacts giải quyết cùng một câu hỏi - khi nào một phòng thí nghiệm biên giới nên tạm dừng hoặc cổng một model - và chúng hội tụ vào một tập hợp nhỏ các danh mục và phân kỳ ở những nơi cụ thể quan trọng.

Sự hội tụ: cả ba đều dán nhãn quyền tự chủ tầm xa như một khả năng đáng class theo dõi. Cả ba đều thừa nhận hành vi lừa đảo (alignment giả mạo, bao cát) là một class rủi ro cụ thể. Cả ba đều có một cơ quan đánh giá nội bộ. Sự khác biệt: OpenAI chia các danh mục thành "Theo dõi" (giảm thiểu bắt buộc) và "Nghiên cứu" (không có trigger tự động). DeepMind gộp quyền tự chủ thành hai miền thay vì đặt tên riêng biệt. Các phòng thí nghiệm đặt tên Tracked vs Research, hoặc Critical vs Moderate, hoặc Tier-1 vs Tier-2; Hậu quả hoạt động của nhóm năng lực A là khác nhau giữa các phòng thí nghiệm.

Đọc chúng cùng nhau là bài tập hữu ích. Khả năng tương tự có thể là "giảm thiểu bắt buộc" ở Anthropic, "được giám sát nhưng không kích hoạt" ở OpenAI và "được theo dõi trong một miền cụ thể" tại DeepMind. Tư thế policy rất quan trọng.

## Khái niệm

### OpenAI Chuẩn bị Framework phiên bản 2 (Tháng Tư 2025)

Kết cấu:

- **Danh mục được theo dõi**: Báo cáo Khả năng trigger (những gì model có thể làm) cộng với Báo cáo Bảo vệ (những biện pháp giảm thiểu đang được áp dụng). Được xem xét bởi Nhóm Tư vấn An toàn trước khi triển khai.
- **Danh mục nghiên cứu**: model khả năng mà phòng thí nghiệm đang theo dõi nhưng chưa cam kết thực hiện các biện pháp giảm thiểu cụ thể. Bao gồm quyền tự chủ tầm xa, bao cát, sao chép và thích ứng tự động, làm suy yếu các biện pháp bảo vệ.

Khung chính trong phiên bản 2: Danh mục nghiên cứu không tự động trigger các biện pháp giảm thiểu. Ngôn ngữ policy là giảm thiểu "tiềm năng". Đây là delta cụ thể cần theo dõi: vị trí của một danh mục trong Tracked vs Research là bản lề giữa "gated" và "observed".

### DeepMind Frontier Safety Framework v3 (Tháng Chín 2025; Mức độ khả năng được theo dõi được thêm vào ngày 17 tháng 4 năm 2026)

Kết cấu:

- **Mức năng lực quan trọng (CCL)**: ngưỡng năng lực trên năm lĩnh vực: Cyber, Bio, ML R&D, CBRN, Autonomy (gộp thành ML R&D và Cyber).
- **Mức độ khả năng được theo dõi**: độ chi tiết bổ sung được thêm vào tháng 4 năm 2026. Ví dụ cụ thể: ML quyền tự chủ R&D cấp độ 1 = tự động hóa hoàn toàn AI pipeline R&D với chi phí cạnh tranh so với con người + công cụ AI.
- **Giám sát alignment lừa đảo**: cam kết rõ ràng về giám sát tự động đối với việc lạm dụng lý luận công cụ.

Khung tự chủ khác với OpenAI. DeepMind không giữ "Quyền tự chủ" như một tên miền cấp cao nhất; nó được gộp vào các lĩnh vực mà quyền tự chủ sẽ gây hại (ML nghiên cứu và phát triển và không gian mạng). Lập luận là quyền tự chủ mà không có tên miền là khả năng không có rủi ro; Lập luận phản đối là quyền tự chủ giữa các tên miền là một siêu rủi ro mà framework nên nêu tên.

### Cả ba hội tụ về điều gì

- Nhóm Tư vấn An toàn Nội bộ (được đặt tên Anthropic SAG, OpenAI SAG, ủy ban nội bộ DeepMind). Xem lại trước khi triển khai để có models có khả năng cao.
- Đề cập rõ ràng về alignment / alignment giả mạo lừa đảo như một class rủi ro.
- Đứng artifacts theo nhịp đã khai báo (Anthropic: Lộ trình An toàn Biên giới, Báo cáo Rủi ro; OpenAI: Báo cáo Năng lực và Biện pháp Bảo vệ; DeepMind: Chu kỳ cập nhật FSF).
- Thừa nhận rằng các biện pháp phòng thủ chỉ giám sát có trần. DeepMind nói rõ ràng: "giám sát tự động sẽ không đủ trong thời gian dài".

### Nơi chúng phân kỳ

- **Anthropic**: Đã xóa cam kết tạm dừng trong phiên bản 3.0; AI ngưỡng R&D-4 là cổng tiếp theo được đặt tên.
- **OpenAI**: Phân chia theo dõi và Nghiên cứu; Danh mục nghiên cứu (bao gồm Quyền tự chủ tầm xa) không tự động cổng.
- **DeepMind**: quyền tự chủ được gộp vào các lĩnh vực khác; Mức độ khả năng được theo dõi bổ sung độ chi tiết vào tháng 4 năm 2026.

### Bao cát: một khả năng cụ thể làm phức tạp cả ba

Bao cát (một model kém hiệu quả về mặt chiến lược trong các đánh giá) nằm trong Danh mục Nghiên cứu của OpenAI. RSP phiên bản 3.0 của Anthropic giải quyết vấn đề này thông qua khoảng cách ngữ cảnh đánh giá (Bài 1). DeepMind giải quyết nó thông qua giám sát alignment lừa đảo trong FSF v3.

Nếu một model bao cát đánh giá, ngưỡng năng lực của mọi framework đều bị đánh giá thấp. framework chỉ hoạt động nếu phép đo hoạt động. Đây là lý do tại sao đo lường bên ngoài (Bài 21, METR) và đánh giá đối nghịch là cần thiết ngoài việc tự đánh giá trong phòng thí nghiệm.

### skill đọc policy

- Định vị: mọi khả năng bạn quan tâm đều có thể tìm thấy trong policy. Nếu không tìm thấy, policy không che nó.
- Phân loại: được theo dõi (giảm thiểu triggers) hay Nghiên cứu (theo dõi nhưng không kích hoạt)? OpenAI đặt tên cho điều này; Anthropic và DeepMind có những thứ tương đương của riêng họ.
- Nhịp điệu: policy được cập nhật theo lịch trình đã khai báo hay chỉ sau các sự kiện cụ thể? Nhịp được công bố mạnh hơn.
- Độc lập: đánh giá bên ngoài là bắt buộc hay tùy chọn? Anthropic hợp tác với Apollo và Viện An toàn AI Hoa Kỳ; OpenAI với METR; DeepMind với SAG nội bộ chủ yếu.

## Ứng dụng

`code/main.py` triển khai một công cụ khác biệt bảng quyết định nhỏ. Với một khả năng (tự chủ, alignment lừa đảo, tự động hóa R & D, nâng cao không gian mạng, v.v.), nó cho biết cách mỗi policies trong số ba  phân loại khả năng và những biện pháp giảm thiểu trigger. Nó là một công cụ hỗ trợ đọc, không phải là một công cụ policy.

## Sản phẩm bàn giao

`outputs/skill-cross-policy-diff.md` tạo ra một so sánh policy chéo cho một khả năng cụ thể, sử dụng ba frameworks làm tham chiếu.

## Bài tập

1. Chạy `code/main.py`. Xác nhận đầu ra của công cụ diff khớp với policies cho ít nhất hai khả năng mà bạn có thể xác minh dựa trên các tài liệu nguồn.

2. Đọc toàn bộ OpenAI Chuẩn bị Framework v2. Xác định từng Danh mục Nghiên cứu. Đối với mỗi mục, hãy viết một câu về lý do tại sao nó lại ở trong Nghiên cứu thay vì Theo dõi.

3. Đọc toàn bộ DeepMind FSF phiên bản 3, cùng với bản cập nhật Mức độ khả năng được theo dõi tháng 4 năm 2026. Xác định ML tiêu chí đánh giá cụ thể của quyền tự chủ R&D cấp độ 1. Bạn sẽ đo lường nó bên ngoài như thế nào?

4. Bao cát nằm trong Danh mục Nghiên cứu của OpenAI. Thiết kế một đánh giá sẽ buộc một model bao cát tiết lộ khả năng thực tế của nó. Tham khảo thảo luận về trò chơi ngữ cảnh đánh giá Bài 1.

5. So sánh ba policies về một khả năng cụ thể (lựa chọn của bạn). Kể tên phân loại policy nào bạn thấy nghiêm ngặt nhất và ít nhất. Biện minh bằng văn bản nguồn.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| Chuẩn bị Framework | "OpenAI mở rộng quy mô policy" | PF v2 (tháng 4 năm 2025); Danh mục được theo dõi so với Nghiên cứu |
| Danh mục được theo dõi | "Giảm thiểu bắt buộc" | Triggers Báo cáo Khả năng + Biện pháp Bảo vệ; Đánh giá SAG |
| Hạng mục nghiên cứu | "Chỉ được giám sát" | Được theo dõi nhưng không có giảm thiểu tự động; bao gồm Quyền tự chủ tầm xa |
| An toàn biên giới Framework | "policy mở rộng quy mô của DeepMind" | FSF v3 (Tháng 9 năm 2025) + Mức khả năng được theo dõi (Tháng Tư 2026) |
| CCL | "Mức năng lực quan trọng" | Ngưỡng DeepMind trên mỗi miền (Cyber, Bio, ML R&D, CBRN) |
| ML Tự chủ R&D cấp độ 1 | "Tự động hóa R&D" | Hoàn toàn tự động hóa pipeline R & D AI với chi phí cạnh tranh |
| Bao cát | "Chiến lược kém hiệu quả" | Model hoạt động kém hiệu quả trên các đánh giá; in OpenAI Danh mục nghiên cứu |
| Lý luận công cụ | "Lý luận phương tiện-kết thúc" | Lý luận về cách đạt được mục tiêu; mục tiêu giám sát DeepMind |

## Đọc thêm

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — Thông báo v2.
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) - tài liệu đầy đủ.
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — Thông báo FSF v3.
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) - Thêm Cấp độ khả năng được theo dõi.
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — ví dụ về Báo cáo rủi ro định dạng FSF.
