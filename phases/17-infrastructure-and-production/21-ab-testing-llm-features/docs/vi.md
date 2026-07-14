# A/B Testing LLM Features - GrowthBook, Statsig và vấn đề Vibes

> Thử nghiệm A/B truyền thống không được xây dựng cho LLMs không xác định. Sự khác biệt quan trọng: đánh giá trả lời "model có thể làm công việc không?" A/B thử nghiệm trả lời "người dùng có quan tâm không?" Cả hai đều là bắt buộc; shipping kiểm tra rung cảm đã kết thúc. Những gì cần kiểm tra vào năm 2026: Kỹ thuật prompt (từ ngữ), lựa chọn model (GPT-4 so với GPT-3.5 so với OSS; accuracy so với chi phí so với độ trễ), parameters thế hệ (temperature, top-p). Các trường hợp thực tế: biến thể model phần thưởng chatbot mang lại +70% thời lượng cuộc trò chuyện và +30% tỷ lệ giữ chân; Nextdoor AI các thí nghiệm dòng chủ đề mang lại +1% CTR sau khi tinh chỉnh chức năng phần thưởng; Khan Academy Khanmigo lặp lại trên trục độ trễ so với accuracy toán học. Phân chia nền tảng: **Statsig** (được OpenAI mua lại với giá 1,1 tỷ đô la vào tháng 9 năm 2025) — thử nghiệm tuần tự, CUPED, tất cả trong một. **GrowthBook** — mã nguồn mở, gốc kho, Bayesian + Frequentist + Công cụ tuần tự, CUPED, kiểm tra SRM, hiệu chỉnh Benjamini-Hochberg + Bonferroni. Bạn chọn dựa trên sở thích SQL kho và liệu "OpenAI mua lại" có quan trọng đối với tổ chức của bạn hay không.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy sequential test simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 13 (Observability), Giai đoạn 17 · 20 (Triển khai lũy tiến)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Phân biệt các đánh giá ("model có thể thực hiện công việc không") với các bài kiểm tra A/B ("người dùng có quan tâm không").
- Liệt kê ba trục có thể kiểm tra (prompt, model, parameters) và chọn số liệu cho mỗi trục.
- Giải thích CUPED, thử nghiệm tuần tự và hiệu chỉnh so sánh nhiều Benjamini-Hochberg.
- Chọn Statsig hoặc GrowthBook dựa trên tư thế SQL kho hàng và lập trường mua lại của công ty.

## Vấn đề

Bạn đã điều chỉnh bằng tay một system prompt. Cảm giác tốt hơn. Bạn ship nó. Chuyển đổi thay đổi theo nhiễu. Bạn đổ lỗi cho số liệu. Hoặc bạn shipped một model mới và chuyển đổi không di chuyển - model có suy giảm hay thay đổi quá nhỏ để phát hiện? Bạn không biết, bởi vì bạn shipped mà không có A/B.

Đánh giá trả lời liệu model có thể thực hiện một nhiệm vụ trên một tập hợp được dán nhãn hay không. Họ không trả lời liệu người dùng có thích đầu ra hay không. Chỉ có một thí nghiệm trực tuyến có kiểm soát mới trả lời điều đó, và chỉ khi thí nghiệm có đủ sức mạnh, kiểm soát tính không quyết định và sửa chữa cho nhiều so sánh.

## Khái niệm

### Kiểm tra Evals so với A/B

**Đánh giá** — ngoại tuyến, tập hợp được dán nhãn, thẩm phán (bảng đánh giá hoặc LLM-với tư cách là thẩm phán hoặc con người). Trả lời: "Đầu ra có chính xác / hữu ích / an toàn trên bản phân phối cố định này không?"

**Kiểm tra A/B **- trực tuyến, người dùng trực tiếp, ngẫu nhiên. Trả lời: "Biến thể mới có di chuyển chỉ số cấp người dùng quan trọng không?"

Cả hai đều bắt buộc. Đánh giá bắt hồi quy trước khi tiếp xúc; A/B xác nhận tác động của sản phẩm sau đó.

### Những gì cần kiểm tra

1. **Prompt kỹ thuật **- từ ngữ, cấu trúc prompt hệ thống, ví dụ. Chỉ số: thành công của nhiệm vụ, tỷ lệ giữ chân người dùng cost/request.
2. **Model lựa chọn** — GPT-4 so với GPT-3.5-Turbo so với Llama-OSS. Chỉ số: accuracy (nhiệm vụ) + cost/request + độ trễ P99. Đa mục tiêu.
3. **Thế hệ parameters** — temperature, top-p, max_tokens. Số liệu: nhiệm vụ cụ thể (đa dạng đầu ra so với quyết định).

### CUPED - giảm variance

Thí nghiệm có kiểm soát sử dụng dữ liệu trước khi thử nghiệm. Rút lui variance trước kỳ kinh nguyệt trước khi so sánh sau kỳ kinh nguyệt. Giảm variance điển hình: 30-70%. Kích thước mẫu hiệu quả tăng lên miễn phí.

Triển khai: cả Statsig và GrowthBook đều triển khai.

### Kiểm tra tuần tự

A/B cổ điển giả định kích thước mẫu cố định. Các xét nghiệm tuần tự ("nhìn trộm và quyết định") kiểm soát tỷ lệ dương tính giả khi nhìn nhiều lần. Các thủ tục tuần tự luôn hợp lệ (mSPRT, chuỗi tin cậy của Howard) cho phép bạn dừng lại sớm với những người chiến thắng rõ ràng.

### Hiệu chỉnh nhiều so sánh

Chạy 20 xét nghiệm A/B với độ tin cậy 95% tình cờ tạo ra một kết quả dương tính giả. Hiệu chỉnh Bonferroni thắt chặt α mỗi lần kiểm tra; Benjamini-Hochberg kiểm soát tỷ lệ phát hiện sai. GrowthBook triển khai cả hai.

### SRM — tỷ lệ mẫu không khớp

Hàm băm gán ngẫu nhiên người dùng thành các biến thể. Nếu 50/50 phân tách mang lại 47/53, một cái gì đó sẽ bị hỏng - kiểm tra SRM sẽ gắn cờ nó. Cả hai nền tảng đều triển khai.

### Statsig vs GrowthBook

**Số liệu thống kê**:
- Được OpenAI mua lại với giá 1,1 tỷ đô la (Tháng 9 năm 2025). Được lưu trữ, SaaS.
- Thử nghiệm tuần tự, CUPED, quần thể được tổ chức.
- Tất cả trong một: feature cờ + thử nghiệm + observability.
- Phù hợp nhất: nhóm đã muốn có một sản phẩm đi kèm, không quan tâm đến quyền sở hữu OpenAI.

**Sổ tăng trưởng**:
- Mã nguồn mở (MIT); warehouse-native (đọc trực tiếp từ Snowflake/BigQuery/Redshift).
- Nhiều động cơ: Bayesian, Frequentist, Sequential.
- Hiệu chỉnh CUPED, SRM, Bonferroni, BH.
- Tự lưu trữ hoặc cloud được quản lý.
- Phù hợp nhất: cửa hàng SQL kho, nhóm dữ liệu kiểm soát lớp số liệu, muốn OSS.

### Thuyết không quyết định làm phức tạp quyền lực

Cùng một prompt tạo ra các kết quả đầu ra khác nhau. Các tính toán công suất truyền thống giả định các quan sát IID. Với LLM tính không xác định, kích thước mẫu hiệu quả thấp hơn danh nghĩa. Nhân kích thước mẫu yêu cầu với ~1,3-1,5 lần làm biên độ an toàn.

### Kết quả trường hợp thực tế

- Phần thưởng Chatbot model biến thể: +70% thời lượng cuộc trò chuyện, +30% tỷ lệ giữ chân.
- Dòng chủ đề Nextdoor: +1% CTR sau khi tinh chỉnh chức năng phần thưởng.
- Khan Academy Khanmigo: giao dịch độ trễ lặp đi lặp lại so với accuracy toán học.

### Phản mẫu: shipping rung cảm

Mọi kỹ sư cấp cao đều có thể kể tên một feature đã được shipped vì "cảm thấy tốt hơn" mà không có A/B. Hầu hết trong số họ đều thoái lui các chỉ số sản phẩm mà nhóm đã không nhận thấy trong nhiều tháng. A/B là hàm cưỡng bức.

### Những con số bạn nên nhớ

- Statsig được OpenAI mua lại: 1,1 tỷ đô la, tháng 9 năm 2025.
- GrowthBook: MIT mã nguồn mở; Bayes + Frequentist + Tuần tự.
- Giảm variance CUPED: 30-70%.
- LLM tính không xác định → + 30-50% dung dịch đệm cỡ mẫu.

## Ứng dụng

`code/main.py` mô phỏng thử nghiệm A/B tuần tự với ranh giới cố định và tuần tự. Cho thấy tuần tự cho phép bạn dừng lại sớm như thế nào.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-ab-plan.md`. Với sự thay đổi feature, khối lượng công việc, đường cơ sở, nền tảng lấy hàng, cổng, kích thước mẫu.

## Bài tập

1. Chạy `code/main.py`. Để có mức tăng dự kiến 5% với chuyển đổi 3% cơ bản, kích thước mẫu đến 80% công suất là bao nhiêu?
2. Chọn Statsig hoặc GrowthBook cho khách hàng tại chỗ được quản lý bởi chăm sóc sức khỏe.
3. Thiết kế một A/B kiểm tra GPT-4 so với GPT-3.5 trên chi phí cho mỗi phiếu được giải quyết. Số liệu chính, số liệu guardrail, thứ cấp là gì?
4. canary của bạn vượt qua nhưng A/B hiển thị chuyển đổi -1,2%. Bạn có ship? Viết tiêu chí leo thang.
5. Áp dụng CUPED cho giai đoạn trước với 60% variance của bài viết. Tính toán mức tăng kích thước mẫu hiệu quả.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Đánh giá | "Kiểm tra ngoại tuyến" | Đánh giá bộ nhãn về khả năng model |
| Kiểm tra A/B | "Thử nghiệm" | So sánh ngẫu nhiên trực tiếp trên người dùng |
| CỐC | "Giảm variance" | Hồi quy trước kỳ kinh nguyệt để giảm variance |
| Kiểm tra tuần tự | "Kiểm tra Peek-OK" | Quy trình luôn hợp lệ cho phép dừng sớm |
| So sánh nhiều lần | "Lỗi gia đình" | Chạy nhiều xét nghiệm làm tăng kết quả dương tính giả |
| Bonferroni | "điều chỉnh chặt chẽ" | Chia α cho số lần kiểm tra |
| Benjamini-Hochberg | "BH FDR" | Kiểm soát tỷ lệ phát hiện sai, ít bảo thủ hơn |
| SRM | "Phân chia xấu" | Tỷ lệ mẫu không khớp; Lỗi gán |
| Thống kê | "OpenAI sở hữu" | Tất cả trong một thương mại, mua lại năm 2025 |
| Sách tăng trưởng | "OSS một" | Nền tảng gốc kho MIT |
| mSPRT | "Kiểm tra tỷ lệ xác suất tuần tự" | Quy trình tuần tự cổ điển |

## Đọc thêm

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
