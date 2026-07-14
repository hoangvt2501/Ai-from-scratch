# Đánh giá và điều phối Benchmarks

> Năm benchmarks 2025-2026 bao gồm không gian đánh giá nhiều agent. **MultiAgentBench / MARBLE** (ACL 2025, arXiv:2503.01935) đánh giá cấu trúc liên kết star/chain/tree/graph với KPI quan trọng; **Đồ thị là tốt nhất cho nghiên cứu**, lập kế hoạch nhận thức thêm ~3% thành tích cột mốc. **COMMA** đánh giá sự phối hợp thông tin bất đối xứng đa phương thức; Các models hiện đại bao gồm cả GPT-4o đấu tranh để đánh bại một đường cơ sở ngẫu nhiên. **MedAgentBoard** (arXiv:2505.12371) bao gồm bốn loại nhiệm vụ y tế và thường thấy đa agent không thống trị một LLM. **AgentArch** (arXiv:2509.10769) benchmarks kiến trúc agent doanh nghiệp kết hợp sử dụng công cụ + bộ nhớ + orchestration. **SWE-bench Pro** ([arXiv:2509.16941](https://arxiv.org/abs/2509.16941)) có 1865 vấn đề trên 41 repos bao gồm các ứng dụng kinh doanh, dịch vụ B2B và công cụ dành cho nhà phát triển; Frontier models đạt điểm ~23% trên Pro so với 70%+ trên Đã xác minh — kiểm tra thực tế về ô nhiễm. Claude Opus 4.7 (tháng 4 năm 2026) được báo cáo ở mức **64,3%** trên Pro với sự phối hợp rõ ràng giữa các nhóm agent (chưa có nguồn chính Anthropic nào được công bố - coi là sơ bộ); Verdent (agent giàn giáo) đạt **76,1% pass@1** trên Verified ([Verdent technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)). **Chương trình Cầu nối AAAI 2026 WMAC **(https://multiagents.org/2026/) là đầu mối cộng đồng năm 2026. Bài học này được xây dựng dựa trên các chỉ số của MARBLE, chạy quét cấu trúc liên kết so với số liệu và ghim quy tắc "chỉ vượt qua SWE-bench Verified không phải là bằng chứng của sự khái quát hóa".

**Loại:** Học
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 15 (Cấu trúc liên kết bỏ phiếu và tranh luận), Giai đoạn 16 · 23 (Chế độ lỗi)
**Thời lượng:** ~75 phút

## Vấn đề

Khi một bài báo tuyên bố "hệ thống đa agent của chúng ta tốt hơn", câu hỏi đặt ra là: tốt hơn cái gì, dựa trên cái gì, đo lường như thế nào? Kỷ nguyên 2023-2024 của đánh giá đa agent là hỗn loạn - mọi người đều chọn số liệu của riêng họ, đường cơ sở của riêng họ và bộ nhiệm vụ của riêng họ. Cấu trúc áp đặt benchmarks 2025-2026.

Nếu không có benchmarks chung, bạn không thể so sánh hai hệ thống đa agent một cách có ý nghĩa. Tệ hơn nữa, nếu không có benchmarks giữ lại, models biên giới có thể gây ô nhiễm. SWE-bench Verified đã bị ô nhiễm một phần trong kho dữ liệu training vào giữa năm 2025; điểm số biên giới bị thổi phồng; Pro được thiết kế như một công cụ kiểm tra thực tế không bị ô nhiễm.

Bài học này liệt kê năm benchmarks kinh điển năm 2026, nêu tên những gì mỗi thước đo và dạy bạn đọc benchmark tuyên bố một cách hoài nghi.

## Khái niệm

### MultiAgentBench (MARBLE) - ACL 2025

arXiv: 2503.01935 (bằng tiếng Anh). Đánh giá bốn cấu trúc liên kết phối hợp (sao, chuỗi, cây, đồ thị) trong các nhiệm vụ nghiên cứu, mã hóa và lập kế hoạch. KPI dựa trên cột mốc theo dõi tiến độ một phần thay vì chỉ thành công cuối cùng.

Kết quả đo được:

- **Biểu đồ** cấu trúc liên kết tốt nhất cho các tình huống nghiên cứu; Hỗ trợ phê bình bất kỳ.
- **Chain** tốt nhất cho mã hóa tinh chỉnh từng bước.
- **Ngôi sao** tốt nhất để củng cố thực tế nhanh chóng.
- **Thuế phối hợp** xuất hiện qua ~4 agents trên biểu đồ.
- **Lập kế hoạch nhận thức** thêm ~3% thành tích cột mốc trên các cấu trúc liên kết.

Sử dụng khi: bạn muốn so sánh cấu trúc liên kết phối hợp với táo. MARBLE repo (https://github.com/ulab-uiuc/MARBLE) cung cấp công cụ đánh giá.

### COMMA — thông tin bất đối xứng đa phương thức

Bao gồm các nhiệm vụ mà agents có các phương thức quan sát khác nhau và phải phối hợp mà không chia sẻ thông tin đầy đủ. Kết quả được báo cáo là không thoải mái: các models biên giới bao gồm cả GPT-4o phải vật lộn để đánh bại **đường cơ sở ngẫu nhiên** về sự hợp tác agent-agent trong COMMA. Tín hiệu là các phương thức đa agent chưa được huấn luyện và đánh giá thấp - LLMs xử lý hợp tác một phương thức một cách hợp lý; sự phối hợp đa phương thức sụp đổ.

Sử dụng khi: hệ thống của bạn có sự phối hợp thông tin đa phương thức hoặc không đối xứng. Kết quả rỗng từ COMMA là một cảnh báo cần đo lường trước khi xác nhận quyền sở hữu.

### MedAgentBoard — kiểm tra căng thẳng miền

arXiv: 2505.12371 (bằng tiếng Anh). Bốn loại nhiệm vụ y tế: chẩn đoán, lập kế hoạch điều trị, tạo báo cáo, giao tiếp với bệnh nhân. So sánh các hệ thống dựa trên quy tắc nhiều agent so với một LLM so với các hệ thống dựa trên quy tắc thông thường.

Phát hiện: đa agent KHÔNG thống trị một LLM trên hầu hết các danh mục. Lợi thế đa agent là hẹp - phân tách nhiệm vụ giúp ích khi các nhiệm vụ phụ có thể tách rời rõ ràng (chẩn đoán + điều trị); Nó gây tổn thương khi chi phí phối hợp vượt quá mức tăng chuyên môn (tạo báo cáo).

Sử dụng khi: miền của bạn có đường cơ sở một LLM rõ ràng. Nếu bài học của MedAgentBoard khái quát hóa, nhiều hệ thống đa agent được đề xuất được thiết kế quá mức.

### AgentArch — kiến trúc doanh nghiệp

arXiv: 2509.10769 (bằng tiếng Anh). Cài đặt doanh nghiệp với việc sử dụng công cụ, bộ nhớ và orchestration được xếp lớp với nhau. Benchmark cô lập sự đóng góp của từng lớp: thêm công cụ giúp ích bao nhiêu? Thêm bộ nhớ? Thêm nhiều agent orchestration?

Sử dụng khi: bạn đang thiết kế một agent stack doanh nghiệp và cần biện minh cho từng lớp. AgentArch giúp tránh mua features bạn không thể đo lường giá trị.

### SWE-bench Pro — kiểm tra thực tế

arXiv: 2509.16941 (bằng tiếng Anh). 1865 vấn đề trên 41 repositories bao gồm các ứng dụng kinh doanh, dịch vụ B2B và công cụ dành cho nhà phát triển. Được thiết kế để **không bị ô nhiễm **với các vết cắt training sau đó. Frontier models điểm ~23% trên Pro so với 70% + trên Đã xác minh. Khoảng trống là tín hiệu ô nhiễm.

Điểm số tháng 4 năm 2026:
- Claude Opus 4.7 trên Pro: **64,3%** (được báo cáo với sự phối hợp rõ ràng giữa các nhóm agent; chưa có nguồn chính Anthropic nào được công bố - coi là sơ bộ).
- Verdent (agent giàn giáo) trên Đã xác minh: **76.1% pass@1** ([technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)).
- Điểm thô Frontier trên Pro không có giàn giáo agent: ~23-35% ([SWE-bench Pro paper](https://arxiv.org/abs/2509.16941)).

Bài học rút ra: "chúng tôi đã đánh bại SWE-bench Verified" không còn là bằng chứng về khả năng. Pro là kiểm tra cổng hiện tại. Giàn giáo nhóm Agent tạo ra lợi ích có thể đo lường được trên Pro (~30-40 điểm delta), đây là một trong những lập luận thực nghiệm mạnh nhất cho sự phối hợp đa agent vào năm 2026.

### AAAI 2026 WMAC

Chương trình Cầu nối AAAI 2026 - Hội thảo về Điều phối đa Agent (https://multiagents.org/2026/). Đầu mối cộng đồng năm 2026 cho nghiên cứu đa agent AI. Các bài báo được chấp nhận và kỷ yếu hội thảo là địa điểm chuẩn mực để đánh giá các phương pháp mới; trì hoãn các tuyên bố được WMAC chấp nhận đối với các bản in trước arXiv cho các quyết định production.

### Đọc benchmark tuyên bố một cách hoài nghi — danh sách kiểm tra năm 2026

Khi ai đó xác nhận quyền sở hữu kết quả nhiều agent:

1. **benchmark nào, phân chia nào?** SWE-bench Verified vs Pro rất quan trọng. Một con số được báo cáo về việc phân chia sai là vô giá trị.
2. **Kiểm tra ô nhiễm.** benchmark có được phát hành sau khi giới hạn training của model không? Nếu không, hãy xử lý một cách thận trọng.
3. **So sánh đường cơ sở.** Vs đường cơ sở một LLM, so với ngẫu nhiên, so với công việc nhiều agent prior. Không phải "so với phiên bản chưa được điều chỉnh của cùng một hệ thống".
4. **Ý nghĩa thống kê.** N thử nghiệm, giá trị p, khoảng tin cậy. models biên giới có variance cao; chạy đơn gây hiểu lầm.
5. **Đa dạng nhiệm vụ.** Một nhiệm vụ hay nhiều? Khái quát hóa quan trọng đối với production.
6. **Tiết lộ chi phí.** Tokens cho mỗi tác vụ, đồng hồ treo tường. Một giải pháp 90% với chi phí gấp 20 lần là một quyết định kinh doanh, không phải là một tuyên bố về năng lực.

### Những gì không có benchmarks đo lường tốt

- **Phối hợp đường chân trời dài.** Những ngày tương tác với đồng hồ treo tường. Tất cả các benchmarks hiện tại đều cạn kiệt.
- **Khả năng phục hồi đối nghịch.** Điều gì xảy ra khi một agent độc hại hoặc bị xâm phạm?
- **Trôi dạt khi triển khai.** Benchmarks là tĩnh; production phân phối thay đổi.
- **Hiệu suất chuẩn hóa chi phí.** Hầu hết các benchmarks báo cáo accuracy thô, không phải accuracy trên mỗi đô la.

Xây dựng benchmark nội bộ của riêng bạn cho trục mà bạn thực sự quan tâm thường là bước đi đúng đắn.

## Tự xây dựng

`code/main.py` là hướng dẫn không tương tác:

- Mô phỏng 3 hệ thống đa agent trên một nhiệm vụ đồ chơi.
- Tính toán các chỉ số cột mốc kiểu MARBLE cho từng chỉ số.
- Chạy kiểm tra ô nhiễm bằng cách giữ lại các tác vụ từ một nhóm "training".
- So sánh với đường cơ sở ngẫu nhiên một cách rõ ràng.
- In thẻ điểm yêu cầu benchmark.

Chạy:

```bash
python3 code/main.py
```

Đầu ra dự kiến: thẻ điểm hệ thống với accuracy thô, thành tích cột mốc, chi phí cho mỗi nhiệm vụ, so với delta cơ sở ngẫu nhiên và ghi chú kiểm tra ô nhiễm.

## Ứng dụng

`outputs/skill-benchmark-reader.md` đọc mọi tuyên bố nhiều agent benchmark và áp dụng danh sách kiểm tra xem xét kỹ lưỡng. Đầu ra: một điểm và cảnh báo.

## Sản phẩm bàn giao

Kỷ luật đánh giá Production:

- **Xây dựng một benchmark** nội bộ phản ánh phân phối production thực tế của bạn. Công benchmarks thông báo nhưng không thay thế.
- **Bao gồm một đường cơ sở ngẫu nhiên** trong mọi so sánh. Nếu bạn không thể đánh bại ngẫu nhiên với tỷ lệ chênh lệch lớn trong một nhiệm vụ phối hợp, nhiệm vụ có thể bị định vị xấu.
- **Báo cáo chi phí cùng với accuracy.** chi phí Token và đồng hồ treo tường. Các đội hoạt động cần cả hai.
- **Xây dựng lại benchmark hàng quý.** Production ca phân phối; cũ benchmarks gây hiểu lầm.
- **Tránh benchmark overfitting xuất bản.** Nếu nhóm của bạn đang tối ưu hóa cụ thể cho các số SWE-bench Pro, bạn sẽ thoái lui trên production.

## Bài tập

1. Chạy `code/main.py`. Xác định hệ thống nào trong số ba hệ thống mô phỏng có chi phí cho mỗi cột mốc tốt nhất. Nó có phù hợp với hệ thống accuracy thô cao nhất không?
2. Đọc MultiAgentBench (arXiv: 2503.01935). Đối với miền nhiệm vụ của riêng bạn, hãy quyết định cấu trúc liên kết nào trong số bốn cấu trúc liên kết mà MARBLE sẽ đề xuất. Biện minh từ kết quả của bài báo.
3. Đọc bài SWE-bench Pro. Điều gì cụ thể làm cho nó có khả năng chống nhiễm bẩn? Kỹ thuật tương tự có thể được áp dụng cho các benchmarks khác mà bạn quan tâm không?
4. Đọc phát hiện của COMMA về phối hợp đa phương thức. Thiết kế một nhiệm vụ phối hợp đa phương thức đơn giản mà bạn có thể thêm vào benchmark nội bộ của mình. Điều gì sẽ được tính là một tín hiệu hữu ích?
5. Áp dụng danh sách kiểm tra yêu cầu benchmark cho kết quả tiêu đề của một bài báo nhiều agent gần đây. Bạn sẽ cho điểm nào để yêu cầu?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| ĐÁ CẨM THẠCH | "MultiAgentBench" | ACL 2025; star/chain/tree/graph cấu trúc liên kết với KPI quan trọng. |
| DẤU PHẨY | "benchmark đa phương thức" | Phối hợp thông tin bất đối xứng đa phương thức; Biên giới models đấu tranh vs ngẫu nhiên. |
| Bảng đại lý y tế | "Kiểm tra căng thẳng miền" | Bốn loại y tế; thường thấy đa agent không thống trị một LLM. |
| Đại lý Arch | "Doanh nghiệp benchmark" | Công cụ + bộ nhớ + orchestration lớp. |
| SWE-bench Pro | "Chống ô nhiễm" | 1865 vấn đề, 41 repos; ~23% so với 70%+ trên Đã xác minh (tín hiệu ô nhiễm). |
| Thành tích cột mốc | "Tín dụng một phần" | Benchmarks đó thưởng cho sự tiến bộ, không chỉ thành công cuối cùng. |
| Ô nhiễm | "Benchmark bị rò rỉ vào training" | Sau khi phát hành, benchmarks trôi dạt vào training thể dữ liệu; điểm số thổi phồng. |
| WMAC | "Chương trình cầu nối AAAI 2026" | Hội thảo về Điều phối đa Agent; đầu mối cộng đồng. |

## Đọc thêm

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — benchmark cấu trúc liên kết với KPI quan trọng
- [MARBLE repository](https://github.com/ulab-uiuc/MARBLE) — triển khai tham chiếu
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — kiểm tra căng thẳng miền; đa agent thường không chiếm ưu thế
- [AgentArch](https://arxiv.org/abs/2509.10769) — kiến trúc agent doanh nghiệp
- [SWE-bench leaderboards](https://www.swebench.com/) — Điểm đã xác minh và điểm chuyên nghiệp cho models biên giới
- [AAAI 2026 WMAC](https://multiagents.org/2026/) - đầu mối cộng đồng năm 2026
