# Bias và tác hại đại diện trong LLMs

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed (Ngôn ngữ học tính toán 2024, arXiv: 2309.00770). Khảo sát cơ bản năm 2024 phân biệt tác hại đại diện (rập khuôn, xóa) với tác hại phân bổ (phân phối tài nguyên không đồng đều) và phân loại các chỉ số đánh giá là dựa trên embedding, dựa trên xác suất hoặc dựa trên văn bản được tạo. 2024-2025 thực nghiệm: An et al. (PNAS Nexus, tháng 3 năm 2025) đo lường giới tính x chủng tộc giao thoa bias trên GPT-3.5 Turbo, GPT-4o, Gemini 1.5 Flash, Claude 3.5 Sonnet Llama 3-70B về đánh giá sơ yếu lý lịch tự động cho 20 công việc mới bắt đầu. WinoIdentity (COLM 2025, arXiv: 2508.07111) giới thiệu đánh giá công bằng dựa trên độ không chắc chắn cho các danh tính giao thoa. Yu & Ananiadou 2025 xác định các tế bào thần kinh giới tính trong các lớp MLP; Ahsan & Wallace 2025 sử dụng SAE để tiết lộ bias chủng tộc lâm sàng; Zhou et al. 2024 (UniBias) thao túng attention đầu để phân kiến. Phê bình tổng hợp (arXiv: 2508.11067): tài liệu 10 năm tập trung không cân xứng vào bias giới tính nhị phân.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, toy embedding-based bias probe)
**Kiến thức tiên quyết:** Giai đoạn 05 (từ embeddings), Giai đoạn 18 · 01 (hướng dẫn sau)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Xác định tác hại đại diện và phân bổ và đưa ra một ví dụ về từng tác hại trong triển khai LLM.
- Kể tên ba danh mục số liệu đánh giá từ Gallegos et al. 2024 và mô tả một số liệu từ mỗi loại.
- Mô tả tính giao thoa và lý do tại sao phép đo công bằng dựa trên độ không đảm bảo của WinoIdentity giải quyết các lỗ hổng trong đánh giá bias một trục.
- Mô tả hai cách tiếp cận khả năng giải thích cơ học đối với bias (tế bào thần kinh giới tính, SAE features, thao tác đầu attention).

## Vấn đề

Các bài học trước bao gồm tác hại có chủ ý (bẻ khóa, âm mưu) và quản trị an toàn. Bias là tác hại xuất hiện mà không có ý định - từ training phân phối dữ liệu, từ prompt đóng khung, từ các lựa chọn thiết kế tích lũy. Đo lường và giảm thiểu nó là một thách thức phương pháp luận khác biệt so với độ bền của đối thủ.

## Khái niệm

### Đại diện so với phân bổ

- **Tác hại đại diện.** Định kiến, xóa bỏ, miêu tả hạ thấp phẩm giá. Một LLM mô tả y tá chỉ là phụ nữ đang tạo ra tác hại đại diện.
- **Tác hại phân bổ.** Kết quả vật chất không bình đẳng. Một LLM chấm điểm sơ yếu lý lịch của ứng viên Da đen thấp hơn một cách có hệ thống đang tạo ra tác hại phân bổ.

Những điều này không giống nhau. Một model có thể "không thiên vị về mặt đại diện" (tạo ra các mô tả đa dạng) trong khi "thiên vị về mặt phân bổ" (đưa ra các khuyến nghị không bình đẳng). Đánh giá cần đo lường cả hai.

### Ba loại đánh giá-chỉ số (Gallegos et al. 2024)

- **Dựa trên Embedding.** Các bài kiểm tra kiểu WEAT trên RLHF embeddings trước. Đo lường mối liên hệ thống kê giữa các thuật ngữ nhận dạng và các thuật ngữ thuộc tính. Giới hạn: đo lường sự đại diện, không phải hành vi.
- **Dựa trên xác suất.** likelihood nhật ký của các lần hoàn thành xác nhận khuôn mẫu so với vi phạm khuôn mẫu. Đo Decoder bên. Nắm bắt một số bias hành vi.
- **Dựa trên văn bản được tạo.** Đo lường tác vụ xuôi dòng trên văn bản được tạo. Chấm điểm sơ yếu lý lịch, viết giới thiệu, đối thoại. Có giá trị sinh thái nhất; khó sinh sản nhất.

### Tính giao thoa

Bias đánh giá về "giới tính" bỏ lỡ bias chỉ bắn vào các cặp (giới tính, chủng tộc). Một et al. 2025 phát hiện GPT-4o phạt phụ nữ Da đen trong sơ yếu lý lịch ghi điểm nhiều hơn đàn ông Da đen và nhiều hơn phụ nữ da trắng một cách riêng biệt. Đánh giá một trục không thể nắm bắt được điều này.

WinoIdentity (COLM 2025) giới thiệu tính công bằng giao thoa dựa trên sự không chắc chắn. Nó đo lường liệu sự không chắc chắn của model về kết quả có khác nhau giữa các bộ nhận dạng giao nhau hay không - không chỉ dự đoán điểm. Điều này nắm bắt các trường hợp model sai như nhau giữa các nhóm nhưng không chắc chắn hơn đối với một số nhóm, điều này tạo ra hành vi phân bổ hạ nguồn khác nhau.

### Phương pháp tiếp cận cơ học

Công việc giải thích 2024-2025 mở ra bias cho can thiệp cơ học:

- **Tế bào thần kinh giới tính (Yu & Ananiadou 2025).** Các tế bào thần kinh MLP cụ thể tương quan với các hành vi cụ thể về giới tính. Loại bỏ các tế bào thần kinh này làm giảm các chỉ số khoảng cách giới tính với chi phí khả năng hạn chế.
- **bias chủng tộc lâm sàng thông qua SAE (Ahsan & Wallace 2025).** Bộ mã hóa tự động thưa thớt features phân tách biểu diễn bên trong thành các kích thước có thể giải thích được; features tương quan chủng tộc có thể được xác định và ngăn chặn.
- **UniBias (Zhou et al. 2024).** Thao túng đầu Attention để phân tán zero-shot. Những cái đầu cụ thể khuếch đại sự nhạy cảm class danh tính; Việc giảm hoặc tái trọng lượng các đầu này sẽ làm giảm bias mà không cần fine-tuning.

### Phê bình meta

Đánh giá tài liệu 10 năm (arXiv: 2508.11067, 2025) cho thấy lĩnh vực này tập trung không cân xứng vào bias giới tính nhị phân. Các trục khác - khuyết tật, tôn giáo, tình trạng di cư, bản sắc đa ngôn ngữ - nhận được ít attention hơn nhiều. Phê bình tổng hợp lập luận rằng sự tập trung hẹp có thể gây hại cho các nhóm bị thiệt thòi do bỏ bê: một model thiên vị về giới tính nhị phân có thể thiên vị nghiêm trọng về các khía cạnh mà không ai kiểm tra.

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài 20-21 bao gồm bias và công bằng một cách chính thức. Bài 22 đề cập đến quyền riêng tư. Bài 23 bao gồm hình mờ. Đây là lớp gây hại cho người dùng bổ sung cho lớp deception/safety trước đó.

## Ứng dụng

`code/main.py` xây dựng một đầu dò bias dựa trên embedding đồ chơi: đo khoảng cách kiểu WEAT giữa các thuật ngữ nhận dạng và các thuật ngữ thuộc tính trong một embedding đồng xuất hiện đơn giản. Bạn có thể tiêm một bias và quan sát đám cháy hệ mét; Áp dụng một thao tác khử lệch đơn giản và quan sát phục hồi một phần.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-bias-eval.md`. Với một thẻ model hoặc yêu cầu công bằng, nó kiểm tra đánh giá trên ba loại chỉ số (embedding, xác suất, văn bản được tạo), phạm vi giao thoa và cơ chế của bất kỳ can thiệp thiên vị nào.

## Bài tập

1. Chạy `code/main.py`. Báo cáo điểm bias kiểu WEAT trước và sau bước phân cực. Giải thích lý do tại sao chỉ số không giảm xuống bằng không.

2. Mở rộng đầu dò bằng một bài kiểm tra giao thoa: (giới tính, chủng tộc) x (sự nghiệp, gia đình). Báo cáo điểm số bias chéo trục.

3. Đọc An et al. 2025 (PNAS Nexus). Xác định hai hiệu ứng giao thoa mà họ báo cáo mà đánh giá giới tính một trục sẽ bỏ sót.

4. Yu & Ananiadou 2025 xác định các tế bào thần kinh giới tính. Phác thảo một thí nghiệm giả mạo để phân biệt "những tế bào thần kinh này gây ra bias giới tính" với "những tế bào thần kinh này tương quan với bias giới tính".

5. Phê bình meta lập luận rằng lĩnh vực này tập trung quá hẹp vào giới tính nhị phân. Chọn một trục chưa được nghiên cứu và mô tả một giao thức đo lường tác hại đại diện cho nó.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Tác hại đại diện | "Định kiến / Xóa" | Mô tả thiên vị về một nhóm |
| Tác hại phân bổ | "quyết định bất bình đẳng" | Kết quả vật chất thiên vị cho một nhóm |
| NƯỚC CHẢY | "Bài kiểm tra embedding" | Kiểm tra liên kết Embedding từ; Đầu dò bias dựa trên đồng xuất hiện |
| Tính giao thoa | "Hiệu ứng nhận dạng kết hợp" | Bias xuất hiện ở giao điểm của nhiều trục nhận dạng |
| Tế bào thần kinh giới tính | "Tế bào thần kinh bias MLP" | Các tế bào thần kinh cụ thể có kích hoạt tương quan với hành vi giới tính cụ thể |
| SAE feature | "Kích thước có thể giải thích" | feature nhận dạng tự động mã hóa thưa thớt; Hữu ích cho phân tích bias cơ học |
| UniBias | "Sai lệch attention đầu" | Zero-shot phân kiến bằng cách cân nhắc lại đầu attention |

## Đọc thêm

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — Khảo sát kinh điển
- [An et al. — Intersectional resume-evaluation bias (PNAS Nexus, March 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) - nghiên cứu giao thoa năm model
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) — benchmark mới
- [UniBias — attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612) - zero-shot phân kiến
