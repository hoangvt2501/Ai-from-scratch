# STaR, V-STaR, Quiet-STaR — Tự học lý luận

> Vòng lặp tự cải thiện nhỏ nhất có thể nằm bên trong cơ sở lý luận. Một model tạo ra một chuỗi suy nghĩ, giữ những câu trả lời đúng và tinh chỉnh chúng. Đó là STaR. V-STaR bổ sung một trình xác minh để lựa chọn inference thời gian sẽ tốt hơn. Quiet-STaR đẩy lý do xuống mọi token. Cả ba đều hoạt động. Không ai trong số chúng là ma thuật - vòng lặp bảo tồn bất kỳ phím tắt nào đã xảy ra để đạt được câu trả lời đúng.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, bootstrap-loop simulator)
**Kiến thức tiên quyết:** Giai đoạn 13 · 01-03 (Lý luận và CoT), Giai đoạn 15 · 01 (khung hình đường chân trời dài)
**Thời lượng:** ~60 phút

## Vấn đề

Cách đơn giản để dạy một model lý luận là thu thập traces lý luận do con người viết. Điều đó đắt đỏ, chậm chạp và bị giới hạn bởi lượng chain-of-thought chất lượng cao mà con người sẵn sàng viết.

STaR (Người lý luận tự học, Zelikman và cộng sự, 2022) đặt câu hỏi: điều gì sẽ xảy ra nếu model viết lý do của riêng mình và chấm điểm chúng dựa trên các câu trả lời đã biết? Vòng lặp là:

1. Mẫu một câu trả lời cộng trace lý luận.
2. Nếu câu trả lời cuối cùng là đúng, hãy giữ trace.
3. Fine-tune trên traces được giữ.
4. Lặp lại.

Nó hoạt động. GSM8K và CommonsenseQA đều được cải thiện mà không cần chú thích con người mới. Nhưng vòng lặp có một bias tích hợp: bất kỳ lý do nào tạo ra câu trả lời đúng đều được giữ lại, bất kể bản thân lý luận đó có hợp lý hay không. V-STaR (Hosseini và cộng sự, 2024) vá lỗi này bằng một trình xác minh đã học; Quiet-STaR (Zelikman và cộng sự, 2024) khái quát hóa ý tưởng thành các lý do bên trong mỗi token.

## Khái niệm

### STaR: bootstrap về những gì hoạt động

Bắt đầu từ một model cơ bản với một số khả năng suy luận yếu. Đối với mỗi vấn đề training, hãy lấy mẫu một lý do cộng với câu trả lời. Nếu câu trả lời khớp với nhãn, hãy giữ ba (vấn đề, lý do, câu trả lời). Fine-tune model trên bộ đã giữ. Lặp lại.

Một bước ngoặt quan trọng. Nếu model không bao giờ có thể giải quyết vấn đề đúng, vòng lặp không thể học hỏi về nó. STaR thêm **hợp lý hóa**: đối với các vấn đề mà model thất bại, hãy đưa câu trả lời đúng làm gợi ý và prompt lại model để đưa ra lý do dẫn đến nó. Các lý do hợp lý được thêm vào tập training.

Kết quả trong bài báo gốc (Zelikman và cộng sự, 2022): cơ sở GPT-J model cải thiện trên GSM8K từ 5,8% lên 10,7% thông qua các vòng STaR lặp đi lặp lại với sự hợp lý hóa - khoảng 5 điểm phần trăm tuyệt đối. Trên CommonsenseQA, GPT-J 6B được huấn luyện STaR đạt 72,5%, tương đương với fine-tuned GPT-3 175B (~73%) - một model lớn hơn khoảng 30 lần được huấn luyện về các lý do được chú thích bằng tay.

### V-STaR: huấn luyện người xác minh với DPO

STaR vứt bỏ những lý do không chính xác. Hosseini et al. (2024) quan sát thấy đó cũng là dữ liệu: mỗi cặp (lý do, "điều này có đúng không") đều có thể huấn luyện một người xác minh. Họ sử dụng Tối ưu hóa tùy chọn trực tiếp cho cả giải pháp đúng và sai để xây dựng trình xếp hạng. Tại thời điểm inference, hãy lấy mẫu N lý do và chọn lựa chọn hàng đầu của người xác minh.

Delta được báo cáo: +4 đến +17 điểm phần trăm so với prior đường cơ sở tự cải thiện trên GSM8K và MATH, với phần lớn lợi ích đến từ việc sử dụng trình xác minh để lựa chọn inference thời gian thay vì fine-tuning máy phát bổ sung.

### Quiet-STaR: lý do nội bộ cho mỗi token

Zelikman et al. (2024) đặt câu hỏi: điều gì sẽ xảy ra nếu model học cách tạo ra một cơ sở lý luận nội bộ ngắn ở mọi vị trí token, không chỉ giữa vấn đề và câu trả lời? Quiet-STaR huấn luyện một model phát ra một "suy nghĩ" ẩn trước mỗi token dự đoán, sau đó kết hợp dự đoán nhận thức về suy nghĩ với dự đoán cơ bản thông qua trọng lượng đã học.

Kết quả: Mistral 7B đạt được những cải tiến zero-shot tuyệt đối trên GSM8K từ 5,9% lên 10,9% và CommonsenseQA từ 36,3% lên 47,2% mà không có fine-tuning cụ thể theo nhiệm vụ. Các model học được "khi nào nên suy nghĩ" - khó khăn tokens có được những lý do bên trong dài hơn; những cái dễ dàng hầu như không có.

### Tại sao cả ba đều có chung mối quan tâm về an toàn

Cả ba phương pháp đều sử dụng câu trả lời cuối cùng làm tín hiệu gradient. Một cơ sở lý luận đi đến câu trả lời đúng thông qua lý luận sai lầm - khai thác lối tắt, đoán hoặc sử dụng một mô hình không khái quát hóa - được củng cố tích cực. Đối với các vấn đề trong phân phối, phím tắt hoạt động. Về các vấn đề ngoài phân phối, nó lặng lẽ phá vỡ.

Trình xác minh của V-STaR giảm thiểu bằng cách học cách xếp hạng các lý do, nhưng trình xác minh được huấn luyện trên cùng một bộ nhãn. Nó có thể học cách thích lý luận sai được định dạng tốt hơn là sự không chắc chắn trung thực. Thiết kế an toàn hơn là kết hợp dữ liệu kiểu STaR với (a) models phần thưởng được giám sát process (thưởng cho các bước trung gian, không chỉ câu trả lời) và (b) đánh giá OOD được tổ chức để phá vỡ các phím tắt đơn giản.

### So sánh

| Phương pháp | Tín hiệu Training | Chi phí Inference | Lãng phí dữ liệu | Chế độ lỗi đã biết |
|---|---|---|---|---|
| STaR | giữ (lý do, trả lời) nếu đúng | 1 lần | Loại bỏ tất cả các lý do không chính xác | Lý do phím tắt |
| STaR + hợp lý hóa | ở trên + câu trả lời đúng được gợi ý thử lại | 1 lần | ít | Các lý do hợp lý hóa có thể không thể tin được |
| V-STaR | Trình xác minh STaR + DPO từ cả hai classes | Nx (tốt nhất của N) | Tối thiểu | Verifier có thể củng cố sự sai lầm tự tin |
| Yên tĩnh-STaR | lý do trên mỗi token + trọng lượng trộn | 1,5-3 lần | Tối thiểu | vẫn có điều kiện trả lời gradient |

### Điều này nằm ở đâu vào năm 2026 stack

STaR đã cũ. Nhưng mô hình này xuất hiện trở lại ở khắp mọi nơi vào năm 2025-2026. RL về các bài toán có thể kiểm chứng (DeepSeek-R1, Kimi-k1.5, o1) là tín hiệu gradient có điều kiện trả lời của STaR, được mở rộng quy mô. Process phần thưởng models (Lightman và cộng sự, 2023; "Hãy xác minh từng bước" của OpenAI) là giải pháp thay thế được giám sát process. AlphaEvolve (Bài 3) là STaR cho mã, với một trình đánh giá chương trình thay vì một nhãn. Darwin Godel Machine (Bài 4) là STaR cho chính giàn giáo agent.

Hiểu STaR làm cho tất cả những nhấp chuột này. Đó là vòng lặp tự cải thiện khả thi tối thiểu.

```figure
reflection-loop
```

## Ứng dụng

`code/main.py` chạy một vòng lặp STaR mô phỏng trên một nhiệm vụ số học đồ chơi. Bạn có thể xem:

- Làm thế nào accuracy leo qua các vòng bootstrap.
- Cách các phím tắt lẻn vào: trình mô phỏng bao gồm một class lý do "lười biếng" nhận được câu trả lời đúng 40% thời gian nhưng khái quát hóa tệ. Xem liệu STaR có giữ chúng hay không.
- Trình xác minh (kiểu V-STaR) giúp inference như thế nào nhưng không thể cắt tỉa hoàn toàn các phím tắt được giới thiệu trong training.

## Sản phẩm bàn giao

`outputs/skill-star-loop-reviewer.md` giúp bạn kiểm tra một pipeline lý luận tự học được đề xuất trước khi bạn huấn luyện về nó.

## Bài tập

1. Chạy trình mô phỏng. Đặt tần số phím tắt thành không, sau đó thành 0.4. accuracy cuối cùng chênh lệch bao nhiêu giữa hai lần chạy, mặc dù cả hai đều đạt >90% trên phân phối training?

2. Thêm thử nghiệm OOD tạm dừng vào trình mô phỏng. Rút ra các vấn đề từ một bản phân phối khác và đánh giá model khởi động trên cả bộ phân phối và OOD. Định lượng khoảng cách.

3. Đọc bài báo Quiet-STaR (arXiv: 2403.09629) Phần 3. Giải thích token "kết thúc suy nghĩ" và đầu trọng lượng hỗn hợp trong ba câu mỗi câu.

4. So sánh bộ lọc giữ nếu đúng của STaR với một giải pháp thay thế được giám sát process để thưởng cho từng bước lý do một cách độc lập. Xác định sự khác biệt về chi phí ghi nhãn và sự khác biệt về chất lượng hợp lý.

5. Thiết kế một đánh giá sẽ nắm bắt các lý do tắt trong một model đã triển khai. Nó không cần phải hoàn hảo - nó phải phá vỡ các phím tắt đơn giản nhất mà vòng lặp STaR sẽ củng cố.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| STaR | "Người lý luận tự học" | Fine-tune dựa trên các lý do do model tạo ra để đưa ra câu trả lời đúng; Lặp lại |
| Hợp lý hóa | "Thử lại gợi ý" | Đưa ra câu trả lời đúng và prompt lại lý do về các vấn đề mà cơ sở model không đạt |
| V-STaR | "Trình xác minh STaR" | DPO huấn luyện người xác minh về cả lý do đúng và sai, hãy sử dụng nó để lựa chọn inference lần |
| Yên tĩnh-STaR | "Lý do cho mỗi token" | Tạo ra những suy nghĩ ẩn giấu ở mọi vị trí token; Kết hợp với dự đoán cơ sở |
| Câu trả lời có điều kiện gradient | "Tín hiệu dựa trên kết quả" | Vòng lặp training thưởng cho câu trả lời cuối cùng, không phải các bước suy luận |
| Process phần thưởng model | "Trình xác minh cấp bước" | Phần thưởng model được huấn luyện về tính chính xác của từng bước, không phải kết quả - trái ngược với STaR |
| Lý do phím tắt | "Câu trả lời đúng, lý luận sai" | Một cơ sở lý luận đạt được nhãn thông qua một mô hình không khái quát hóa; STaR giữ những thứ này |

## Đọc thêm

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — bài báo gốc.
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) - thêm trình xác minh DPO để lựa chọn inference lần.
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — lý do nội bộ mỗi token.
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - process phần thưởng models, tín hiệu gradient thay thế.
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) - RL về các nhiệm vụ có thể xác minh, STaR đã mở rộng quy mô sang training biên giới.
