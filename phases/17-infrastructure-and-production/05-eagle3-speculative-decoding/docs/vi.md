# Giải mã suy đoán EAGLE-3 trong Production

> Giải mã đầu cơ ghép nối model nháp nhanh với model mục tiêu. Dự thảo đề xuất K tokens; mục tiêu xác minh trong một chuyển tiếp duy nhất; được chấp nhận tokens là miễn phí. Vào năm 2026, EAGLE-3 là biến thể cấp production - nó huấn luyện một đầu dự thảo trên các trạng thái ẩn của model mục tiêu thay vì trên các tokens thô, đẩy tỷ lệ chấp nhận alpha vào dải 0,6-0,8 trên trò chuyện chung. Câu hỏi đúng không phải là "bản nháp nhanh như thế nào" mà là "alpha trên lưu lượng truy cập của tôi là gì?" Nếu alpha giảm xuống dưới ~0,55, giải mã đầu cơ là âm ròng ở đồng thời cao vì mỗi bản nháp bị từ chối đều có giá forward pass mục tiêu thứ hai. Bài học này dạy bạn đo alpha trước và lật cờ thứ hai.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy acceptance-rate simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 04 (vLLM Phục vụ Nội bộ), Giai đoạn 10 · 18 (Dự đoán nhiều Token)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Kể tên ba thế hệ giải mã đầu cơ và giải thích những thay đổi của EAGLE-3 so với EAGLE-2 và từ một model dự thảo cổ điển.
- Xác định alpha tỷ lệ chấp nhận, tính toán tốc độ dự kiến từ alpha và K (độ dài bản nháp) và xác định alpha hòa vốn cho đồng thời mục tiêu của bạn.
- Giải thích lý do tại sao giải mã suy đoán là chọn tham gia (không mặc định) trong vLLM 2026 và tại sao bật nó mà không đo alpha là một phản mẫu production.
- Viết kế hoạch đo lường: benchmark nào, phân phối prompt nào, điểm đồng thời nào, chỉ số nào cần cổng.

## Vấn đề

Giải mã bị ràng buộc với bộ nhớ. Trên H100 chạy Llama 3.3 70B FP8, mỗi token được giải mã đọc ~140 GB/s trọng lượng và phát ra một token. Điện toán GPU gần như không hoạt động trong quá trình giải mã - nút cổ chai là băng thông HBM, không phải thông lượng matmul.

Giải mã suy đoán khai thác khoảng trống. Tạo tokens ứng viên K với model nháp giá rẻ, sau đó yêu cầu model mục tiêu xác minh tất cả K trong một forward pass. Mỗi token đã được xác minh đều miễn phí một cách hiệu quả (được khấu hao thành chuyển tiếp batch-of-K mà mục tiêu sẽ phải làm).

Cách tiếp cận model dự thảo cổ điển sử dụng một model nhỏ hơn của cùng một họ (Llama 3.2 1B soạn thảo cho Llama 3.3 70B). Nó hoạt động nhưng tỷ lệ chấp nhận là tầm thường - phân phối model nhỏ hơn khác với mục tiêu. EAGLE, sau đó là EAGLE-2, sau đó là EAGLE-3 huấn luyện một đầu gió lùa nhẹ trực tiếp vào trạng thái bên trong của model mục tiêu, vì vậy sự phân bố của mớn nước theo dõi mục tiêu chặt chẽ hơn nhiều. Đó là lý do tại sao alpha đi từ 0,4 với model nháp lên 0,6-0,8 với EAGLE-3.

Điểm mấu chốt: EAGLE-3 đang chọn tham gia vLLM 2026. `speculative_config` phải được đặt rõ ràng. Không có cờ, không tăng tốc. Các nhóm bật nó mà không đo lường alpha trên lưu lượng truy cập thực của họ thường thấy độ trễ đuôi trở nên tồi tệ hơn chứ không phải tốt hơn.

## Khái niệm

### Giải mã đầu cơ thực sự mua gì

Nếu không có giải mã thông số kỹ thuật, chi phí mỗi token là một mục tiêu chuyển tiếp. Với giải mã thông số kỹ thuật ở độ dài nháp K và alpha chấp nhận, tokens dự kiến cho mỗi mục tiêu chuyển tiếp là `1 + K * alpha`. Việc tăng tốc là `(1 + K * alpha) / (1 + epsilon)` khi epsilon là dự thảo cộng với xác minh chi phí. Đối với K = 5, alpha = 0,7: `(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`. Các con số trong thế giới thực tập trung khoảng 2-3x vì alpha hiếm khi có lưu lượng truy cập production cao và epsilon phát triển ở kích thước batch cao.

### Tại sao alpha là chỉ số duy nhất quan trọng

Các tokens bị từ chối không biến mất - chúng buộc mục tiêu thứ hai về phía trước cho token đầu tiên bị từ chối. Trên khối lượng công việc mà alpha giảm xuống 0,4, bạn phải trả chi phí nháp cộng với xác minh cộng với roll lại. Ở tính đồng thời cao (giả sử 256 đồng thời), batch giải mã đã đủ lớn để khoảng cách băng thông bộ nhớ giữa "mục tiêu một mình" và "mục tiêu có xác minh" thu hẹp. Dưới alpha 0.55 trên hầu hết phần cứng năm 2026, giải mã thông số kỹ thuật là âm ròng.

Alpha thay đổi tùy theo khối lượng công việc. Trên trò chuyện chung theo phong cách ShareGPT, EAGLE-3 được huấn luyện trên ShareGPT đạt 0,6-0,8. Trên lưu lượng truy cập theo miền cụ thể (mã, y tế, pháp lý), người đứng đầu dự thảo được huấn luyện về dữ liệu chung giảm xuống còn 0,4-0,6. Training một đầu nháp dành riêng cho miền khôi phục alpha - đó là một công việc training nhẹ, nhanh chóng so với tinh chỉnh mục tiêu.

### Sơ lược về các thế hệ EAGLE

- **model nháp cổ điển**: model nhỏ cùng họ. Alpha 0,3-0,5. Cơ sở hạ tầng đơn giản - hai models được tải, bản nháp chạy K về phía trước cho mỗi mục tiêu về phía trước.
- **EAGLE-1 (2024)**: đầu nháp đơn được huấn luyện về các trạng thái ẩn mục tiêu (lớp cuối cùng). Alpha ~0.5-0.6. Tham số nhỏ trên đầu mục tiêu.
- **EAGLE-2 (2025)**: độ dài bản nháp thích ứng và bản nháp dựa trên cây (xác minh nhiều branches trong một lần vượt qua mục tiêu). Alpha ~0,6-0,7. Bộ lập lịch nháp phức tạp hơn.
- **EAGLE-3 (2025-2026)**: trưởng dự thảo được huấn luyện trên nhiều lớp mục tiêu (không chỉ cuối cùng), tốt hơn alignment. Alpha ~0.6-0.8 trong trò chuyện chung.

### Công thức production 2026

1. Ship mục tiêu model đồng bằng. Đo lường TTFT, ITL, thông lượng cơ bản tại đồng thời mục tiêu.
2. Bật bản nháp EAGLE-3 thông qua vLLM `speculative_config`. Chạy lại benchmark.
3. Tỷ lệ chấp nhận nhật ký alpha. vLLM V1 báo cáo điều này là `spec_decode_metrics.accepted_tokens_per_request`. Chia cho độ dài bản nháp được yêu cầu để có được alpha.
4. Nếu alpha < 0,55 trên phân phối lưu lượng truy cập production, hãy tắt giải mã thông số kỹ thuật hoặc huấn luyện bản nháp EAGLE-3 dành riêng cho miền.
5. Khi production đồng thời, hãy chạy lại. Xác nhận P99 ITL không trở nên tồi tệ hơn.

### Cạm bẫy production: Đuôi P99

Ý nghĩa ITL giảm với giải mã thông số kỹ thuật. P99 có thể trở nên tồi tệ hơn nếu bạn không điều chỉnh. Các bản nháp bị từ chối trigger trình tự hai lần (nháp + xác minh-không đạt + cuộn lại). Dưới batch đầy đủ, hai vé đó được nối tiếp. Xem P99 ITL, không phải P50.

### Nơi EAGLE-3 đã được triển khai

Google đã triển khai giải mã suy đoán trong Tổng quan AI vào năm 2025 (chất lượng tương tự, phản hồi nhanh hơn). vLLM V1 ships `speculative_config` làm giao diện được ghi lại; Giải mã suy đoán N-gram GPU trong V1 là biến thể tương thích với prefill theo khối. SGLang hỗ trợ EAGLE-3 làm đường dẫn nháp được đề xuất cho khối lượng công việc nặng tiền tố.

### Toán hòa vốn trong một dòng

Tăng tốc dự kiến: `S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`. Cài đặt `S = 1` giải cho alpha: `alpha_breakeven = verify_overhead / K`. Đối với verify_overhead điển hình ~0.15 và K = 5: `alpha_breakeven = 0.03`. Nhưng đó là toán giải mã thô. Ở tính đồng thời cao, chi phí xác minh tăng lên và batch giải mã đã khấu hao các lần đọc bộ nhớ trên các chuỗi, vì vậy hiệu quả alpha_breakeven tăng lên ~0,45-0,55 trong thực tế.

### Khi nào không nên sử dụng giải mã suy đoán

- Tạo ngoại tuyến Batch-1 trong đó độ trễ không quan trọng. Sử dụng mục tiêu đơn giản.
- Đầu ra rất ngắn (dưới 50 tokens). Dự thảo chi phí chung và xác minh chi phí chi phối.
- Các miền chuyên biệt không có người đứng đầu bản nháp được huấn luyện tên miền. Alpha quá thấp.
- vLLM v0.18.0 cộng với giải mã thông số kỹ thuật model nháp cộng với `--enable-chunked-prefill`. Sự kết hợp này không biên dịch. Ngoại lệ được ghi nhận là giải mã thông số kỹ thuật N-gram GPU trong V1.

## Ứng dụng

`code/main.py` mô phỏng một vòng lặp giải mã có và không có giải mã suy đoán trên một loạt các giá trị alpha và độ dài nháp K. Nó in alpha hòa vốn, tốc độ đo lường và hành vi đuôi. Chạy nó trên một số kết hợp (alpha, K) để xem chính xác nơi giải mã đầu cơ ngừng trả tiền.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-eagle3-rollout.md`. Với model mục tiêu, mô tả phân phối lưu lượng và mục tiêu đồng thời, nó tạo ra một kế hoạch rollout EAGLE-3 theo giai đoạn - benchmark đường cơ sở, kích hoạt config, đo alpha, cổng trên alpha > = 0,55, xem P99 ITL.

## Bài tập

1. Chạy `code/main.py`. Ở K = 5, bạn cần alpha nào để tăng tốc gấp 2 lần? Để tăng tốc gấp 3 lần? Điều đó nhạy cảm với verify_overhead như thế nào?
2. Hãy tưởng tượng lưu lượng truy cập production chia 70% trò chuyện chung, 30% mã. Trò chuyện chung đạt alpha 0.7 với EAGLE-3 được huấn luyện trên ShareGPT; mã đạt alpha 0.4. Alpha hỗn hợp là gì và giải mã thông số kỹ thuật là tích cực ròng?
3. Đọc tài liệu về vLLM `speculative_config`. Đặt tên cho ba chế độ (model nháp, EAGLE, N-gram) và chế độ nào tương thích với phần nạp trước.
4. Bạn thấy ITL trung bình giảm 25% sau khi kích hoạt EAGLE-3 nhưng P99 ITL tăng 15%. Chẩn đoán và đề xuất giảm thiểu.
5. Tính toán chi phí bộ nhớ của đầu dự thảo EAGLE-3 cho Llama 3.3 70B. Làm thế nào để so sánh với việc chạy Llama 3.2 1B như một bản nháp cổ điển?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Giải mã suy đoán | "Bản nháp cộng với xác minh" | Đề xuất K tokens với model giá rẻ, xác minh tất cả K trong một mục tiêu chuyển tiếp |
| Tỷ lệ chấp nhận alpha | "Tỷ lệ chấp nhận thông số kỹ thuật" | Phần dự thảo tokens được mục tiêu chấp nhận; Số liệu duy nhất quan trọng |
| Chiều dài nháp K | "Thông số kỹ thuật K" | Dự thảo đề xuất bao nhiêu tokens cho mỗi mục tiêu chuyển tiếp; Điển hình 4-8 |
| Xác minh epsilon trên cao | "chi phí thông số kỹ thuật" | Chi phí bổ sung để xác minh và cuộn lại so với một mục tiêu đơn giản về phía trước; phát triển cùng với batch |
| ĐẠI BÀNG-3 | "EAGLE mới nhất" | Biến thể 2025-2026; huấn luyện đầu nháp trên nhiều lớp mục tiêu; Alpha 0.6-0.8 trên trò chuyện chung |
| `speculative_config` | "config thông số kỹ thuật vLLM" | Chọn tham gia rõ ràng trong vLLM V1; Không mặc định có nghĩa là không tăng tốc |
| Giải mã thông số kỹ thuật N-gram | "Bản nháp N-gram" | Bản nháp GPU mặt bằng cách sử dụng tra cứu N-gram trong prompt; Chunked-Prefill-tương thích |
| Hòa vốn alpha | "Alpha không hoạt động" | Alpha mà tại đó giải mã thông số kỹ thuật không tăng tốc; Xem nội dung này tại production đồng thời |
| Hai lần nháp bị từ chối | "chi phí cuộn lại" | Hai mục tiêu chuyển tiếp khi bản nháp từ chối; lái đuôi P99 |

## Đọc thêm

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — nguồn có thẩm quyền về khả năng tương thích `speculative_config` và điền trước theo khối trong V1.
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — tập hợp trường chính xác.
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) - công thức đầu dự thảo EAGLE ban đầu.
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) - bản nháp và cây thích ứng.
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) - hệ thống LLM hiệu quả với giải mã đầu cơ.
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) - production rollout danh sách kiểm tra.
