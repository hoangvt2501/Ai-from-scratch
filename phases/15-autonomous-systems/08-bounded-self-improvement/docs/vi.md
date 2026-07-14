# Thiết kế tự cải thiện có giới hạn

> Nghiên cứu đã hội tụ vào bốn primitives để giới hạn một vòng lặp cải thiện bản thân. Các bất biến chính thức phải được giữ trong mọi sửa đổi. Alignment các neo không thể sửa đổi. Các ràng buộc đa mục tiêu mà mọi khía cạnh (an toàn, công bằng, mạnh mẽ) phải được giữ vững, không chỉ hiệu suất. Phát hiện hồi quy tạm dừng vòng lặp khi các chỉ số lịch sử đề xuất khả năng loss. Không ai trong số chúng là bằng chứng về sự an toàn - kết quả lý thuyết thông tin (phức tạp Kolmogorov, định lý Lob) ràng buộc những gì bất kỳ hệ thống nào có thể chứng minh về những người kế nhiệm của chính nó. Chúng là những biện pháp giảm thiểu làm tăng chi phí của sự thất bại thầm lặng.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, bounded-loop with invariant check)
**Kiến thức tiên quyết:** Giai đoạn 15 · 07 (RSI), Giai đoạn 15 · 04 (Tổng giám đốc)
**Thời lượng:** ~60 phút

## Vấn đề

Trình mô phỏng cuộc đua của Bài 7 cho thấy sự khác biệt về tỷ lệ nhỏ kết hợp thành những khoảng cách lớn. Nghiên cứu điển hình DGM của bài 4 cho thấy các vòng lặp có thể chủ động đánh giá của chính chúng. Cả hai kết quả đều chỉ ra cùng một câu hỏi kỹ thuật: bạn có thể đặt những ràng buộc nào vào một vòng lặp tự cải thiện sao cho các ràng buộc không thể bị suy yếu một cách âm thầm bởi chính vòng lặp?

Tóm tắt Hội thảo RSI ICLR 2026 (openreview.net/pdf?id=OsPQ6zTQXV) xác định bốn primitives như vậy. RSP v3.0 của Anthropic (Bài 19) và FSF v3 của DeepMind (Bài 20) đều tham chiếu chúng trong ngưỡng khả năng. Meta HyperAgents làm việc và frameworks cộng đồng như SAHOO (tháng 3 năm 2026) triển khai các tập hợp con trong production.

Khung trung thực: đây là những biện pháp giảm thiểu. Kết quả lý thuyết thông tin ràng buộc những gì bất kỳ hệ thống nào có thể chứng minh về người kế nhiệm của chính nó, và không có thiết kế hiện tại nào đóng vấn đề một cách chính thức. Một vòng lặp có giới hạn tốt sẽ an toàn hơn một vòng lặp không giới hạn, không an toàn về mặt tuyệt đối.

## Khái niệm

### Primitive 1: bất biến chính thức

Bất biến là một thuộc tính phải giữ trước và sau mỗi lần tự sửa đổi. Ví dụ:

- Phân phối đầu ra có điều kiện trên một tiêu đề hiến pháp cố định (Bài 17).
- Không có lệnh gọi công cụ nào đến endpoint trái phép.
- Ghi bộ nhớ đi qua một đường dẫn được ghi lại, có dấu.
- Hàm băm mô-đun của trình đánh giá khớp với phiên bản đã được phê duyệt.

Bất biến được kiểm tra bằng mã bên ngoài mà vòng lặp không thể chỉnh sửa. Nếu một sửa đổi được đề xuất vi phạm bất biến, nó sẽ bị từ chối. Điều này bắt được chế độ giả mạo đánh giá kiểu DGM (Bài 4).

Phần khó khăn là chọn các bất biến cần thiết để đảm bảo an toàn và có thể tính toán được trong thời gian hợp lý. Xác định quá mức tiến độ gian hàng; Under-specifying cho phép vòng lặp đi qua các thuộc tính quan trọng.

### Primitive 2: alignment neo

Neo alignment là biểu diễn bất biến của mục tiêu cốt lõi của vòng lặp, được ghim bên ngoài bề mặt chỉnh sửa của vòng lặp. Ví dụ:

- Một văn bản hiến pháp (Bài 17) được tải từ vị trí chỉ đọc trên mỗi lần gọi.
- Một người đánh giá trong một repository riêng biệt với kiểm soát truy cập độc lập.
- Tệp kê khai quyền công cụ do con người ký và xác minh lại mỗi chu kỳ.

Vai trò của mỏ neo là ngăn chặn sự trôi dạt khách quan. Vòng lặp có thể cải thiện cách nó theo đuổi mục tiêu, nhưng không thể chỉnh sửa mục tiêu là gì.

Chế độ thất bại tinh tế: một neo mà vòng lặp không thể chỉnh sửa vẫn có thể được diễn giải lại bằng một vòng lặp trôi dạt theo cách nó đọc neo. Hiến pháp AI (Bài 17) rõ ràng dựa trên lý trí để xử lý các tình huống mới; lớp lý luận đó là nơi trôi dạt diễn giải. Neo là cần thiết, không đủ.

### Primitive 3: Ràng buộc đa mục tiêu

Một vòng lặp tối ưu hóa một điểm vô hướng duy nhất sẽ tìm thấy các phím tắt. Một vòng lặp phải thỏa mãn đồng thời nhiều ràng buộc cứng có ít phím tắt hơn. Các trục điển hình:

- Hiệu suất (benchmark cấp nhiệm vụ)
- An toàn (đánh giá đội đỏ, tỷ lệ từ chối đã biết-xấu)
- Công bằng (giới hạn tác động khác nhau đối với các nhóm nhỏ nhạy cảm)
- Độ bền (bộ kiểm tra OOD, xử lý đầu vào đối nghịch)

Một sửa đổi chỉ được chấp nhận nếu mọi ràng buộc đều được giữ nguyên. Thống đốc chi phí của bài 13 stacks điều này với những hạn chế về tài chính. Llama Guard của bài 18 cắm vào như một trục an toàn.

### Primitive 4: Phát hiện hồi quy

Điểm số của mỗi chu kỳ được so sánh với phân phối lịch sử. Giảm vượt quá dung sai đã định cấu hình sẽ tạm dừng vòng lặp. Điều này bắt được khả năng im lặng loss nếu không sẽ được hấp thụ vào mức trung bình khi vòng lặp "cải thiện quá khứ" nó.

Một cách triển khai thực tế: lưu trữ điểm số cho mỗi nhiệm vụ của N chu kỳ cuối cùng. Mỗi chu kỳ mới tính toán một delta cho mỗi tác vụ. Nếu bất kỳ delta nào giảm xuống dưới ngưỡng, chu kỳ sẽ bị từ chối và con người sẽ xem xét.

### Giới hạn lý thuyết thông tin

Độ phức tạp Kolmogorov và định lý Lob đặt ra giới hạn trên về những gì một hệ thống có thể chứng minh về chính nó. Máy Godel chính thức của Schmidhuber (Bài học 4) nhắm đến giới hạn cao nhất như vậy; không ai hoàn thành một chứng minh không tầm thường. Kết quả của Lob cho biết: nếu một hệ thống chứng minh được tin rằng "Tôi sẽ làm X nếu tôi chứng minh tôi nên làm X", nó sẽ làm X mà không chứng minh được nó nên làm, một thất bại tự tham chiếu nổi tiếng.

Hàm ý đối với primitives của chúng tôi: họ không thể đóng bài toán an toàn. Chúng làm cho thất bại thầm lặng trở nên tốn kém hơn. Một vòng lặp độc hại hoặc trôi dạt sẽ lặng lẽ bỏ qua một kiểm tra bị thiếu giờ đây phải chủ động làm suy yếu một vòng lặp rõ ràng, đó là một chữ ký dễ phát hiện hơn.

### Một ví dụ điển hình

Giả sử một agent đề xuất sửa đổi. Các stack cổng:

1. Kiểm tra bất biến: hàm băm mô-đun, tệp kê khai quyền công cụ, tiêu đề hiến pháp.
2. Kiểm tra neo: câu lệnh mục tiêu khớp với phiên bản đã được phê duyệt (theo byte hoặc ngữ nghĩa).
3. Đánh giá đa mục tiêu: trục hiệu suất, an toàn, công bằng, mạnh mẽ.
4. Phát hiện hồi quy: không có trục nào giảm nhiều hơn dung sai.

Cả bốn đều phải vượt qua để chỉnh sửa hạ cánh. Bất kỳ lỗi đơn lẻ nào cũng sẽ tạm dừng vòng lặp.

## Ứng dụng

`code/main.py` chạy một vòng lặp cải thiện bản thân có giới hạn trên đồ chơi kiểu DGM từ Bài học 4, nhưng với bốn primitives được xếp lớp trên cùng. Mỗi primitive có thể được bật hoặc tắt riêng lẻ. Minh chứng là mỗi primitive bắt được một class lỗi cụ thể và việc loại bỏ bất kỳ  nào trong số chúng sẽ cho phép lỗi đó class qua.

## Sản phẩm bàn giao

`outputs/skill-bounded-loop-review.md` kiểm tra một vòng lặp có giới hạn được đề xuất và chấm điểm cái nào trong bốn primitives mà nó thực sự thực hiện so với tuyên bố.

## Bài tập

1. Chạy `code/main.py` với tất cả các primitives được bật. Xác nhận vòng lặp vẫn cải thiện chỉ số chính mà không để hack giành chiến thắng.

2. Tắt tính năng phát hiện hồi quy. Xây dựng một đầu vào mà điều này dẫn đến khả năng im lặng loss được chấp nhận.

3. Tắt ràng buộc đa mục tiêu. Hiển thị vòng lặp hội tụ trên trục hiệu suất trong khi trục an toàn giảm xuống.

4. Thiết kế một neo alignment cho một agent mã hóa. Văn bản nào, được lưu trữ ở đâu, kiểm tra như thế nào?

5. Đọc tóm tắt Hội thảo RSI ICLR 2026. Chọn một trong bốn primitives và đề xuất một cải tiến cụ thể cho tình trạng hiện tại.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| Bất biến | "Tài sản luôn đúng" | Thuộc tính được kiểm tra bằng mã bên ngoài trước và sau mỗi lần chỉnh sửa |
| Alignment neo | "Mục tiêu được ghim" | Biểu diễn mục tiêu cốt lõi bất biến bên ngoài bề mặt chỉnh sửa của vòng lặp |
| Ràng buộc đa mục tiêu | "Tất cả các trục phải giữ" | Hiệu suất, an toàn, công bằng, mạnh mẽ - tất cả đều được yêu cầu |
| Phát hiện hồi quy | "Tạm dừng khi thả" | Tạm dừng vòng lặp khi các delta chỉ số lịch sử đề xuất khả năng loss |
| Kolmogorov bị ràng buộc | "Giới hạn lý thuyết thông tin" | Giới hạn những gì một hệ thống có thể chứng minh về người kế nhiệm của chính nó |
| Định lý Lob | "Bẫy tự tham khảo" | Hệ thống có thể hành động theo "Tôi nên" mà không cần chứng minh nó nên |
| Cổng stack | "Kiểm tra nhiều lớp" | Nhiều primitives kết hợp; bất kỳ thất bại nào cũng từ chối sửa đổi |
| Cải tiến có giới hạn | "Giảm thiểu, không phải bằng chứng" | Tăng chi phí thất bại thầm lặng; không đóng vấn đề an toàn |

## Đọc thêm

- [ICLR 2026 RSI Workshop summary (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) - hội tụ bốn primitive.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — ngưỡng khả năng đa mục tiêu.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — giám sát alignment lừa đảo như một primitive bất biến.
- [Schmidhuber (2003). Godel Machines](https://people.idsia.ch/~juergen/goedelmachine.html) - tổ tiên chứng minh chính thức của những primitives này.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) - mỏ neo alignment dựa trên lý do.
