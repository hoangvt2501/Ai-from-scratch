# Dòng Tối ưu hóa Tùy chọn Trực tiếp

> Rafailov et al. (2023) cho thấy tối ưu của RLHF có dạng đóng về dữ liệu ưu tiên, vì vậy bạn có thể bỏ qua model phần thưởng rõ ràng và tối ưu hóa policy trực tiếp. Cái nhìn sâu sắc đó đã tạo ra một gia đình - IPO, KTO, SimPO, ORPO, BPO - mỗi gia đình sửa chữa một chế độ thất bại của DPO. Vào năm 2026, các thuật toán alignment trực tiếp ship nhiều lần chạy sau training hơn so với PPO. Nhưng đường cong tối ưu hóa quá mức từ Bài học 2 vẫn được áp dụng: DAA không thoát khỏi Goodhart, chúng chỉ di chuyển đến nơi nó cắn.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, six-variant preference-loss comparator)
**Kiến thức tiên quyết:** Giai đoạn 18 · 01 (InstructGPT), Giai đoạn 18 · 02 (Hack phần thưởng), Giai đoạn 10 · 08 (DPO thông tin cơ bản)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Lấy dạng đóng DPO từ RLHF-with-KL tối ưu.
- Nêu chế độ thất bại của từng bản sửa lỗi IPO, KTO, SimPO, ORPO, BPO trong DPO.
- Phân biệt "khoảng cách phần thưởng ngầm" với "sức mạnh ưu tiên" và giải thích lý do tại sao ánh xạ nhận dạng của IPO lại quan trọng.
- Giải thích lý do tại sao Rafailov và cộng sự (NeurIPS 2024) chứng minh DAA tối ưu hóa quá mức mặc dù không có RM rõ ràng.

## Vấn đề

Mục tiêu RLHF (Bài 1):

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

có mức tối ưu đã biết:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

Vì vậy, phần thưởng được xác định ngầm bởi tỷ lệ giữa policy tối ưu với tham chiếu:

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

Thay thế điều này vào likelihood tùy chọn Bradley-Terry và hàm phân vùng `Z(x)` hủy bỏ vì nó chỉ phụ thuộc vào `x`. Những gì còn lại chỉ là một loss trong policy parameters - không cần phần thưởng model. Đó là DPO.

Nếp nhăn: dẫn xuất giả định có thể đạt được mức tối ưu, dữ liệu ưu tiên đang phân phối và policy tham chiếu là neo chế độ thực. Không có cái nào trong số này đúng chính xác. Mỗi thành viên trong gia đình sửa một giả định vi phạm khác nhau.

## Khái niệm

### DPO (Rafailov và cộng sự, 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

Điều gì có thể xảy ra:

- Khoảng cách phần thưởng ngầm `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` là không giới hạn. Một sở thích nhỏ có thể tạo ra một khoảng cách lớn tùy ý.
- loss điều khiển các đầu dò nhật ký đã chọn và bị từ chối theo các hướng ngược lại. Nó có thể đẩy log-prob tuyệt đối đã chọn xuống miễn là bị từ chối giảm nhanh hơn. Đây là hiện tượng Phản ứng được chọn suy giảm.
- Tùy chọn ngoài phân phối (cặp hiếm so với cặp hiếm hiếm) tạo ra phần thưởng ngầm tùy ý.

### IPO (Azar và cộng sự, 2024)

Tối ưu hóa tùy chọn danh tính thay thế log-sigmoid bằng ánh xạ danh tính trên xác suất ưu tiên. loss trở thành sai số bình phương trên một mục tiêu có giới hạn:

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

Lề được giới hạn bởi `1/(2 beta)`. Sức mạnh ưu tiên và khoảng cách phần thưởng ngầm là tỷ lệ. Không nổ tung.

### KTO (Ethayarajh và cộng sự, 2024)

Tối ưu hóa Kahneman-Tversky giảm hoàn toàn cấu trúc theo cặp. Cho một đầu ra được dán nhãn duy nhất và một tín hiệu nhị phân "mong muốn" hoặc "không mong muốn", nó ánh xạ đến một tiện ích lý thuyết triển vọng:

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

với trọng số khác nhau cho lãi và lỗ (ác cảm loss). Lợi ích: bạn có thể sử dụng dữ liệu chưa ghép nối, dữ liệu này phong phú hơn nhiều.

### SimPO (Meng và cộng sự, 2024)

Tối ưu hóa tùy chọn đơn giản căn chỉnh tín hiệu training với thế hệ. Xóa hoàn toàn các policy tham chiếu và chuẩn hóa likelihood nhật ký theo độ dài:

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

với một biên độ `gamma` để ổn định. Chuẩn hóa chiều dài loại bỏ động lực khai thác chế độ hỏng hóc bias chiều dài của DPO (`y_w` dài hơn mang lại khoảng cách log-prob lớn hơn theo cấu trúc).

### ORPO (Hong và cộng sự, 2024)

Tối ưu hóa tùy chọn tỷ lệ cược thêm thuật ngữ ưu tiên vào likelihood nhật ký phủ định SFT tiêu chuẩn:

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

Không có policy tham khảo - thuật ngữ SFT là bộ chính quy. Huấn luyện trong một giai đoạn duy nhất từ model cơ sở đến model căn chỉnh. Không có checkpoint SFT riêng biệt.

### BPO (đệ trình ICLR 2026, OpenReview id = b97EwMUWu7)

Xác định vấn đề Phản hồi đã chọn bị suy giảm: DPO giữ nguyên `y_w > y_l` xếp hạng nhưng xác suất nhật ký tuyệt đối của `y_w` có thể giảm. BPO thêm một đợt điều chỉnh một dòng để phạt các động thái đi xuống trên phản ứng đã chọn. Báo cáo +10.1% accuracy trên Llama-3.1-8B-Hướng dẫn về lý luận toán học trên DPO.

### Kết quả chung: DAA vẫn tối ưu hóa quá mức

Rafailov và cộng sự. "Luật mở rộng quy mô cho phần thưởng Model tối ưu hóa quá mức trong các thuật toán Alignment trực tiếp" (NeurIPS 2024) đã huấn luyện policies với DPO, IPO, SLiC trên nhiều datasets trên ngân sách KL. Các đường cong vàng-phần thưởng-vs-KL có cùng hình dạng đỉnh và sụp đổ của Gao et al. Phần thưởng ngầm định truy vấn các mẫu ngoài phân phối trong quá trình training; Chính quy hóa KL không ổn định điều này.

DAA không thoát khỏi Goodhart. Họ thay đổi bề mặt nơi nó cắn từ "phần thưởng model tối ưu hóa quá mức" thành "tỷ lệ tham chiếu policy được tối ưu hóa quá mức". Bản sửa lỗi phổ quát - dữ liệu tốt hơn, tổng hợp, dừng sớm - áp dụng cho cả hai.

### Lựa chọn trong số đó (2026)

- Nếu bạn có dữ liệu tùy chọn được ghép nối lớn: DPO với beta bảo thủ, SimPO nếu bias độ dài rõ ràng.
- Nếu bạn có phản hồi nhị phân chưa ghép nối: KTO.
- Nếu bạn muốn một giai đoạn pipeline từ một model cơ sở: ORPO.
- Nếu bạn thấy các probs nhật ký đã chọn bị suy giảm trong nhật ký DPO: BPO.
- Nếu sức mạnh ưu đãi rất khác nhau và DPO bão hòa: IPO.

Mỗi phòng thí nghiệm chạy tất cả năm pin và chọn người chiến thắng cho mỗi nhiệm vụ. Không có lý do gì tối ưu lại giống nhau đối với lý luận toán học và an toàn.

```figure
dpo-margin
```

## Ứng dụng

`code/main.py` so sánh sáu khoản lỗ (DPO, IPO, KTO, SimPO, ORPO, BPO) trên một dataset sở thích đồ chơi trong đó sức mạnh ưu tiên thực sự thay đổi theo cặp. Mỗi loss được tối ưu hóa dựa trên cùng một mẫu 500 cặp với một softmax policy nhỏ. Biểu đồ tỷ lệ thắng cuối cùng, độ trôi được chọn-log-prob và chênh lệch phần thưởng ngầm cho mỗi phương pháp.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-preference-loss-selector.md`. Đưa ra số liệu thống kê dataset (ghép nối so với không ghép nối, cường độ ưu tiên thay đổi so với đồng đều, phân bố độ dài) và mục tiêu (một giai đoạn hoặc SFT-sau đó-ưu tiên), đề xuất loss ưu tiên và báo cáo chế độ lỗi mà nó bảo vệ.

## Bài tập

1. Chạy `code/main.py`. Báo cáo sự sụt giảm cuối cùng của chosen-log-prob đối với DPO và BPO. BPO nên giữ xác suất tuyệt đối được chọn cao hơn - hãy xác minh điều này.

2. Sửa đổi dữ liệu ưu tiên để tất cả các cặp có sức mạnh như nhau. Phương pháp nào trong sáu phương pháp mạnh mẽ nhất? Cái nào xuống cấp? Giải thích lợi thế của IPO tại đây.

3. Làm cho các câu trả lời bị từ chối dài hơn trung bình 2 lần so với đã chọn. Không thay đổi bất cứ điều gì khác, hiển thị khai thác độ dài của DPO bằng số và bản sửa lỗi của SimPO.

4. Rafailov và cộng sự (NeurIPS 2024) tuyên bố DAA tối ưu hóa quá mức. Tái tạo phiên bản một điểm: phân kỳ KL được chọn trừ và quan sát tối ưu hóa quá mức trong DPO ở beta lớn.

5. Đọc tóm tắt bài báo BPO (OpenReview b97EwMUWu7). Viết ra hiệu chỉnh một dòng BPO thêm vào DPO. Xác nhận chống lại việc thực hiện trong `code/main.py`.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| DPO | "RLHF không có phần thưởng model" | Loss có nguồn gốc từ dạng đóng RLHF tối ưu; Chỉ policy parameters |
| Phần thưởng ngầm | "Tỷ lệ log" | 'beta * log(pi(y\ | x) / pi_ref (y \ | x))' — phần thưởng ngụ ý DPO |
| IPO | "DPO giới hạn" | Thay thế log-sigmoid bằng danh tính; Khoảng cách phần thưởng ngầm được giới hạn bởi `1/(2 beta)` |
| KTO | "DPO chưa ghép nối" | Tiện ích lý thuyết triển vọng so với các nhãn đơn lẻ với ác cảm loss |
| SimPO | "DPO không có tài liệu tham khảo" | Độ dài-chuẩn hóa log-likelihood + lề; Không có policy tham khảo |
| ORPO | "DPO một giai đoạn" | NLL + thuật ngữ ưu tiên tỷ lệ cược; Tàu từ căn cứ model trong một lần vượt qua |
| BPO | "DPO bảo quản được chọn" | DPO cộng với một hình phạt cho việc giảm log-prob tuyệt đối của phản hồi đã chọn |
| Suy thoái được chọn | "Được chọn đi xuống" | DPO làm giảm log-prob đã chọn miễn là bị từ chối giảm nhanh hơn |
| DAA | "thuật toán alignment trực tiếp" | Bất kỳ phương thức loss tùy chọn nào bỏ qua RM rõ ràng |

## Đọc thêm

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) - IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
