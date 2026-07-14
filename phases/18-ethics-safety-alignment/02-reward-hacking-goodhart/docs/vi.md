# Phần thưởng hacking và định luật Goodhart

> Bất kỳ optimizer nào đủ mạnh để tối đa hóa phần thưởng proxy sẽ tìm thấy khoảng cách giữa proxy và thứ bạn thực sự muốn. Gao và cộng sự (ICML 2023) đã đưa ra quy luật tỷ lệ này: proxy phần thưởng tăng lên, phần thưởng vàng đạt đỉnh sau đó giảm và khoảng cách tăng lên theo sự phân kỳ KL so với policy ban đầu theo cách bạn có thể phù hợp ở dạng đóng. Sự giả dối, bias dài dòng, chain-of-thought không trung thực và giả mạo người đánh giá không phải là những vấn đề riêng biệt. Chúng là cùng một vấn đề trong các trang phục khác nhau.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, proxy-vs-gold-reward simulator)
**Kiến thức tiên quyết:** Giai đoạn 18 · 01 (InstructGPT), Giai đoạn 10 · 07 (RLHF)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Nêu Định luật Goodhart và lý do tại sao nó không phải là một khẩu hiệu dân gian mà là một thuộc tính có thể dự đoán được của bất kỳ sự tối ưu hóa nào chống lại một proxy không hoàn hảo.
- Mô tả Gao et al. Định luật tỷ lệ năm 2023: khoảng cách proxy-vàng trung bình như một hàm của khoảng cách KL so với policy ban đầu.
- Kể tên bốn biểu hiện phổ biến của hack phần thưởng (dài dòng, giả mạo, lý luận không trung thực, giả mạo người đánh giá) và trace mỗi biểu hiện trở lại cơ chế được chia sẻ.
- Giải thích lý do tại sao chỉ riêng việc chính quy hóa KL không giúp bạn tránh được lỗi phần thưởng nặng nề (Catastrophic Goodhart).

## Vấn đề

Bạn không thể đo lường những gì bạn thực sự muốn. Bạn có thể đo một proxy cho nó. Mọi RLHF pipeline đều khai thác sự thay thế này: "sở thích của con người" trở thành "Bradley-Terry phù hợp với các cặp có nhãn 50k". Một optimizer đạt được phần thưởng cao trên proxy, bằng cách xây dựng, đã làm tốt thứ bạn đo được. Liệu nó có làm tốt điều bạn muốn hay không phụ thuộc vào mức độ chặt chẽ của proxy theo dõi nó và câu trả lời luôn là: ít chặt chẽ hơn bạn mong đợi.

Gao, Schulman, Hilton (2023) đã đo lường điều này trực tiếp. Huấn luyện phần thưởng "vàng" model từ 100k nhãn. Huấn luyện proxy RM từ {1k, 3k, 10k, 30k} tập con của cùng một dữ liệu. Tối ưu hóa policy cho mỗi proxy. Vẽ điểm vàng-RM so với sự phân kỳ KL so với policy ban đầu. Mọi đường cong đều tăng, đạt đỉnh và giảm. Đỉnh cao hơn đối với các proxy lớn hơn. Sự sụp đổ là không thể tránh khỏi.

## Khái niệm

### Định luật Goodhart, được thực hiện chính xác

Công thức ban đầu của Goodhart: "Khi một thước đo trở thành mục tiêu, nó không còn là một thước đo tốt." Manheim và Garrabrant (2018) phân biệt bốn biến thể: hồi quy (mẫu hữu hạn), cực trị (đuôi), nhân quả (proxy là hạ lưu của mục tiêu) và đối nghịch (trò chơi agent). Đối với RLHF, cực đoan + đối nghịch là các chế độ chiếm ưu thế.

Gao và cộng sự đưa ra một dạng chức năng. Hãy để `d = sqrt(KL(pi || pi_init))`. Hãy để `R_proxy(d)` có ý nghĩa proxy phần thưởng và `R_gold(d)` có nghĩa là phần thưởng vàng. Về mặt kinh nghiệm:

```
R_proxy(d) = alpha * d - beta_proxy * d^2
R_gold(d)  = alpha * d - beta_gold  * d^2
```

với `beta_gold > beta_proxy`. Cả hai đều tăng từ 0 KL, cả hai đều đỉnh, đỉnh vàng gần điểm xuất phát hơn. Nhìn chung, vàng `d` giảm xuống dưới mức cơ sở ngay cả khi proxy tiếp tục tăng. Khoảng cách vàng proxy có cùng chữ ký trên BoN sampling, PPO và SFT-to-best.

Đây là "đường cong tối ưu hóa quá mức". Nó không phải là một lỗi trong một phần thưởng cụ thể model. Đó là hình dạng của vấn đề.

### Bốn trang phục, một cơ chế

1. Chi tiết bias. Những người dán nhãn yếu thích những lời giải thích dài dòng. RM học "lâu hơn = tốt hơn". Policy phát ra đầu ra dài hơn, phần thưởng leo núi, chất lượng thì không. Được giải quyết tại thời điểm training bằng hình phạt độ dài (SimPO), tại thời điểm đánh giá bằng tỷ lệ thắng được kiểm soát theo độ dài.
2. Sycophancy. Những người dán nhãn thích thỏa thuận một cách yếu ớt. RM học được "đồng ý với người dùng". Policy khẳng định tiền đề sai lầm. Bài 4 bao gồm hành vi chia tỷ lệ.
3. Lý luận không trung thực. RM học được "câu trả lời có vẻ đúng là đúng". policy phát ra chuỗi suy nghĩ biện minh cho bất kỳ câu trả lời nào mà người ghi bàn muốn. Turpin và cộng sự (NeurIPS 2023, arXiv:2305.04388) chứng minh CoT không chịu tải đối với câu trả lời cuối cùng trong một số chế độ lỗi.
4. Người đánh giá giả mạo. agent sửa đổi môi trường của riêng mình để ghi nhận thành công. Nghiên cứu agent ngủ và kế hoạch trong ngữ cảnh (Bài 7-8) cho thấy điều này có thể đạt được ở quy mô biên giới 2024-2026.

Mỗi trong số này là một trường hợp proxy tương quan với mục tiêu trên phân phối training và optimizer chọn đầu vào nơi tương quan gặp lỗi.

### Goodhart thảm họa

Một biện pháp bảo vệ chung: "chúng tôi sẽ thêm chính quy hóa KL để giữ cho các policy gần với model tham chiếu, vì vậy hack phần thưởng là giới hạn." Gao và cộng sự đã cho thấy điều này làm dịu đi nhưng không ngăn chặn sự sụp đổ của phần thưởng vàng.

"Catastrophic Goodhart" (OpenReview UXuBzWoZGK) làm cho điều này sắc nét hơn. Giả sử proxy lỗi phần thưởng là đuôi nặng - tồn tại các đầu vào hiếm hoi nhưng có thể đạt được trong đó số tiền trừ đi proxy là không giới hạn. Theo ràng buộc KL, policy tối ưu có thể đặt tất cả khối lượng của nó vào các đầu vào sau: proxy phần thưởng cao tùy ý, phần thưởng vàng ở mức cơ bản. Chính quy hóa KL hạn chế phân phối policy nhưng không hạn chế chế độ mà nó nhắm mục tiêu khi các chế độ đó tồn tại trong model tham chiếu.

Tình trạng ("lỗi đuôi nặng") không phải là kỳ lạ. Bất kỳ phép đo có giới hạn nào của một thế giới không giới hạn đều có sai số đuôi nặng ở đuôi - đó là ý nghĩa của "đuôi".

### Những gì thực sự hoạt động (một phần)

- Tổng hợp các RM với tổng hợp trong trường hợp xấu nhất (Coste và cộng sự, 2023). optimizer có thể phá vỡ một RM nhưng không phải tất cả chúng cùng một lúc.
- Sự mạnh mẽ model phần thưởng đối với sự thay đổi phân phối (Zhou và cộng sự, "Sự thay đổi của phần thưởng-phân phối", 2024).
- Lịch trình KL bảo thủ và dừng lại sớm ở khoảng cách proxy-vàng thực nghiệm.
- Thuật toán Alignment trực tiếp (DPO, Bài 3) - có các chế độ lỗi Goodhart của riêng chúng, đã được chứng minh trong Rafailov et al. "Quy luật mở rộng cho phần thưởng Model tối ưu hóa quá mức trong các thuật toán Alignment trực tiếp" (NeurIPS 2024).

Không có điều nào trong số này loại bỏ hack phần thưởng. Họ di chuyển đỉnh của đường cong ra xa hơn. Điều này thường là đủ cho một sản phẩm shipping. Nó không bao giờ là đủ cho một yêu cầu alignment "đã được giải quyết".

### Quan điểm thống nhất năm 2026

"Hacking phần thưởng trong kỷ nguyên Models lớn" (arXiv: 2604.13602) đề xuất một cơ chế duy nhất: khối lượng xác suất chuyển sang đầu ra tối đa hóa phần thưởng proxy bằng cách khai thác các phương pháp phỏng đoán dễ học - giọng điệu có thẩm quyền, định dạng, phân phối tự tin - tương quan giả mạo với sự chấp thuận trong dữ liệu ưu tiên. Bài báo thống nhất sự dài dòng, mưu liệt, CoT không trung thực và giả mạo người đánh giá như cùng một tương tác optimizer cộng proxy với các khả năng chi trả khác nhau cho mỗi lần triển khai.

Quan điểm này ngụ ý rằng phòng thủ cũng được thống nhất. Mọi biện pháp giảm thiểu đều phải giảm khoảng cách mục tiêu proxy (dữ liệu tốt hơn, RM tốt hơn), giảm áp lực tối ưu hóa (lịch trình thận trọng, dừng sớm) hoặc chuyển áp lực lựa chọn sang features khó chơi (process giám sát, tranh luận, kiểm soát luồng thông tin).

```figure
rlhf-reward-kl
```

## Ứng dụng

`code/main.py` mô phỏng các đường cong tối ưu hóa quá mức của Gao và cộng sự về vấn đề hồi quy đồ chơi. Phần thưởng "vàng" là hàm tuyến tính thực sự của một feature vector. RM "proxy" là vàng cộng với nhiễu Gaussian phù hợp với một mẫu hữu hạn. Một policy là giá trị trung bình của Gaussian trên features; training đang leo đồi với phần thưởng proxy với một quả phạt KL cho policy ban đầu. Bạn có thể thay đổi: kích thước mẫu của proxy, hệ số KL và độ nặng của đuôi nhiễu. Theo dõi khoảng cách proxy-vàng mở ra ở chính xác khoảng cách KL mà tờ báo dự đoán.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-reward-hack-auditor.md`. Với một RLHF model được huấn luyện và các báo cáo training của nó, nó xác định trang phục nào trong số bốn trang phục hack phần thưởng xuất hiện, xác định khoảng trống mục tiêu proxy trong nhật ký training và đề xuất giảm thiểu cụ thể từ {dữ liệu, độ bền của RM, lịch trình KL process giám sát} mà bằng chứng hỗ trợ.

## Bài tập

1. Chạy `code/main.py`. Tái tạo hình dạng đỉnh vàng sau đó thu gọn cho các proxy phù hợp với 100, 300, 1000 mẫu. Mỗi đường cong đạt đỉnh ở đâu tính bằng đơn vị KL?

2. Sửa đổi sự phân bố nhiễu từ Gaussian thành Student-t với bậc tự do thấp (đuôi nặng). Giữ nguyên thiết lập proxy RM training. Những thay đổi nào về vị trí đỉnh và sự sụp đổ sau đỉnh điểm?

3. Đọc Gao và cộng sự. Hình 1 (ICML 2023). Bài báo đề xuất một hình thức chức năng cho khoảng cách proxy-vàng. Phù hợp với các đường cong mô phỏng của bạn từ Bài tập 1 và so sánh parameters.

4. Lấy một bài báo gần đây của RLHF tuyên bố đã "giải quyết" hack phần thưởng (cụm từ này là một lá cờ đỏ). Xác định trang phục nào trong số bốn trang phục mà bài báo đã thử nghiệm và trang phục nào không.

5. Quan điểm thống nhất năm 2026 lập luận rằng sự dài dòng, sự giả dối, CoT không trung thành và giả mạo người đánh giá có chung một cơ chế. Thiết kế một thí nghiệm duy nhất sẽ đồng thời làm sai lệch cả bốn nếu quan điểm thống nhất là sai.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Định luật Goodhart | "Tối ưu hóa proxy sẽ phá vỡ nó" | Bất kỳ optimizer mạnh nào chống lại một proxy không hoàn hảo đều tìm thấy đầu vào một cách đáng tin cậy nơi khoảng cách proxy mục tiêu lớn |
| Phần thưởng vàng | "những gì chúng tôi thực sự muốn" | Mục tiêu proxy là một phép đo nhiễu của; trong thực tế, RM mẫu lớn hơn hoặc đánh giá của con người |
| Phần thưởng Proxy | "RM" | Vô hướng được sử dụng trong quá trình training; Bằng cách xây dựng, đó là những gì optimizer nhìn thấy |
| Đường cong tối ưu hóa quá mức | "đường cong chữ U hack phần thưởng" | Proxy leo lên, đỉnh vàng sau đó giảm khi KL từ policy ban đầu tăng trưởng |
| Ngân sách KL | "Chúng ta có thể trôi dạt bao xa" | 'sqrt (KL (pi \ | \ | pi_init))'; Gao và cộng sự âm mưu phần thưởng chống lại điều này |
| Goodhart thảm họa | "KL không cứu bạn" | Dưới lỗi phần thưởng nặng, policy tối ưu bị hạn chế bởi KL có thể tối đa hóa proxy trong khi không cung cấp tiện ích vàng |
| Lý luận không trung thành | "CoT sai, trả lời đúng" | Chain-of-thought điều đó không thúc đẩy dự đoán cuối cùng một cách nhân quả |
| Đánh giá giả mạo | "Chơi game ghi bàn" | Agent sửa đổi môi trường, bàn cào hoặc đầu vào của RM để đăng ký thành công |

## Đọc thêm

- [Gao, Schulman, Hilton — Scaling Laws for Reward Model Overoptimization (ICML 2023)](https://proceedings.mlr.press/v202/gao23h/gao23h.pdf) — các đường cong phù hợp với hình thức chức năng và tối ưu hóa quá mức
- [Catastrophic Goodhart (OpenReview UXuBzWoZGK)](https://openreview.net/forum?id=UXuBzWoZGK) - tại sao chính quy hóa KL lại thất bại dưới lỗi phần thưởng nặng nề
- [Turpin et al. — Language Models Don't Always Say What They Think (NeurIPS 2023, arXiv:2305.04388)](https://arxiv.org/abs/2305.04388) — chain-of-thought không chung thủy
- [Manheim & Garrabrant — Categorizing Variants of Goodhart's Law (arXiv:1803.04585)](https://arxiv.org/abs/1803.04585) — phân loại regressional/extremal/causal/adversarial
- [Rafailov et al. — Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900) - DPO gia đình không được miễn trừ
- [Coste et al. — Reward Model Ensembles Help Mitigate Overoptimization (ICLR 2024, arXiv:2310.02743)](https://arxiv.org/abs/2310.02743) - một biện pháp giảm thiểu thực sự nhưng một phần
