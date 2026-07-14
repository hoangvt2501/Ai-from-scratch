# Song song DualPipe

> DeepSeek-V3 được huấn luyện trên 2.048 GPUs H800 với MoE chuyên gia nằm rải rác trên các nút. Giao tiếp tất cả của chuyên gia đa nút tốn 1 GPU giờ giao tiếp cho mỗi 1 GPU giờ tính toán. GPUs nhàn rỗi một nửa thời gian. DualPipe (DeepSeek, tháng 12 năm 2024) là một pipeline hai chiều chồng chéo tính toán tiến và lùi với các giao tiếp tất cả mà chúng trigger. Bong bóng giảm, thông lượng tăng lên và việc giữ hai bản sao model parameter ("kép" đặt tên) là rẻ khi Expert Parallelism đã phân tán các chuyên gia trên các cấp bậc. Bài học này là một hướng dẫn kiểu Tìm hiểu về những gì DualPipe thực sự làm và lý do tại sao cải tiến DualPipeV của Sea AI Lab giảm chi phí parameter gấp 2 lần với chi phí là bong bóng chặt chẽ hơn một chút.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, schedule simulator)
**Kiến thức tiên quyết:** Giai đoạn 10 · 05 (training phân tán, FSDP, DeepSpeed), Giai đoạn 10 · 14 (kiến trúc model mở và MoE)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Đặt tên cho bốn thành phần của một đoạn tiến-lùi DualPipe và lý do tại sao mỗi thành phần có cửa sổ chồng chéo riêng.
- Giải thích vấn đề bong bóng pipeline trên quy mô lớn và ý nghĩa của "bong bóng không có bong bóng" trong thực tế so với trong tiếp thị.
- Trace lịch trình DualPipe bằng tay cho 8 cấp độ PP và 16 batches vi mô và xác nhận các luồng tiến và ngược lấp đầy các vị trí nhàn rỗi của nhau.
- Nêu sự đánh đổi mà DualPipeV (Sea AI Lab, 2025) thực hiện: giảm bản sao parameter 2x với chi phí là bong bóng lớn hơn một chút khi Expert Parallelism không hoạt động.

## Vấn đề

Training một MoE model 671B trên 2k H800 GPUs gặp phải ba nút thắt cổ chai kép:

1. **Áp lực bộ nhớ.** Mỗi GPU chứa một lát cắt của model. Bộ nhớ kích hoạt ở trình tự 8k trên 61 lớp trên 128 đầu là rất lớn.
2. **Pipeline bong bóng.** Song song pipeline truyền thống (GPipe, 1F1B) khiến GPUs không hoạt động trong khi họ chờ đầu vào hoặc gradient của giai đoạn. Ở 8 giai đoạn, khoảng 12% thời gian GPU có thể là bong bóng ngay cả với lịch trình 1F1B.
3. **Nút chéo tất cả.** MoE với tính song song của chuyên gia phân tán các chuyên gia trên các nút. Mỗi forward pass triggers một tokens tất cả để gửi cho các chuyên gia của họ và một  khác để kết hợp. Ở mức 2k GPUs điều này dễ dàng trở thành tỷ lệ tính toán trên giao tiếp 1:1.

Mỗi giải pháp đều có các giải pháp riêng biệt: gradient điểm kiểm tra bộ nhớ, Zero Bubble (Sea AI Lab, 2023) cho bong bóng pipeline, hạt nhân giao tiếp song song chuyên gia cho tất cả. Những gì DualPipe làm là làm cho chúng chơi cùng nhau. Lịch trình chồng chéo tính toán và giao tiếp trong một khối tiến-lùi duy nhất, tiêm đồng thời micro-batches từ cả hai đầu của pipeline và sử dụng lịch trình kết quả để ẩn tất cả bên trong windows tính toán.

Kết quả được báo cáo: gần như loại bỏ bong bóng pipeline, hơn 95% GPU sử dụng trong lần chạy 14.8T-token training của DeepSeek-V3.

## Khái niệm

### Pipeline bồi dưỡng song song

Chia model lớp N trên các thiết bị P. Thiết bị `i` giữ các lớp `i * N/P .. (i+1) * N/P - 1`. Một batch vi mô chảy về phía trước qua các thiết bị 0 đến P-1, sau đó lùi từ P-1 đến 0. Mỗi thiết bị chỉ có thể bắt đầu chuyển tiếp stage khi thiết bị prior gửi đầu ra của nó và chỉ có thể bắt đầu lùi khi thiết bị xuôi dòng gửi gradient ngược dòng.

GPipe (Huang và cộng sự, 2019) lên lịch một batch vi mô tại một thời điểm, điều này lãng phí hầu hết thời gian GPU. 1F1B (Narayanan và cộng sự, 2021) xen kẽ các đường chuyền tiến và lùi cho nhiều batches vi mô. Zero Bubble (Qi và cộng sự, 2023) chia backward pass thành hai phần - lùi cho đầu vào (B) và ngược cho trọng số (W) - và lên lịch cho chúng để lấp đầy bong bóng. Sau Zero Bubble, pipeline gần như chặt chẽ.

DualPipe là bước tiếp theo. Nó thêm hai ý tưởng lên trên:

### Ý tưởng 1: phân hủy khối

Mỗi đoạn chuyển tiếp được chia thành bốn thành phần:

- **Attention.** Q/K/V phép chiếu, attention, dự báo đầu ra.
- **Gửi tất cả.** Giao tiếp giữa các nút gửi tokens cho các chuyên gia của họ.
- **MLP.** Tính toán chuyên gia MoE.
- **Kết hợp tất cả.** Giao tiếp giữa các nút mang lại kết quả đầu ra của chuyên gia.

Một chunk ngược thêm gradient phiên bản của mỗi phần này. DualPipe lên lịch cho chúng để quá trình gửi tất cả xảy ra song song với tính toán attention của chunk tiếp theo và kết hợp tất cả xảy ra song song với tính toán MLP của chunk tiếp theo.

### Ý tưởng 2: lập lịch hai chiều

Hầu hết các lịch trình pipeline tiêm micro-batches từ stage 0 và chảy về phía stage P-1. DualPipe tiêm micro-batches từ CẢ HAI đầu. Stage 0 nhìn thấy micro-batches chuyển tiếp bắt nguồn từ đó; stage P-1 nhìn thấy micro-batches về phía trước cũng bắt nguồn từ đó. Hai luồng gặp nhau ở giữa.

Để điều này hoạt động, `i` thiết bị phải giữ CẢ lớp pipeline đầu `i` VÀ lớp pipeline muộn `P - 1 - i`. Đó là phần "kép" của DualPipe: mỗi thiết bị giữ hai bản sao của model lớp mà nó cần phục vụ (một cho mỗi hướng). Ở quy mô của DeepSeek-V3, đây là chi phí sao chép parameter gấp 2 lần. Nó có giá cả phải chăng vì Expert Parallelism đã trải rộng các chuyên gia MoE mỏng đến mức sao chép các lớp không chuyên hai lần là khoai tây nhỏ.

Điều quan trọng là dòng chảy về phía trước theo một hướng và dòng chảy ngược theo hướng khác chồng lên nhau chính xác vị trí của các bong bóng theo lịch trình một hướng. Các bong bóng biến mất.

### Lịch trình theo dõi thủ công

Xem xét P = 4 bậc, 8 batches vi mô, chia 4 tiến / 4 lùi. Thời gian di chuyển từ trái sang phải; hàng là thứ hạng thiết bị.

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

Đọc ký hiệu "F4/F5R": thứ hạng 1 đang chạy về phía trước của micro-batch 4 (đi từ trái sang phải trong pipeline) VÀ về phía trước của micro-batch 5 (đi từ phải sang trái) trong cùng một khoảng thời gian. Đó là ý nghĩa của "hai chiều" về mặt hoạt động.

Ở hạng 2, các luồng chéo chồng lên nhau sớm hơn, ở hạng 0 và P-1, chúng trùng lặp mới nhất. Trong giai đoạn giữa ổn định của lịch trình, mọi thứ hạng chạy về phía trước của hướng X chồng lên với hướng ngược của Y. Điện toán đang bận. Các công văn tất cả cho forward pass ẩn bên trong điện toán ngược. Kết hợp tất cả để tất cả ẩn bên trong tính toán chuyển tiếp. Các bong bóng bị ép ra.

### Kế toán bong bóng

Bong bóng pipeline tiêu chuẩn 1F1B (thời gian lãng phí cho mỗi cấp bậc):

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Tinh chỉnh Zero Bubble làm giảm nó nhưng không về không. DualPipe, trong giai đoạn ổn định, có bong bóng bằng không nếu số lượng vi batch chia hết cho 2 lần độ sâu pipeline. Bên ngoài giai đoạn ổn định (khởi động và hồi chiêu), có một số bong bóng nhưng nó không phát triển theo số vi batches - một thuộc tính quan trọng mà bài báo nhấn mạnh.

Về mặt tiếp thị: "không có bong bóng". Về mặt kỹ thuật: bong bóng không phát triển với số lượng batch vi mô. Phân tích tiếp theo của Sea AI Lab (DualPipeV / Cut-in-half) chỉ cho thấy bong bóng hoàn toàn bằng không khi Expert Parallelism không phải là nút thắt cổ chai; với tất cả mọi thứ do EP điều khiển, một số thỏa hiệp về lịch trình luôn hiện diện.

### DualPipeV — sự tinh chỉnh

Sea AI Lab (2025) quan sát thấy rằng việc sao chép parameter 2x là lãng phí khi sự chồng chéo của EP comm không phải là vấn đề. Lịch trình DualPipeV của họ gấp việc tiêm hai chiều thành một lịch trình "hình chữ V" chạy trên một bản sao parameter duy nhất. Bong bóng lớn hơn một chút so với DualPipe, nhưng tiết kiệm bộ nhớ là đáng kể. DeepSeek đã áp dụng DualPipeV trong việc triển khai DualPipe mã nguồn mở của họ như một chế độ EP-off.

Sự đánh đổi:

| Feature | Ống kép | Ống képV | 1F1B | Không bong bóng |
|---------|---------|-----------|------|------------|
| Bản sao tham số trên mỗi thiết bị | 2 | 1 | 1 | 1 |
| Bong bóng so với vi batches | hằng số | Tăng trưởng nhỏ | phát triển | phát triển |
| Chồng chéo điện toán-giao tiếp | đầy đủ | Một phần | Tối thiểu | Một phần |
| Sử dụng khi | EP-nặng MoE | dày đặc hoặc EP-light | Đường cơ sở | bất kỳ pipeline nào |

### Ý nghĩa của việc chạy 14,8T-token

Máy training trước của DeepSeek-V3 đã tiêu thụ 14,8 tấn tokens trên 2.048 H800 GPUs trong khoảng 2,8 triệu GPU giờ. Với 1F1B ngây thơ, họ sẽ mất 12-15% trong số đó cho pipeline bong bóng - 340-420K GPU giờ, đủ để huấn luyện toàn bộ 70B model. DualPipe đã thu hồi hầu hết số đó. Định lượng trực tiếp sự đóng góp là rất khó nếu không có nhật ký nội bộ, nhưng tuyên bố trong bài báo là hơn 95% GPU mức sử dụng trung bình trên training.

Đối với các lần chạy nhỏ hơn (dưới 1k GPUs), DualPipe là quá mức cần thiết - pipeline bong bóng nhỏ hơn so với tổng chi phí và model training dày đặc hiếm khi chạm vào nút thắt cổ chai tất cả. Đối với MoE training biên giới ở quy mô hàng nghìn GPU, nó được yêu cầu một cách hiệu quả.

### Nó nằm ở đâu trong stack

- Bổ sung cho **FSDP** (Giai đoạn 10 · 05). FSDP phân mảnh model parameters trên các cấp bậc; DualPipe lên lịch tính toán giữa các cấp bậc. Chúng kết hợp.
- Tương thích với gradient sharding **ZeRO-3**. Việc ghi sổ kế toán cho bản sao hai bản sao cần hợp tác với gradients phân mảnh của ZeRO.
- Yêu cầu **hạt nhân tất cả tùy chỉnh** được điều chỉnh cho cấu trúc liên kết cụm cụ thể. Các hạt nhân mã nguồn mở của DeepSeek là triển khai tham chiếu.

```figure
expert-capacity
```

## Ứng dụng

`code/main.py` là một trình mô phỏng lịch trình pipeline. Nó mất `(P, n_micro_batches, schedule)` và in việc sử dụng giai đoạn ổn định cho mỗi 1F1B, Zero Bubble, DualPipe và DualPipeV. Nó là một công cụ giảng dạy - các con số khớp với các tuyên bố định tính trong các bài báo, chúng không phải là tuyên bố về production tốc độ đo lường.

Giá trị của trình mô phỏng: chạy nó với số lượng P và micro-batch khác nhau và xem phần bong bóng tăng lên như thế nào đối với 1F1B nhưng không phải DualPipe.

Những điều cần cân nhắc khi tích hợp để chạy training thực sự:

- Chọn độ sâu song song pipeline chia rõ ràng thành số batch vi mô của bạn.
- Đảm bảo lưới song song chuyên gia của bạn hỗ trợ tất cả hai chiều. Nhân của DeepSeek là tài liệu tham khảo.
- Dự kiến sẽ đốt cháy một tuần thời gian gỡ lỗi trên chính lịch trình lần đầu tiên. Việc ghi sổ kế toán rất phức tạp.
- Theo dõi việc sử dụng GPU trên mỗi cấp bậc, không chỉ tổng hợp. Lợi ích của DualPipe đến từ việc thắt chặt những người đi lạc.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-dualpipe-planner.md`. Với đặc tả cụm training (số lượng GPU, cấu trúc liên kết, kết nối model hình dạng), nó đề xuất chiến lược song song pipeline, thuật toán lập lịch sử dụng và phân số bong bóng dự kiến ở thang đích.

## Bài tập

1. Chạy `code/main.py` trên `(P=8, micro_batches=16, schedule=dualpipe)` và `(P=8, micro_batches=16, schedule=1f1b)`. Tính toán chênh lệch mức sử dụng GPU và biểu thị dưới dạng thu hồi GPU giờ trên một triệu tokens training.

2. Phác thảo bảng lịch trình cho `(P=4, micro_batches=8, schedule=dualpipe)` bằng tay. Đánh dấu từng khe thời gian bằng ID và hướng batch vi mô. Xác định khoảng thời gian đầu tiên không có bong bóng.

3. Đọc Hình 5 của báo cáo kỹ thuật DeepSeek-V3 (arXiv:2412.19437). Xác định cửa sổ chồng chéo cho việc gửi tất cả bên trong một đoạn chuyển tiếp DualPipe. Giải thích cách lịch trình điện toán ẩn nó.

4. Tính chi phí parameter gấp 2 lần của DualPipe cho một model dày đặc 70B với P=8 giai đoạn pipeline và MoE model 671B với P=16 giai đoạn pipeline. Cho thấy lý do tại sao chi phí của trường hợp MoE nhỏ hơn theo tỷ lệ (hầu hết parameters là chuyên gia, được phân mảnh trên một nhóm EP lớn).

5. So sánh DualPipe với Chimera (một bộ lập lịch hai chiều cạnh tranh từ năm 2021). Xác định hai thuộc tính cụ thể mà DualPipe đã thêm vào mà Chimera không có, sử dụng Mục 3.4 của bài báo làm tài liệu tham khảo.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Pipeline bong bóng | "Thời gian nhàn rỗi trên mỗi cấp bậc" | GPU chu kỳ bị lãng phí vì một giai đoạn pipeline đang chờ đầu vào hoặc gradient |
| 1F1B | "Lịch trình pipeline mặc định" | Một lịch trình xen kẽ tiến / lùi; nhịp đập DualPipe cơ bản |
| Không bong bóng | "Phòng thí nghiệm AI biển 2023" | Chia ngược thành B (gradient đầu vào) và W (trọng lượng gradient); gần như thắt chặt hoàn toàn pipeline |
| Ống kép | "Lịch trình DeepSeek-V3" | pipeline hai chiều + tính toán-giao tiếp chồng chéo; bong bóng không phát triển với số lượng vi batch |
| Ống képV | "Cắt đôi" | Tinh chỉnh hình chữ V giúp giảm khả năng sao chép parameter 2x với chi phí là bong bóng lớn hơn một chút |
| Khối | "Đơn vị pipeline công việc" | Chuyển tiếp hoặc backward pass một micro-batch qua một pipeline stage |
| Công văn tất cả | "Gửi tokens cho chuyên gia" | Giao tiếp giữa các nút định tuyến tokens đến các chuyên gia MoE được chỉ định của họ |
| Kết hợp tất cả | "Mang lại kết quả đầu ra của chuyên gia" | Giao tiếp giữa các nút thu thập kết quả đầu ra của chuyên gia sau MLP |
| Chuyên gia song song (EP) | "Các chuyên gia trên khắp GPUs" | Các mảnh vỡ MoE các chuyên gia trên các cấp bậc khác nhau GPUs nắm giữ các chuyên gia khác nhau |
| Pipeline Độ song song (PP) | "Các lớp trên GPUs" | Các mảnh model lớp trên các cấp bậc; kích thước Lịch trình DualPipe |
| Phân số bong bóng | "Lãng phí thời gian GPU" | (bubble_time / total_time); phân số DualPipe hướng về số không |

## Đọc thêm

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — tham chiếu DualPipe chính
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — triển khai tham chiếu mã nguồn mở, bao gồm chế độ DualPipeV (Cắt đôi)
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — tiền thân của Zero Bubble
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) - phân tích DualPipeV đã thông báo cho chế độ EP-off của DeepSeek
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) - lịch trình 1F1B mà DualPipe so sánh với
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — bài toán pipeline song song và bong bóng ban đầu
