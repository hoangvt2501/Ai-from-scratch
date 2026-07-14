# VLA hiện thân: RT-2, OpenVLA, π0, GR00T

> Lần đầu tiên một model đọc công thức từ một trang web và thực hiện nó trong robot nhà bếp là RT-2 (Google DeepMind, Tháng Bảy 2023). RT-2 đã rời rạc các hành động dưới dạng tokens văn bản, đồng fine-tuned VLM dữ liệu web cộng với dữ liệu hành động của robot và chứng minh rằng kiến thức ngôn ngữ thị giác quy mô web chuyển sang điều khiển robot. OpenVLA (tháng 6 năm 2024) shipped tham chiếu 7B mở. Dòng π0 của Physical Intelligence (2024-2025) đã bổ sung các chuyên gia hành động phù hợp với dòng chảy. GR00T N1 của NVIDIA (tháng 3 năm 2025) cung cấp điều khiển hệ thống kép (Hệ thống 1 / Hệ thống 2) cho robot hình người trên quy mô lớn. primitive VLA - thị giác-ngôn ngữ-hành động, một model duy nhất nhìn, đọc và hành động - là cầu nối giữa models hiểu biết của giai đoạn này và các hệ thống tự trị trong Giai đoạn 15.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, action tokenizer + VLA inference skeleton)
**Kiến thức tiên quyết:** Giai đoạn 12 · 05 (LLaVA), Giai đoạn 15 (Hệ thống tự trị, tham chiếu)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Mô tả tokenization hành động: mã hóa thùng rời rạc (RT-2), tokens hành động hiệu quả NHANH, hành động khớp luồng liên tục (π0).
- Giải thích lý do tại sao co-fine-tuning trên web + dữ liệu robot bảo toàn việc chuyển giao kiến thức chung cho các nhiệm vụ mới.
- So sánh OpenVLA (mở 7B Llama+VLM), π0 (khớp luồng) và GR00T N1 (hệ thống kép) trên cùng một tác vụ rô-bốt.
- Đặt tên cho Open X-Embodiment dataset và vai trò của nó như kho dữ liệu training RT-X.

## Vấn đề

Một robot làm việc nhà từ các hướng dẫn ngôn ngữ tự nhiên đã trở thành mục tiêu nghiên cứu từ những năm 1970. Câu trả lời của những năm 2020: model tầm nhìn-ngôn ngữ-hành động (VLA). Kiến trúc VLM tương tự được sử dụng cho VQA, nhưng đầu ra là các hành động (mô-men xoắn chung, tư thế hiệu ứng cuối, lệnh rời rạc) thay vì văn bản.

Những thách thức cụ thể đối với VLA:

1. Không gian hành động liên tục (góc khớp, lực) và high-dimensional (cánh tay 7-DOF + bộ kẹp 3-DOF = 10 độ mờ ở 30 Hz).
2. Dữ liệu training dành riêng cho robot rất khan hiếm. Open X-Embodiment có ~1 triệu quỹ đạo; hình ảnh văn bản web là 5B +.
3. Kiểm soát tần số rất quan trọng. Vòng điều khiển 30 Hz có nghĩa là ngân sách 33ms cho mỗi hành động.
4. Sự an toàn. Một hành động sai sẽ làm hỏng phần cứng, con người hoặc tài sản.

## Khái niệm

### tokenization hành động (RT-2)

Thủ thuật của RT-2: biểu diễn mỗi mục tiêu chung dưới dạng một token văn bản lượng tử hóa. Rời rạc phạm vi [-1, 1] chuẩn hóa thành 256 thùng, ánh xạ mỗi thùng với một ID từ vựng. Hành động 10-DOF trở thành 10 tokens ở mỗi bước điều khiển.

Đồng fine-tune VLM PaLM-X trên hỗn hợp:

- Các cặp hình ảnh-văn bản trên web (chú thích, VQA).
- Trình diễn robot, hành động như tokens.

model thấy "nhặt khối màu đỏ" (ngôn ngữ) → hình ảnh (tầm nhìn) → chuỗi hành động 10 token (mục tiêu chung rời rạc). Web pretraining duy trì chuyển giao kiến thức chung: RT-2 có thể theo sau "di chuyển về phía đối tượng chuyển động nhanh" mặc dù "di chuyển nhanh" không có trong dữ liệu training.

Inference ở 3-5 Hz trong bài RT-2, bị giới hạn bởi VLM giải mã tự hồi quy.

### OpenVLA — tham chiếu 7B mở

OpenVLA (Kim và cộng sự, tháng 6 năm 2024) là RT-2 tương đương với trọng lượng mở. Đường trục Llama 7B, encoder tầm nhìn kép DINOv2 + SigLIP, hành động tokenization hơn 256 thùng.

Được huấn luyện về Open X-Embodiment (970k quỹ đạo trên 22 robot). Ships với sự hỗ trợ LoRA fine-tuning để thích ứng với robot mới.

Inference: 4-5 Hz trên A100 có quantization. Đủ nhanh để thao tác chậm, không phải để điều khiển tần số cao.

### FAST tokenizer — giải mã hành động nhanh hơn

Pertsch et al. (2024) chỉ ra rằng tokenization thùng rời rạc không hiệu quả - hầu hết các hành động tập trung trong một vùng nhỏ của không gian thùng. FAST (Trình tự hành động miền tần số Tokenizer) nén các chuỗi hành động thông qua DCT và lượng tử hóa các hệ số.

Quỹ đạo hành động 30 bước trở thành ~10 tokens NHANH thay vì 300 tokens thùng rời rạc. Inference tăng tốc gấp 3-5 lần mà không cần loss chất lượng.

### π0 và các hành động khớp luồng

π0 của Physical Intelligence (Black et al., tháng 10 năm 2024) thay thế tokens hành động rời rạc bằng một chuyên gia hành động khớp luồng:

- Một hành động nhỏ transformer đọc các trạng thái ẩn của VLM và xuất ra chuỗi hành động 50 bước liên tục thông qua luồng được chỉnh lưu.
- Đầu hành động huấn luyện với loss phù hợp với dòng chảy; VLM pretraining không thay đổi.
- Inference: chuỗi hành động đầy đủ được phát ra trong ~5 bước khử nhiễu, điều khiển hiệu quả 50 Hz.

Tuyên bố của π0: đánh bại OpenVLA và Octo trên một loạt các tác vụ thao tác. Công thức hoạt động liên tục duy trì sự mượt mà mà sự rời rạc phá hủy.

π0.5 và π0-FAST là các nâng cấp gia tăng. π0-FAST kết hợp tokenization FAST với kết hợp dòng chảy.

### GR00T N1 — hệ thống kép cho hình người

GR00T N1 của NVIDIA (tháng 3 năm 2025) được chế tạo cho robot hình người (>30 DOF, toàn thân):

- Hệ thống 2: một cảnh đọc VLM lớn + hướng dẫn, tạo ra các mục tiêu phụ cấp cao ở ~1 Hz.
- Hệ thống 1: một đầu hành động nhỏ transformer tạo ra các lệnh chung 50-100 Hz mức thấp có điều kiện trên các mục tiêu phụ.

Sự phân chia ánh xạ đến tư duy nhanh và chậm của Kahneman: Hệ thống 2 lập kế hoạch, Hệ thống 1 hành động. Lợi ích: lập kế hoạch VLM chậm không cản trở việc kiểm soát nhanh; Hệ thống 1 vẫn nhỏ vì độ trễ.

GR00T N1.7 (cuối năm 2025) cải thiện khả năng mở rộng dữ liệu. GR00T tinh chỉnh với dữ liệu từ mô phỏng thành thực từ Omniverse.

### Mở X-Embodiment

Dữ liệu training. RT-X (tháng 10 năm 2023) đã lắp ráp 22 datasets bao gồm 1M quỹ đạo trên 22 robot. Open X-Embodiment là kho dữ liệu mà mọi người sử dụng:

- ALOHA / Bridge V2 / Droid / RT-2 Nhà bếp / Bảng ngôn ngữ.
- Mỗi mẫu: (trạng thái robot, chế độ xem camera, hướng dẫn, trình tự hành động).
- Training vệ sinh: thống nhất không gian hành động, bình thường hóa phạm vi khớp, thay đổi kích thước camera.

OpenVLA và π0 huấn luyện trên Open X-Embodiment. Khoảng cách miền với bất kỳ robot cụ thể nào được thu hẹp bởi LoRA fine-tuning trên 100-1000 bản demo nhiệm vụ cụ thể.

### Đồng fine-tuning so với chỉ robot

Co-fine-tuning kết hợp dữ liệu VQA web với quỹ đạo rô-bốt. Tỷ lệ quan trọng: quá nhiều VQA và model quên hành động; Quá nhiều dữ liệu robot và người model mất đi kiến thức chung.

Tỷ lệ của RT-2: ~1: 1. OpenVLA: ~0,5:1 web-to-robot. π0: Tương tự. Tỷ lệ chính xác là một hyperparameter để điều chỉnh cho mỗi kích thước dataset.

training chỉ dành cho rô-bốt tạo ra các models cụ thể theo nhiệm vụ không thành công trong các hướng dẫn ngoài phân phối. Đồng fine-tuning là sự khác biệt giữa "nhặt khối lập phương màu đỏ (trong bản demo)" và "nhặt vật thể lớn thứ ba từ bên trái (cụm từ mới lạ)".

### Giới hạn an toàn và hành động

Mỗi production VLA ships với:

- Giới hạn khớp cứng (không thể mô-men xoắn vượt quá thông số kỹ thuật).
- Giới hạn vận tốc (cắt mềm).
- Giới hạn không gian làm việc (end-effector không thể rời khỏi bảng).
- Phê duyệt con người trong vòng lặp cho các nhiệm vụ mới.

Chúng nằm bên ngoài VLA dưới dạng kiểm tra lớp điều khiển. Đầu ra của VLA là một gợi ý, không phải một lệnh.

## Ứng dụng

`code/main.py`:

- Thực hiện hành động 256 thùng tokenization và khử tokenization.
- Phác thảo một tokenizer NHANH CHÓNG dựa trên DCT + quantization.
- So sánh số lượng token trên mỗi bước hành động (thùng rác rời rạc, FAST, luồng liên tục).
- In tóm tắt dòng RT-2 → OpenVLA → π0 → GR00T.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-vla-action-format-picker.md`. Được giao nhiệm vụ rô-bốt (thao tác, điều hướng, toàn bộ cơ thể hình người), chọn giữa thùng rời rạc + RT-2, FAST + OpenVLA, khớp luồng + π0 hoặc hệ thống kép + GR00T.

## Bài tập

1. Cánh tay 10-DOF ở tốc độ điều khiển 30 Hz. tokenization thùng rời rạc ở 256 thùng phát ra bao nhiêu tokens mỗi giây? Một VLM 7B có thể theo kịp không?

2. FAST tokenization nén quỹ đạo 30 bước xuống ~10 tokens. Người dùng mất gì nếu quỹ đạo có chuyển động tần số cao (ví dụ: đánh trống)?

3. Đầu khớp dòng chảy của π0 khử nhiễu trong ~5 bước. So sánh thông lượng với giải mã tự hồi quy của OpenVLA ở 4-5 Hz.

4. Hệ thống 1 / Hệ thống 2 của GR00T chia bản đồ thành Kahneman. Đề xuất một phân chia khác (Hệ thống 3?) có thể giúp đi bộ hai chân.

5. Đọc Mở X-Embodiment Phần 4 về quản lý dataset. Đặt tên cho ba quy tắc quản lý ngăn chặn rò rỉ tên miền.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| VLA | "Tầm nhìn-ngôn ngữ-hành động" | Model nhận hình ảnh + lệnh và xuất ra các lệnh hành động |
| Hành động tokenization | "Thùng rác rời rạc" | Lượng tử hóa các mục tiêu chung liên tục thành 256 thùng trên mỗi độ sáng, mỗi thùng là ID từ vựng |
| tokenizer NHANH | "Hành động tần số tokens" | DCT + lượng tử hóa để nén quỹ đạo 30 bước xuống ~10 tokens |
| Đồng fine-tune | "Kết hợp web + robot" | Huấn luyện dữ liệu VQA trên web cùng với trình diễn robot để bảo tồn kiến thức chung |
| Đầu hành động khớp luồng | "Đầu ra liên tục π0" | transformer nhỏ xuất ra chuỗi hành động 50 bước thông qua luồng chỉnh lưu |
| Hệ thống 1 / Hệ thống 2 | "Điều khiển hệ thống kép" | Kế hoạch VLM lớn chậm, đầu hành động nhỏ hành động nhanh chóng; Mẫu GR00T |
| Mở X-Embodiment | "RT-X dataset" | dataset rô-bốt chéo quỹ đạo 1M; Kho dữ liệu training |

## Đọc thêm

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
