# Benchmarks: WebArena và OSWorld

> WebArena kiểm tra khả năng agent web trên bốn ứng dụng tự lưu trữ. OSWorld kiểm tra khả năng agent trên máy tính để bàn trên Ubuntu, Windows macOS. Khi phát hành (2023–2024), cả hai đều cho thấy khoảng cách lớn giữa class agents tốt nhất và con người. Khoảng cách đang thu hẹp; Các chế độ thất bại không thay đổi.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 19 (SWE-băng ghế dự bị, GAIA)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Mô tả bốn ứng dụng tự lưu trữ của WebArena và lý do tại sao đánh giá dựa trên thực thi lại quan trọng.
- Giải thích lý do tại sao OSWorld sử dụng ảnh chụp màn hình hệ điều hành thực thay vì APIs trợ năng.
- Đặt tên cho hai chế độ lỗi chính của OSWorld: grounding GUI và kiến thức vận hành.
- Tóm tắt những gì OSWorld-G và OSWorld-Human thêm vào đầu benchmark cơ sở.

## Vấn đề

Tổng quát agents có thể gọi các công cụ. Họ có thể điều khiển trình duyệt qua 20 cú nhấp chuột để hoàn tất thanh toán mua sắm không? Họ có thể định cấu hình hộp Linux chỉ bằng bàn phím và chuột không? Đây là những câu hỏi mà WebArena và OSWorld trả lời.

## Khái niệm

### WebArena (Chu và cộng sự, ICLR 2024)

- 812 nhiệm vụ dài trên bốn web apps tự lưu trữ: trang web mua sắm, diễn đàn, công cụ phát triển giống GitLab, CMS kinh doanh.
- Cộng với các tiện ích: bản đồ, máy tính, bảng cào.
- Đánh giá dựa trên việc thực hiện thông qua APIs phòng tập thể dục - đơn đặt hàng đã được đặt, vấn đề đã đóng chưa, trang CMS có được cập nhật không?
- Khi phát hành: GPT-4 agent tốt nhất đạt 14,41% thành công so với con người 78,24%.

Khung hình tự lưu trữ rất quan trọng - benchmark không bị bong tróc vì các ứng dụng mục tiêu được ghim và có thể tái tạo.

### Tiện ích mở rộng

- **VisualWebArena** — các nhiệm vụ dựa trên hình ảnh trong đó thành công phụ thuộc vào việc diễn giải hình ảnh (ảnh chụp màn hình là quan sát class đầu tiên).
- **TheAgentCompany** (Tháng 12 năm 2024) — thêm thiết bị đầu cuối + mã hóa; giống như một môi trường làm việc từ xa thực sự.

### OSWorld (Xie và cộng sự, NeurIPS 2024)

- 369 tác vụ máy tính thực tế trên Ubuntu, Windows, macOS.
- Điều khiển bàn phím và chuột dạng tự do của các ứng dụng thực.
- 1920×1080 ảnh chụp màn hình như quan sát.
- Khi phát hành: tốt nhất model 12,24% so với con người 72,36%.

### Chế độ lỗi chính

1. **GUI grounding.** Ánh xạ phần tử Pixel →. Models phải vật lộn để bản địa hóa các yếu tố giao diện người dùng một cách đáng tin cậy vào năm 1920×1080.
2. **Kiến thức vận hành.** Menu nào có cài đặt, phím tắt nào, ngăn tùy chọn nào. Đuôi tri thức mà con người xây dựng qua nhiều năm.

### Theo dõi

- **OSWorld-G** — Bộ grounding 564 mẫu + bộ training Jedi. Phân tách grounding từ lập kế hoạch để bạn có thể đo lường chúng riêng biệt.
- **OSWorld-Human** — quỹ đạo hành động vàng được sắp xếp thủ công. Hiển thị agents hàng đầu sử dụng nhiều bước hơn 1.4-2.7 lần so với mức cần thiết (khoảng cách quỹ đạo-hiệu quả).

### Tại sao điều này lại quan trọng

Claude sử dụng máy tính, OpenAI CUA, Gemini 2.5 Sử dụng máy tính (Bài 21) tất cả đều tập luyện trên khối lượng công việc được định hình bởi WebArena và OSWorld. Các benchmarks là mục tiêu; production models là câu trả lời shipped.

### Điểm chuẩn bị lỗi ở đâu

- **Đánh giá chỉ ảnh chụp màn hình.** OSWorld dựa trên ảnh chụp màn hình; Đánh giá một agent sử dụng DOM hoặc APIs trợ năng trên OSWorld bỏ lỡ thử thách grounding.
- **Bỏ qua độ dài quỹ đạo.** Chỉ ghi điểm tỷ lệ thành công bỏ lỡ 1,4-2,7x bước kém hiệu quả OSWorld-Human surfaces.
- **Ứng dụng tự lưu trữ cũ.** Ứng dụng của WebArena ghim các phiên bản cụ thể; cập nhật mà không cần quản lý lại sẽ phá vỡ khả năng so sánh.

## Tự xây dựng

`code/main.py` triển khai agent harness web đồ chơi:

- Một máy trạng thái "ứng dụng mua sắm" tối thiểu: list_items, add_to_cart, thanh toán.
- Quỹ đạo vàng cho 3 nhiệm vụ.
- Một agent có kịch bản cố gắng thực hiện từng nhiệm vụ.
- Công cụ đánh giá dựa trên thực thi (kiểm tra trạng thái) và chỉ số hiệu quả quỹ đạo (số bước so với vàng).

Chạy nó:

```
python3 code/main.py
```

Đầu ra: tỷ lệ thành công trên mỗi tác vụ và hiệu quả quỹ đạo, phản ánh phương pháp luận của OSWorld-Human.

## Ứng dụng

- **WebArena đã xác minh** tự lưu trữ trên một cụm nội bộ để đánh giá liên tục.
- **OSWorld** trong nhóm máy ảo dành cho agents máy tính để bàn.
- **agents sử dụng máy tính **(Bài 21) - Claude, OpenAI CUA, Gemini - tất cả đều được huấn luyện về khối lượng công việc như thế này.
- **Dòng sản phẩm của riêng bạn** — nắm bắt quỹ đạo vàng cho 20 nhiệm vụ hàng đầu của bạn; chạy agents chống lại họ hàng tuần.

## Sản phẩm bàn giao

`outputs/skill-web-desktop-harness.md` xây dựng một web/desktop agent harness với chỉ số hiệu quả quỹ đạo và đánh giá dựa trên thực thi.

## Bài tập

1. Mở rộng harness đồ chơi bằng ứng dụng thứ hai (diễn đàn). Viết 3 nhiệm vụ cộng với quỹ đạo vàng.
2. Thêm báo cáo hiệu quả quỹ đạo cho mỗi nhiệm vụ. Trên đồ chơi của bạn, agent 1x, 2x hay 3x trên vàng?
3. Triển khai một công cụ "đánh lạc hướng" - một công cụ mà quỹ đạo vàng không bao giờ sử dụng. Các agent theo kịch bản có bị cám dỗ không?
4. Đọc OSWorld-G. Làm thế nào để bạn phân biệt thất bại grounding với thất bại trong kế hoạch trong đánh giá của chính bạn?
5. Đọc ứng dụng của WebArena README. Điều gì sẽ xảy ra khi bạn nâng cấp một trong các phiên bản ứng dụng đã ghim?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Đấu trường web | "Web agent benchmark" | 812 tác vụ trên 4 ứng dụng tự lưu trữ; Đánh giá kiểu phòng tập thể dục |
| Đấu trường VisualWebArena | "Đấu trường web trực quan" | WebArena có cơ sở trực quan; Ảnh chụp màn hình là quan sát |
| Hệ điều hànhTrang web | "Máy tính để bàn agent benchmark" | 369 nhiệm vụ trên Ubuntu/Windows/macOS thực |
| GUI grounding | "Ánh xạ pixel-to-phần tử" | Model bản địa hóa các thành phần giao diện người dùng trong 1920x1080 |
| Kiến thức vận hành | "Bí quyết hệ điều hành" | Menu nào, phím tắt nào, ngăn tùy chọn nào |
| Hệ điều hànhWorld-G | "Grounding dãy phòng" | 564 mẫu chỉ dành cho grounding + bộ training |
| OSWorld-Con người | "Quỹ đạo vàng" | Trình tự hành động thủ công của chuyên gia để đo lường hiệu quả |
| Hiệu quả quỹ đạo | "Bước qua vàng" | Số bước Agent chia cho mức tối thiểu của con người |

## Đọc thêm

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — benchmark web bốn ứng dụng
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — benchmark máy tính để bàn đa hệ điều hành
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — khả năng hình benchmark của Claude
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) - Số OSWorld và WebArena
