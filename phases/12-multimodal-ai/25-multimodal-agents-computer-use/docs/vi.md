# Agents đa phương thức và sử dụng máy tính (Capstone)

> Sản phẩm biên giới năm 2026 là một agent đa phương thức đọc ảnh chụp màn hình, nhấp vào nút, điều hướng giao diện người dùng web, điền vào biểu mẫu và hoàn thành quy trình làm việc từ đầu đến cuối. SeeClick và CogAgent (2024) đã chứng minh GUI-grounding primitive. Ferret-UI đã thêm thiết bị di động. ChartAgent đã giới thiệu công cụ trực quan sử dụng cho biểu đồ. VisualWebArena và AgentVista (2026) là những benchmarks rượt đuổi biên giới - và thậm chí Gemini 3 Pro và Claude Opus 4.7 đạt điểm ~30% cho các nhiệm vụ khó khăn của AgentVista. Capstone này tập hợp mọi thread của Giai đoạn 12: nhận thức (VLM độ phân giải cao), suy luận (LLM với việc sử dụng công cụ), grounding (đầu ra tọa độ), bộ nhớ chân trời dài và đánh giá.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (stdlib, action schema + agent loop skeleton)
**Kiến thức tiên quyết:** Giai đoạn 12 · 05 (LLaVA), Giai đoạn 12 · 09 (Qwen-VL JSON), Giai đoạn 14 (Agent Engineering)
**Thời lượng:** ~240 phút

## Mục tiêu học tập

- Thiết kế vòng lặp agent đa phương thức: nhận thức lý do → → hành động → quan sát → lặp lại.
- Xây dựng GUI grounding schema đầu ra (nhấp vào tọa độ, nhập văn bản, cuộn, kéo) mà VLM có thể phát ra dưới dạng JSON.
- So sánh agents chỉ có ảnh chụp màn hình so với agents cây trợ năng và agents kết hợp.
- Thiết lập đánh giá agent benchmark đa phương thức trên một lát cắt VisualWebArena nhỏ.

## Vấn đề

Quy trình làm việc trên trang web đặt chỗ: "tìm cho tôi một chuyến bay đến Tokyo vào ngày 15 tháng 4, chỗ ngồi gần lối đi dưới 800 đô la, đặt trước."

Một agent đa phương thức cần:

1. Chụp ảnh màn hình của trình duyệt.
2. Phân tích cú pháp ảnh chụp màn hình + URL + mục tiêu thành một kế hoạch.
3. Phát ra một hành động có cấu trúc: nhấp vào (ở x,y), nhập "Tokyo" (ở phần tử E), cuộn xuống, chọn (nút radio).
4. Áp dụng hành động cho trình duyệt.
5. Quan sát trạng thái mới (ảnh chụp màn hình tiếp theo).
6. Lặp lại cho đến khi hoàn thành nhiệm vụ.

Mỗi bước là một cuộc gọi VLM đa phương thức. Đầu ra VLM phải có thể phân tích cú pháp JSON. Lỗi kết hợp qua các bước, vì vậy việc khôi phục rất quan trọng.

## Khái niệm

### GUI grounding - primitive

GUI grounding là: được cung cấp ảnh chụp màn hình và hướng dẫn ngôn ngữ tự nhiên, xuất tọa độ (x, y) để nhấp vào (hoặc hành động khác).

SeeClick (arXiv:2401.10935) là kết quả mở đầu tiên trên quy mô lớn: fine-tune VLM trên dữ liệu GUI tổng hợp + thực, tọa độ đầu ra dưới dạng tokens văn bản thuần túy. Tác phẩm.

CogAgent (arXiv:2312.08914) đã thêm mã hóa độ phân giải cao 1120x1120 cho giao diện người dùng dày đặc. Điểm: ~84% trên điều hướng web.

Ferret-UI (arXiv:2404.05719) tập trung vào giao diện người dùng di động, tích hợp với dữ liệu trợ năng iOS.

Định dạng đầu ra thường được JSON:

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` giúp khôi phục: nếu tọa độ trôi dạt giữa các ảnh chụp màn hình, gợi ý ngữ nghĩa cho phép hệ thống nối đất lại.

### Hành động schemas

Một schema hành động điển hình có 6-10 loại hành động:

- `click`: (x, y)
- `type`: (văn bản, x?, y?)
- `scroll`: (hướng, số lượng)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (URL)
- `wait`: (mili giây)
- `done`: (thành công, giải thích)

agent phát ra một hành động trên mỗi bước. Trình bao bọc trình duyệt thực thi và trả về trạng thái mới.

### Chỉ ảnh chụp màn hình so với cây trợ năng

Hai chế độ đầu vào:

- Chỉ ảnh chụp màn hình: hình ảnh đầy đủ, không có thông tin cấu trúc. Tổng quát nhất; hoạt động trên mọi ứng dụng.
- Cây trợ năng: thông tin trợ năng DOM / iOS có cấu trúc. Đáng tin cậy hơn nhiều cho grounding; làm việc ở nơi có cây.
- Hybrid: cả hai, với cây như một nền tảng đáng tin cậy cho các hành động nguyên tử và ảnh chụp màn hình cho ngữ cảnh ngữ nghĩa.

Production agents sử dụng hybrid khi có thể. Tự động hóa trình duyệt (Selenium + trợ năng) luôn có cây; Các ứng dụng dành cho máy tính để bàn đôi khi có.

### Bộ nhớ đường chân trời dài

Quy trình làm việc gồm 20 bước tạo 20 ảnh chụp màn hình. Bối cảnh của VLM lấp đầy nhanh chóng. Ba chiến lược nén:

- Chuỗi tóm tắt: sau mỗi 5 bước, tóm tắt những gì đã xảy ra, bỏ ảnh chụp màn hình cũ.
- Bỏ qua khung hình: giữ ảnh chụp màn hình đầu tiên, cuối cùng và mọi ảnh chụp màn hình thứ 3.
- Nhật ký được ghi lại bởi công cụ: thực hiện các hành động, giữ nhật ký văn bản về những gì đã được thực hiện; Đừng xem lại ảnh chụp màn hình cũ.

API sử dụng máy tính của Claude sử dụng mẫu nhật ký. Đơn giản hơn, đáng tin cậy hơn.

### Sử dụng công cụ trực quan

ChartAgent (arXiv:2510.04514) giới thiệu cách sử dụng công cụ trực quan để hiểu biểu đồ: cắt, thu phóng, OCR, gọi phát hiện bên ngoài. agent có thể xuất ra "cắt theo vùng (100, 200, 300, 400) sau đó gọi OCR" dưới dạng lệnh gọi công cụ. Công cụ trả về văn bản; VLM tiếp tục lý luận.

Mẫu này khái quát hóa: prompting tập hợp dấu, chú thích vùng và các công cụ phát hiện bên ngoài đều phù hợp với cùng một schema "xuất lệnh gọi công cụ, nhận phản hồi có cấu trúc".

### The 2026 benchmarks

- ScreenSpot-Pro (bằng tiếng Anh). GUI grounding trên ~1k ảnh chụp màn hình web. Mở SOTA Qwen2.5-VL-72B ~85%. Biên giới ~90%.
- VisualWebArena (bằng tiếng Anh). Các tác vụ web đầu cuối (cửa hàng, diễn đàn, rao vặt). Mở SOTA ~20%. Gemini 3 Pro ~27%.
- AgentVista (arXiv:2602.23166). Năm 2026 khó khăn nhất benchmark. Quy trình làm việc thực tế trên 12 miền. Frontier models điểm 27-40%; mở models 10-20%.
- WebArena / WebShop (bằng tiếng Anh). benchmarks lớn tuổi; bão hòa bởi biên giới.

### Tại sao nó vẫn khó khăn

Agent tắc nghẽn hiệu suất:

1. Hình ảnh grounding ở quy mô nhỏ. "Nhấp vào dấu X nhỏ" thường không thành công ở độ phân giải di động.
2. Lập kế hoạch dài hạn. Sau 10 hành động, agent trôi khỏi khung thành.
3. Khôi phục lỗi. Khi một cú nhấp chuột không thành công (nút sai), việc phát hiện + khôi phục hiếm khi là dữ liệu được huấn luyện.
4. Ngữ cảnh chéo trang. Nhảy giữa các tab hoặc biểu mẫu dài sẽ mất trạng thái.

Hướng nghiên cứu: kiến trúc bộ nhớ, lập kế hoạch lại rõ ràng, xác minh đa phương thức (đối sánh ảnh chụp màn hình để hành động thành công).

### Capstone xây dựng nó

Nhiệm vụ capstone: xây dựng một agent sử dụng máy tính:

1. Đọc HTML + ảnh chụp màn hình của trang mô phỏng trang web đặt phòng.
2. Lập kế hoạch trình tự nhiều bước: tìm kiếm → chọn → điền vào biểu mẫu → gửi.
3. Phát ra các hành động JSON phù hợp với schema hành động.
4. Đánh giá trên một lát cắt 10 nhiệm vụ cố định.

Bài học cung cấp mã giàn giáo dễ dàng mở rộng vào trình duyệt thực.

## Ứng dụng

`code/main.py` là giàn giáo capstone:

- Hành động schema JSON định nghĩa (10 hành động).
- Trạng thái trình duyệt giả dưới dạng dict.
- Agent khung vòng lặp: nhận trạng thái, phát hành động, áp dụng, vòng lặp.
- benchmark nhỏ 10 tác vụ (trang tổng hợp) để đo lường tỷ lệ thành công từ đầu đến cuối.
- Khôi phục lỗi hook khi một hành động không thành công.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-multimodal-agent-designer.md`. Cho một sản phẩm sử dụng máy tính (miền, bộ hành động, mục tiêu đánh giá), thiết kế vòng lặp agent đầy đủ, chiến lược bộ nhớ, chế độ grounding và điểm benchmark dự kiến.

## Bài tập

1. Mở rộng schema hành động bằng công cụ `screenshot_region` (cắt + thu phóng). Những nhiệm vụ nào được hưởng lợi?

2. Đọc AgentVista (arXiv:2602.23166). Mô tả loại nhiệm vụ khó nhất và lý do tại sao models biên giới vẫn thất bại.

3. Nén bộ nhớ đường chân trời dài: thiết kế chuỗi tóm tắt với ≤4 ảnh chụp màn hình được lưu trực tiếp, bất kỳ số nào được ghi lại.

4. Xây dựng hook khôi phục lỗi: khi hành động không thành công (không tìm thấy nút), agent làm gì tiếp theo?

5. So sánh Claude 4.7 chỉ có ảnh chụp màn hình với ảnh chụp màn hình kết hợp + cây trợ năng Qwen2.5-VL trên 10 tác vụ web. Cái nào chiến thắng trong những nhiệm vụ nào?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| GUI grounding | "Nhấp vào tọa độ" | Model đầu ra (x,y) cho mục tiêu của lệnh trên ảnh chụp màn hình |
| Hành động schema | "Định nghĩa công cụ" | JSON mô tả các hành động hợp lệ (nhấp, nhập, cuộn, kéo) |
| Cây trợ năng | "DOM có cấu trúc" | Hệ thống phân cấp giao diện người dùng có thể đọc được bằng máy từ browser/iOS APIs |
| agent lai | "Ảnh chụp màn hình + cây" | Sử dụng cả hình ảnh và thông tin có cấu trúc; đáng tin cậy hơn một trong hai |
| Sử dụng công cụ trực quan | "Zoom/crop/detect" | Agent gọi các công cụ thị giác bên ngoài (OCR, phát hiện) giữa kế hoạch |
| Chuỗi tóm tắt | "Nén bộ nhớ" | Tóm tắt văn bản định kỳ thay thế lịch sử ảnh chụp màn hình dài |
| Đấu trường VisualWebArena | "Băng ghế web E2E" | benchmark 2024 cho các tác vụ web đầu cuối |
| Đặc vụ Vista | "Băng ghế cứng năm 2026" | Quy trình làm việc thực tế 12 miền; thậm chí Gemini 3 điểm Pro ~30% |

## Đọc thêm

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
