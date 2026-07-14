# Khối bộ nhớ và Điện toán thời gian ngủ (Letta)

> MemGPT trở thành Letta vào năm 2024. Sự phát triển năm 2026 bổ sung hai ý tưởng: khối bộ nhớ chức năng rời rạc mà model có thể chỉnh sửa trực tiếp và agent thời gian ngủ hợp nhất bộ nhớ không đồng bộ trong khi agent chính không hoạt động. Đây là cách bạn mở rộng bộ nhớ vượt ra ngoài một cuộc trò chuyện.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 07 (MemGPT)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Kể tên ba cấp bộ nhớ mà Letta sử dụng (lõi, recall, lưu trữ) và vai trò của mỗi cấp.
- Giải thích mẫu khối bộ nhớ: Khối con người, khối Persona và khối do người dùng xác định là các đối tượng được nhập class đầu tiên.
- Mô tả điện toán thời gian ngủ là gì, tại sao nó nằm ngoài đường dẫn tới hạn và tại sao nó có thể chạy model mạnh hơn so với agent chính.
- Triển khai vòng lặp hai agent theo kịch bản, trong đó agent chính phục vụ phản hồi và thời gian ngủ agent hợp nhất các khối giữa các lượt.

## Vấn đề

MemGPT (Bài 07) đã giải quyết luồng điều khiển bộ nhớ ảo. Ba vấn đề production xuất hiện:

1. **Độ trễ.** Mọi hoạt động bộ nhớ đều nằm trên đường dẫn quan trọng. Nếu agent phải cắt tỉa, tóm tắt hoặc đối chiếu trong khi người dùng chờ đợi, độ trễ đuôi sẽ tăng lên.
2. **Rot bộ nhớ.** Ghi tích lũy. Sự thật mâu thuẫn vẫn còn. Truy xuất chìm trong nội dung cũ.
3. **Cấu trúc loss.** Một kho lưu trữ phẳng không thể diễn tả "khối Con người luôn ở trong prompt; khối Persona luôn nằm trong prompt; việc hoán đổi khối nhiệm vụ trên mỗi session."

Letta (letta.com) là bản viết lại năm 2026. Các khối bộ nhớ làm cho cấu trúc rõ ràng; Điện toán thời gian ngủ di chuyển hợp nhất ra khỏi đường dẫn quan trọng.

## Khái niệm

### Ba tầng

| Bậc | Phạm vi | Nơi nó sống | Được viết bởi |
|------|-------|----------------|------------|
| Cốt lõi | Luôn hiển thị | Bên trong prompt chính | Agent gọi công cụ + viết lại thời gian ngủ |
| Recall | Lịch sử cuộc trò chuyện | Có thể truy xuất | Ghi nhật ký tự động |
| Lưu trữ | Sự thật tùy ý | Vector + KV + đồ thị | Agent gọi công cụ + nhập thời gian ngủ |

Lõi là lõi MemGPT. Recall là bộ đệm cuộc trò chuyện với cái đuôi bị đuổi ra ngoài. Lưu trữ là kho lưu trữ bên ngoài. Việc phân tách làm sạch tình trạng quá tải hai tầng của MemGPT.

### Khối bộ nhớ

Khối là một phần được nhập, liên tục, có thể chỉnh sửa của bậc cốt lõi. Bài báo gốc của MemGPT định nghĩa hai:

- **Khối con người** — sự thật về người dùng (tên, vai trò, sở thích, mục tiêu).
- **Khối tính cách** — khái niệm bản thân của agent (danh tính, giọng điệu, ràng buộc).

Letta khái quát hóa các khối tùy ý do người dùng xác định: một khối `Task` cho mục tiêu hiện tại, một khối `Project` cho các dữ kiện cơ sở mã, một khối `Safety` cho các ràng buộc cứng. Mỗi khối có một `id`, `label`, `value`, `limit` (giới hạn ký tự), `description` (để model biết khi nào cần chỉnh sửa).

Các khối có thể chỉnh sửa thông qua bề mặt công cụ:

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` - cô đọng một khối gần giới hạn của nó.

### Điện toán thời gian ngủ

Bổ sung Letta năm 2025: chạy agent thứ hai ở chế độ nền, ra khỏi đường dẫn quan trọng. Thời gian ngủ agents process bản ghi cuộc trò chuyện và ngữ cảnh cơ sở mã, ghi `learned_context` vào các khối được chia sẻ và hợp nhất hoặc vô hiệu hóa các bản ghi lưu trữ.

Các thuộc tính rơi ra:

- **Không mất phí độ trễ.** Phản hồi chính không đợi hoạt động bộ nhớ.
- **Cho phép model mạnh hơn.** agent thời gian ngủ có thể tốn kém hơn, model chậm hơn vì nó không bị hạn chế về độ trễ.
- **Cửa sổ hợp nhất tự nhiên.** Dedup, tóm tắt, vô hiệu hóa các sự kiện mâu thuẫn khi người dùng không chờ đợi.

Hình dạng phù hợp với cách con người làm việc: bạn thực hiện nhiệm vụ, bạn ngủ trên đó, trí nhớ dài hạn lắng đọng qua đêm.

### Letta V1 và lý luận gốc

Letta V1 (`letta_v1_agent`, 2026) không dùng `send_message`/nhịp tim và `Thought:` tokens nội tuyến để ủng hộ suy luận gốc. Phản hồi API (OpenAI) và Tin nhắn API với tư duy mở rộng (Anthropic) phát ra lý luận trên một kênh riêng biệt, được chuyển qua các lượt (được mã hóa giữa các nhà cung cấp trong production). Vòng điều khiển vẫn là ReAct. Tư tưởng trace là cấu trúc, không phải hình dạng prompt.

### Mô hình này sai ở đâu

- **Chặn cồng kềnh.** Infinite `block_append` nhanh chóng đạt đến giới hạn. Nối một trình tóm tắt khối trước khi ghi đẩy qua nắp.
- **Trôi dạt im lặng.** Thời gian ngủ agent viết lại một khối và agent chính không bao giờ nhận thấy. Các khối phiên bản và chênh lệch bề mặt trong trace.
- **Hợp nhất bị nhiễm độc.** Thời gian ngủ agent processes nội dung mà kẻ tấn công có thể truy cập vào lõi. Bài 27 cũng áp dụng cho bề mặt thời gian ngủ.

## Tự xây dựng

`code/main.py` thực hiện:

- `Block` — id, nhãn, giá trị, giới hạn, mô tả.
- `BlockStore` - Trình trợ giúp CRUD + `near_limit(label)`.
- Hai agents có kịch bản - `PrimaryAgent` phục vụ một lượt, `SleepTimeAgent` hợp nhất giữa các lượt.
- Một trace hiển thị một cuộc trò chuyện ba lượt với các lần viết khối, cộng với một thẻ thời gian ngủ tóm tắt một khối và vô hiệu hóa một sự thật cũ.

Chạy nó:

```
python3 code/main.py
```

Bản ghi cho thấy sự phân tách: lượt chính nhanh và tạo ra các bản ghi thô; Sleep Pass thu gọn và dọn dẹp.

## Ứng dụng

- **Letta** (letta.com) để triển khai tham khảo. Tự lưu trữ hoặc cloud được quản lý.
- **Claude Agent SDK skills** như kiến thức hình khối - skill là một khối hướng dẫn được đặt tên, có phiên bản, có thể truy xuất mà agent tải theo yêu cầu.
- **Bản dựng tùy chỉnh** dành cho các nhóm muốn kiểm soát phần phụ trợ lưu trữ. Sử dụng hợp đồng Letta API để bạn có thể di chuyển sau.

## Sản phẩm bàn giao

`outputs/skill-memory-blocks.md` tạo ra một hệ thống khối hình Letta với hooks thời gian ngủ cho bất kỳ runtime nào, bao gồm các quy tắc an toàn và hệ thống dây trích dẫn.

## Bài tập

1. Thêm công cụ `block_summarize` thay thế giá trị khối bằng bản tóm tắt do model tạo khi `near_limit` trả về true. Ngưỡng trigger nào giảm thiểu cả lệnh gọi tóm tắt và tràn khối?
2. Triển khai khử thời gian ngủ trên kho lưu trữ: hai bản ghi có văn bản có >90% token trùng lặp thu gọn thành một. Chỉ làm điều đó trong giấc ngủ, không bao giờ trên con đường quan trọng.
3. Khối phiên bản. Trên mỗi ghi, ghi lại giá trị cũ và một diff. Hiển thị `block_history(label)` để các toán tử có thể gỡ lỗi "tại sao agent quên X".
4. Đối xử với thời gian ngủ agents như những nhà văn không đáng tin cậy. Khi họ chạm vào khối Persona hoặc Safety, hãy yêu cầu xem xét agent thứ hai trước khi cam kết.
5. Chuyển ví dụ để sử dụng Letta API (`letta_v1_agent`). Những thay đổi nào trong khối schema và lý luận bản địa thay đổi hình dạng trace như thế nào?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Khối bộ nhớ | "Phần prompt có thể chỉnh sửa" | Phân đoạn bộ nhớ lõi được nhập, liên tục LLM chỉnh sửa |
| Khối người | "Bộ nhớ người dùng" | Sự thật về người dùng, được ghim trong lõi |
| Khối Persona | "Danh tính Agent" | Khái niệm bản thân, giọng điệu, ràng buộc, được ghim trong cốt lõi |
| Điện toán thời gian ngủ | "Công việc bộ nhớ không đồng bộ" | Thứ hai agent thực hiện hợp nhất ra khỏi con đường quan trọng |
| Cốt lõi / Recall / Lưu trữ | "Bậc" | Chia bộ nhớ ba lớp: luôn hiển thị / hội thoại / bên ngoài |
| Giới hạn khối | "Nắp" | Giới hạn ký tự trên mỗi khối; Tóm tắt lực lượng |
| Lý luận gốc | "Kênh tư duy" | Đầu ra suy luận cấp nhà cung cấp, không phải `Thought:` cấp prompt |
| Bối cảnh đã học | "Đầu ra ngủ" | Sự kiện thời gian ngủ agent ghi vào các khối được chia sẻ |

## Đọc thêm

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — mô hình khối
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — hợp nhất không đồng bộ
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) - viết lại lý luận gốc
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — nguồn gốc
