# Kiến trúc song song / Swarm / nối mạng

> Trái ngược với người giám sát: không có người quyết định trung tâm. Agents đọc bus sự kiện được chia sẻ, chọn công việc không đồng bộ, ghi lại kết quả. LangGraph hỗ trợ rõ ràng "Kiến trúc Swarm" cho các môi trường năng động, phi tập trung. Ma trận (arXiv: 2511.21686) đại diện cho cả luồng điều khiển và dữ liệu dưới dạng các thông báo tuần tự được chuyển qua hàng đợi phân tán để loại bỏ tắc nghẽn của trình điều phối. Sự đánh đổi rất rõ ràng: quyết định luận và khả năng truy xuất nguồn gốc cho khả năng mở rộng. Swarm phù hợp với các nhiệm vụ với nhiều bài toán phụ độc lập; Nó không phù hợp với các nhiệm vụ cần một kế hoạch mạch lạc duy nhất.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib, `threading`, `queue`)
**Kiến thức tiên quyết:** Giai đoạn 16 · 05 (Mẫu giám sát), Giai đoạn 16 · 04 (Primitive Model)
**Thời lượng:** ~75 phút

## Vấn đề

Người giám sát mở rộng quy mô đến một vài workers. Còn hàng trăm thì sao? Bản thân người giám sát trở thành nút thắt cổ chai: mọi quyết định về việc ai làm gì đều đi qua một agent. Một bước kế hoạch chậm sẽ khiến toàn bộ hệ thống bị đình trệ.

Swarm kiến trúc lật ngược thiết kế. Thay vì một nhà lập kế hoạch trung tâm điều động công việc, workers chọn công việc từ hàng đợi chung. "Sự phối hợp" được đưa vào ngữ nghĩa xe buýt sự kiện. Không có người điều phối; hệ thống mở rộng quy mô cho đến khi hàng đợi thực hiện.

## Khái niệm

### Hình dạng

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

Không có người điều phối. Mỗi worker lặp lại: kéo một tác vụ, process, viết kết quả (và tùy chọn xếp hàng theo dõi).

### Khi swarm phù hợp

- **Nhiều nhiệm vụ độc lập.** Cạo, biến đổi, phân loại. Nhiệm vụ không phụ thuộc vào nhau.
- **Thời lượng làm việc thay đổi.** Nếu một số tác vụ mất 100 mili giây và những tác vụ khác mất 10 giây, swarm sẽ tự động cân bằng tải — nhanh workers kéo các tác vụ tiếp theo. Một người giám sát phải dự đoán thời gian.
- **Thông lượng hơn quyết định.** Bạn quan tâm đến tổng thời gian hoàn thành, không phải đặt hàng nghiêm ngặt.

### Khi swarm không thành công

- **Quy trình làm việc theo thứ tự.** Nếu bước 3 cần đầu ra của bước 2, swarm có nguy cơ bị kích hoạt bước 3 trước khi bước 2 được thực hiện.
- **Nhiệm vụ lập kế hoạch toàn cầu.** Các câu hỏi nghiên cứu phức tạp được hưởng lợi từ một công cụ lập kế hoạch. Một swarm các nhà nghiên cứu tạo ra các sự kiện độc lập, không phải một báo cáo mạch lạc.
- **Gỡ lỗi.** Không có nhật ký trung tâm và công việc không đồng bộ, việc tái tạo lỗi rất tốn kém.

### Ma trận (arXiv: 2511.21686)

Ma trận là bài báo năm 2025 đưa swarm đến kết luận tự nhiên của nó: cả luồng điều khiển và luồng dữ liệu đều là các thông điệp tuần tự trên hàng đợi phân tán. Không có điều phối viên trung tâm. Khả năng chịu lỗi đến từ độ bền của tin nhắn. Khả năng mở rộng là vấn đề của nhà môi giới tin nhắn, không phải của hệ thống.

Đóng góp: một model lập trình trong đó sự phối hợp đa agent là "agent đăng ký chủ đề thông điệp nào?" thay vì "người giám sát chọn agent nào tiếp theo?" Điều này làm cho hệ thống trông giống như một lưới sự kiện pub/sub.

### Kiến trúc Swarm của LangGraph

Tài liệu LangGraph 2025 mô tả rõ ràng "Kiến trúc Swarm" là một trong những mẫu đa agent: agents là các nút, nhưng các cạnh tạo thành một biểu đồ có hướng với các chu kỳ và bất kỳ nút nào cũng có thể được kích hoạt từ nhóm. Một worker chọn từ công việc có sẵn theo điều kiện, không phải theo sự phân công của người giám sát.

### Chế độ thất bại: đói và đốm nóng

Nếu tất cả workers thực hiện nhiệm vụ nhanh nhất có sẵn, các nhiệm vụ chạy lâu dài sẽ không bao giờ được chọn cho đến khi chúng là những nhiệm vụ duy nhất còn lại. Chết đói hàng đợi cổ điển.

Giảm thiểu:
- Hàng đợi ưu tiên với lão hóa rõ ràng (tăng mức độ ưu tiên với thời gian chờ).
- Worker chuyên môn: một số workers chỉ nhận nhiệm vụ "dài".
- Áp suất ngược: giới hạn số lượng tác vụ nhanh vào hàng đợi.

### Liên kết định tuyến dựa trên nội dung

Swarm kết hợp tự nhiên với định tuyến dựa trên nội dung (Bài 22). Thay vì hàng đợi chung, hãy có một hàng đợi cho mỗi loại tin nhắn. Chuyên gia workers chỉ đăng ký loại của họ. Đây là cơ sở cho các kiến trúc message-bus có quy mô lên hàng nghìn agents.

## Tự xây dựng

`code/main.py` thực hiện swarm 4 worker threads kéo từ một `queue.Queue` dùng chung. Nhiệm vụ có thời lượng thay đổi (một số nhanh, một số chậm). Bản demo tương phản:

- **Đường cơ sở tuần tự: **một worker processes nối tiếp tất cả các nhiệm vụ.
- **Nhiệm vụ cố định: **mỗi nhiệm vụ được giao trước cho một worker cụ thể (kiểu giám sát).
- **Swarm:** workers kéo từ hàng đợi dùng chung.

Swarm cân bằng tự động tải; Bài tập cố định rời đi nhanh workers nhàn rỗi khi nhiệm vụ được giao của họ chậm.

Chạy:

```
python3 code/main.py
```

Đầu ra hiển thị số lượng tác vụ trên mỗi worker (swarm phân phối không đồng đều nhưng tối ưu) và thời gian đồng hồ treo tường.

## Ứng dụng

`outputs/skill-swarm-fit.md` đánh giá xem một nhiệm vụ có nên sử dụng swarm so với người giám sát hay không. Đầu vào: tính độc lập của tác vụ, thời lượng variance, yêu cầu đặt hàng, nhu cầu gỡ lỗi.

## Sản phẩm bàn giao

Danh sách kiểm tra:

- **Hàng đợi ưu tiên với lão hóa.** Ngăn ngừa nạn đói trong thời gian dài.
- **Worker idempotency.** Một nhiệm vụ có thể được kéo nhiều hơn một lần nếu một worker gặp sự cố giữa chừng. Workers phải là idempotent.
- **Hàng đợi bền bỉ.** Sử dụng Kafka, Redis Streams hoặc hàng đợi được hỗ trợ bởi cơ sở dữ liệu để production. `queue.Queue` chỉ có trong bộ nhớ.
- **Observability cho mỗi nhiệm vụ.** Mỗi nhiệm vụ đều có ID trace; Mỗi worker ghi nhật ký start/end với nó.
- **Áp suất ngược.** Nếu hàng đợi phát triển nhanh hơn workers xả nước, hãy làm chậm nhà sản xuất.

## Bài tập

1. Chạy `code/main.py`. swarm nhanh hơn nhiều so với tuần tự trên khối lượng công việc có thời lượng thay đổi? Nhanh hơn bao nhiêu so với nhiệm vụ cố định?
2. Thêm biến thể hàng đợi ưu tiên (sử dụng `queue.PriorityQueue`). Chỉ định mức độ ưu tiên theo trường "tầm quan trọng" của nhiệm vụ. Quan sát xem các nhiệm vụ có mức độ ưu tiên thấp có bao giờ bị bỏ đói khi tải liên tục hay không.
3. Triển khai máy dò điểm nóng: ghi nhật ký khi có bất kỳ tác vụ nào worker processes nhiều hơn 3× so với worker chậm nhất. Điều đó chỉ ra điều gì về phân phối thời lượng nhiệm vụ?
4. Đọc tóm tắt bài báo Ma trận (arXiv: 2511.21686) và Phần 3. Xác định một sự đánh đổi cụ thể mà Ma trận chấp nhận (tăng khả năng mở rộng) và một sự đánh đổi mà nó từ bỏ (truy xuất nguồn gốc, tính xác định).
5. Chuyển đổi bản demo swarm để sử dụng `queue.Queue` bộ (task_type, payload), workers chỉ đăng ký các loại cụ thể. Quy tắc định tuyến nào có ý nghĩa khi các tác vụ không đồng nhất?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Kiến trúc Swarm | "agents phi tập trung" | Workers kéo từ hàng đợi dùng chung; không có người điều phối trung tâm. |
| Xe buýt sự kiện | "Agents đăng ký chủ đề" | Trình trung chuyển tin nhắn định tuyến các tác vụ đến workers theo loại hoặc nội dung. |
| Đói | "Nhiệm vụ không bao giờ chạy" | Nhiệm vụ có mức độ ưu tiên thấp không bao giờ được chọn vì công việc có mức độ ưu tiên cao hơn đến liên tục. |
| Điểm nóng | "Một worker chết đuối" | Mất cân bằng tải khi một worker nhận được hầu hết các nhiệm vụ. |
| Áp suất ngược | "Làm chậm nhà sản xuất" | Cơ chế báo hiệu ngược dòng ngừng sản xuất khi hàng đợi đầy. |
| Idempotent worker | "An toàn để chạy lại" | Một tác vụ được xử lý hai lần sẽ tạo ra cùng một kết quả. Bắt buộc vì workers có thể gặp sự cố giữa chừng. |
| Hàng đợi bền bỉ | "Sống sót sau sự cố" | Hàng đợi được hỗ trợ bởi đĩa hoặc bộ nhớ sao chép; Nhiệm vụ không bị mất khi một worker gặp sự cố. |
| Ma trận framework | "swarm truyền tin nhắn đầy đủ" | Cả dữ liệu và luồng điều khiển đều là thông báo tuần tự trên hàng đợi phân tán. |

## Đọc thêm

- [LangGraph workflows and agents — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — hỗ trợ swarm rõ ràng
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686) — swarm truyền tin nhắn đầy đủ
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system) - tại sao một hệ thống production cụ thể rõ ràng chọn người giám sát thay vì swarm
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/) — diễn viên theo hướng sự kiện viết lại, gần swarm hơn so với GroupChat của v0.2
