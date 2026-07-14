# Agents nền chạy lâu: Thực thi bền bỉ

> Production agents đường chân trời dài không chạy trong `while True`. Mỗi cuộc gọi LLM trở thành một hoạt động với checkpoint, thử lại và phát lại. Tích hợp OpenAI Agents SDK của Temporal đã được GA vào tháng 3 năm 2026. Claude Code Routines (Anthropic) chạy các lệnh gọi Code Claude đã lên lịch mà không cần process cục bộ liên tục. Sessions tạm dừng đầu vào của con người, sống sót sau khi triển khai và tiếp tục từ checkpoint mới nhất do `thread_id` khóa. Đằng sau công thái học mới là một mô hình cũ - quy trình làm việc orchestration - với một đầu vào mới: LLM gọi là các hoạt động không xác định phải được phát lại một cách xác định khi phục hồi.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, minimal durable-execution state machine)
**Kiến thức tiên quyết:** Giai đoạn 15 · 10 (Chế độ cho phép), Giai đoạn 15 · 01 (Đường chân trời dài agents)
**Thời lượng:** ~60 phút

## Vấn đề

Hãy xem xét một agent chạy trong bốn giờ. Nó gọi ba công cụ, prompts người dùng hai lần và thực hiện bốn mươi cuộc gọi LLM. Nửa chừng, máy chủ mà nó đang chạy khởi động lại. Điều gì xảy ra?

- Trong một vòng lặp `while True` ngây thơ: mọi thứ đều bị mất. Quá trình chạy bắt đầu lại từ đầu. Ba lệnh gọi công cụ (với tác dụng phụ thực sự) thực thi lại. Người dùng được nhắc lại về những nội dung họ đã phê duyệt. Bốn mươi cuộc gọi LLM được lập hóa đơn lại.
- Với khả năng thực thi bền bỉ: quá trình chạy tiếp tục từ checkpoint gần đây nhất. Các hoạt động đã hoàn thành không được thực hiện lại; Kết quả của chúng được phát lại từ nhật ký bền. Người dùng không phê duyệt lại những thứ họ đã phê duyệt. Các cuộc gọi LLM đã được thực hiện sẽ không được lập hóa đơn lại.

Đây là mô hình tương tự mà các công cụ quy trình làm việc đã shipped trong một thập kỷ (Temporal, Cadence, Cherami của Uber). Điều mới là các cuộc gọi LLM giờ đây là một loại hoạt động - không xác định, tốn kém, có tác dụng phụ - và chúng hoàn toàn phù hợp với mô hình này.

Chủ đề của bài học: độ tin cậy của đường chân trời dài giảm (METR quan sát thấy "sự suy giảm trong 35 phút" - tỷ lệ thành công giảm gần bậc hai với đường chân trời). Khả năng thực thi bền bỉ cho phép chạy lâu hơn cấu hình độ tin cậy hỗ trợ, đây là một cách mới để hỏng hóc một cách an toàn nếu thiết kế đúng và không an toàn nếu thiết kế sai.

## Khái niệm

### Hoạt động, quy trình làm việc và phát lại

- **Quy trình làm việc**: mã orchestration xác định. Xác định trình tự các hoạt động, branches, chờ đợi. Phải xác định để có thể phát lại từ nhật ký sự kiện mà không có sự phân kỳ đáng ngạc nhiên.
- **Hoạt động**: một đơn vị công việc không xác định, có khả năng thất bại. LLM gọi, gọi công cụ, ghi tệp HTTP yêu cầu. Mỗi hoạt động được ghi lại với đầu vào và (sau khi hoàn thành) đầu ra của nó.
- **Nhật ký sự kiện**: kho sao lưu bền. Mọi hoạt động bắt đầu, hoàn thành, thất bại, thử lại và mọi quyết định quy trình làm việc đều được ghi lại.
- **Phát lại**: khi khôi phục, mã quy trình làm việc sẽ chạy lại từ đầu; Mọi hoạt động đã hoàn thành sẽ trả về kết quả đã ghi mà không cần thực thi lại. Chỉ những hoạt động chưa hoàn thành mới thực sự chạy.

Đây là hình dạng tương tự như React tái render dựa trên một DOM ảo, hoặc Git xây dựng lại một cây làm việc từ commits. Quyết định luận trong bộ điều phối là điều làm cho độ bền trở nên rẻ.

### Tại sao LLM cuộc gọi phù hợp với mẫu

LLM cuộc gọi là:
- Không xác định (temperature > 0; thậm chí temperature 0 trôi qua các phiên bản model).
- Đắt tiền (tiền và độ trễ).
- Có khả năng thất bại (rate limits, timeouts).
- Tác dụng phụ (nếu chúng gọi các công cụ).

Đây chính xác là hồ sơ hoạt động. Việc bao bọc mọi lệnh gọi LLM dưới dạng một hoạt động cho phép bạn thử lại với tính năng lùi theo cấp số nhân, điểm kiểm tra qua các lần khởi động lại và trace có thể phát lại để gỡ lỗi.

### Checkpoints được khóa bởi `thread_id`

LangGraph, Microsoft Agent Framework, Cloudflare Durable Objects và Claude Code Routines đều hội tụ trên cùng một hình dạng API: một `thread_id` (hoặc tương đương) xác định session; mỗi quá trình chuyển đổi trạng thái vẫn tồn tại sang một chương trình phụ trợ (PostgreSQL mặc định, SQLite cho dev, Redis cho bộ nhớ cache); sơ yếu lý lịch đọc checkpoint mới nhất.

Lựa chọn phụ trợ rất quan trọng:

- **PostgreSQL**: bền bỉ, có thể truy vấn, tồn tại sau khi triển khai. Mặc định cho LangGraph.
- **SQLite**: chỉ dành cho nhà phát triển cục bộ; mất dữ liệu trên các máy chủ.
- **Redis**: nhanh nhưng phù du trừ khi AOF/snapshot cấu hình.
- **Cloudflare Durable Objects**: được phân phối trong suốt; phạm vi bởi một khóa duy nhất; tồn tại trong nhiều giờ đến vài tuần.

### Đầu vào của con người dưới dạng trạng thái class đầu tiên

Đề xuất sau đó commit (Bài 15) đòi hỏi một trạng thái "chờ đợi con người" lâu dài. Quy trình làm việc tạm dừng, hàng đợi bên ngoài giữ yêu cầu đang chờ xử lý và phê duyệt tiếp tục từ chính thời điểm đó. Nếu không có độ bền, đây là nỗ lực tốt nhất; Với nó, một phê duyệt qua đêm sẽ đến và quy trình làm việc bắt đầu vào buổi sáng.

### Sự xuống cấp trong 35 phút

METR quan sát thấy rằng mỗi agent class đo cho thấy độ tin cậy giảm sau ~35 phút hoạt động liên tục. Tăng gấp đôi thời lượng nhiệm vụ tăng gần gấp bốn lần tỷ lệ thất bại. Thực thi bền bỉ không khắc phục được điều này; Nó cho phép bạn chạy lâu hơn cấu hình độ tin cậy hỗ trợ. Mô hình an toàn là kết hợp độ bền với checkpoints yêu cầu HITL mới khi nhập lại và với các công tắc ngắt kết nối ngân sách (Bài 13) giới hạn tổng số tính toán bất kể thời gian đồng hồ treo tường.

### Khi thực thi bền bỉ là câu trả lời sai

- Chạy ngắn hơn vài phút mà không có đầu vào của con người. Lợi ích > chi phí.
- Truy xuất thông tin chỉ đọc nghiêm ngặt.
- Các nhiệm vụ mà tính đúng đắn đòi hỏi từ đầu đến cuối trong một context window (một số nhiệm vụ lý luận; một số one-shot tạo).

```figure
memory-consolidation
```

## Ứng dụng

`code/main.py` triển khai một công cụ thực thi bền tối thiểu trong Python stdlib. Nó hỗ trợ:

- `@activity` trình trang trí ghi nhật ký đầu vào và đầu ra vào nhật ký sự kiện JSON.
- Một chức năng quy trình làm việc sắp xếp các hoạt động.
- Một hàm `run_or_replay(workflow, event_log)` phát lại các hoạt động đã hoàn thành mà không cần thực thi lại.

Trình điều khiển mô phỏng quy trình làm việc ba hoạt động, gặp sự cố giữa chừng và hiển thị (a) thử lại ngây thơ để thực hiện lại mọi thứ so với (b) phát lại chỉ chạy hoạt động bị thiếu.

## Sản phẩm bàn giao

`outputs/skill-durable-execution-review.md` xem xét một triển khai agent chạy dài được đề xuất để có hình dạng thực thi bền vững chính xác: hoạt động, tính xác định, checkpoint phụ trợ, trạng thái đầu vào của con người và policy HITL-on-resume.

## Bài tập

1. Chạy `code/main.py`. Quan sát sự khác biệt về số lần thực hiện hoạt động giữa thử lại và phát lại. Thay đổi điểm sự cố và hiển thị số lần phát lại thay đổi cho phù hợp.

2. Chuyển đổi công cụ đồ chơi để sử dụng `thread_id` một cách rõ ràng. Mô phỏng hai sessions đồng thời chia sẻ công cụ và xác nhận nhật ký sự kiện của chúng không va chạm.

3. Thực hiện một hoạt động trong công cụ đồ chơi. Giới thiệu tính không xác định (dấu thời gian đồng hồ treo tường bên trong quyết định quy trình làm việc). Thể hiện sự phân kỳ khi phát lại. Giải thích cách các công cụ thực xử lý điều này (đăng ký tác dụng phụ, `Workflow.now()` APIs).

4. Đọc bài đăng "Runtime đằng sau production agents sâu" của LangChain. Liệt kê mọi trạng thái mà runtime vẫn tồn tại và đặt tên cho mỗi chế độ lỗi.

5. Thiết kế checkpoint policy cho tác vụ mã hóa tự động kéo dài 6 giờ. Bạn checkpoint ở đâu? Tiếp tục khi gặp sự cố trông như thế nào? Điều gì yêu cầu HITL mới?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| Quy trình làm việc | "Agent script" | Mã orchestration xác định; Có thể phát lại từ nhật ký sự kiện |
| Hoạt động | "Một bước" | Đơn vị không xác định (cuộc gọi LLM, cuộc gọi công cụ); được ghi lại trước và sau |
| Nhật ký sự kiện | "Cửa hàng hỗ trợ" | Hồ sơ bền bỉ của mọi chuyển đổi trạng thái |
| Phát lại | "Sơ yếu lý lịch" | Chạy lại quy trình làm việc; Các hoạt động đã hoàn thành trả về kết quả đã ghi mà không cần thực thi lại |
| Checkpoint | "Điểm lưu" | Trạng thái bền bỉ được khóa bởi thread_id; Chiến thắng mới nhất trên sơ yếu lý lịch |
| thread_id | "Phím Session" | Mã định danh phạm vi trạng thái bền bỉ |
| Suy giảm trong 35 phút | "Phân rã độ tin cậy" | METR: tỷ lệ thành công giảm ~ bậc hai với đường chân trời |
| Thuyết không xác định | "Trôi dạt khi phát lại" | Đồng hồ treo tường, ngẫu nhiên, đầu ra LLM; phải được đăng ký là tác dụng phụ |

## Đọc thêm

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) - ngữ nghĩa ngân sách, lượt và sơ yếu lý lịch.
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — Hình dạng RequestInfoEvent.
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — yêu cầu cụ thể runtime.
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — hình dạng hoạt động cho LLM cuộc gọi.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) - tham chiếu xuống cấp 35 phút.
