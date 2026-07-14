# Bàn giao và thói quen - Stateless Orchestration

> Swarm của OpenAI (tháng 10 năm 2024) đã chắt lọc nhiều agent orchestration thành hai primitives: **quy trình** (hướng dẫn + công cụ dưới dạng system prompt) và **handoffs** (một công cụ trả về một Agent khác). Không có máy trạng thái, không có DSL phân nhánh - các tuyến LLM bằng cách gọi công cụ chuyển giao bên phải. OpenAI Agents SDK (tháng 3 năm 2025) là người kế nhiệm production. Bản thân Swarm vẫn là tài liệu tham khảo khái niệm rõ ràng nhất - toàn bộ nguồn của nó nằm gọn trong vài trăm dòng. Mô hình này lan truyền vì bề mặt API gần "agent = prompt + công cụ; handoff = hàm trả về agent." Hạn chế: không trạng thái, vì vậy bộ nhớ là vấn đề của người gọi.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 04 (Primitive Model)
**Thời lượng:** ~60 phút

## Vấn đề

Mọi agent framework đa đều muốn bạn tìm hiểu DSL của nó: các nút và cạnh LangGraph, nhóm và nhiệm vụ của CrewAI, AutoGen GroupChat và người quản lý. DSL là những thứ trừu tượng thực sự, nhưng chúng làm cho mọi thứ cảm thấy nặng nề hơn mức cần thiết.

Swarm đẩy theo hướng ngược lại: sử dụng khả năng gọi công cụ mà model đã có. Chuyển giao trở thành lệnh gọi công cụ. Người điều phối là bất kỳ agent nào hiện đang tổ chức cuộc trò chuyện. Bộ máy nhà nước ngầm nằm trong hệ thống agents prompts.

## Khái niệm

### Hai primitives

**Routine.** Một system prompt xác định vai trò của agent và các công cụ có sẵn. Hãy nghĩ về nó giống như một tập hợp các hướng dẫn có phạm vi: "bạn là một agent phân loại; Nếu người dùng hỏi về việc hoàn tiền, hãy chuyển cho agent hoàn tiền."

**Handoff.** Một công cụ mà agent có thể gọi trả về một đối tượng Agent mới. Swarm runtime phát hiện giá trị trả về Agent và chuyển đổi agent đang hoạt động cho lượt tiếp theo.

Đó là toàn bộ sự trừu tượng.

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

system prompt của agent phân loại khiến nó chọn bàn giao phù hợp dựa trên tin nhắn của người dùng. Việc gọi công cụ của LLM thực hiện định tuyến.

### Tại sao nó lại lan truyền

- **API nhỏ.** Hai khái niệm để học.
- **Sử dụng những gì model đã làm.** Tool calling đã được cấp production trên các nhà cung cấp.
- **Không có gánh nặng máy trạng thái.** Bạn không mô tả biểu đồ; agents' prompts mô tả họ giao cho ai.

### Thương mại không quốc tịch

Swarm rõ ràng là không có trạng thái giữa các lần chạy. framework lưu giữ lịch sử tin nhắn trong quá trình chạy, nhưng nó không tồn tại bất cứ điều gì. Bộ nhớ, tính liên tục, các tác vụ chạy dài - tất cả đều là vấn đề của người gọi.

Vào production (OpenAI Agents SDK, tháng 3 năm 2025), đây là một trong những điều chính đã thay đổi: SDK bổ sung tính năng quản lý session, guardrails và theo dõi tích hợp trong khi vẫn giữ primitive bàn giao.

### Khi Swarm/handoffs phù hợp

- **Các mẫu phân loại.** agent tuyến đầu định tuyến người dùng đến chuyên gia.
- **Chuyển giao dựa trên Skill.** "Nếu nhiệm vụ cần mã, hãy gọi mã viên; Nếu nó cần nghiên cứu, hãy gọi cho nhà nghiên cứu."
- **Cuộc trò chuyện ngắn, có giới hạn.** Hỗ trợ khách hàng, Câu hỏi thường gặp, quy trình làm việc đơn giản.

### Khi Swarm gặp khó khăn

- **sessions dài với bộ nhớ được chia sẻ.** Handoffs đặt lại trạng thái hội thoại về lịch sử cộng prompt của agent mới. Không có trạng thái liên tục trên agents nếu không có bộ nhớ do người gọi quản lý.
- **Thực thi song song.** Handoff là một lần một - chuyển đổi agent đang hoạt động. Tính song song yêu cầu người gọi điều phối nhiều lần chạy Swarm.
- **Kiểm tra và phát lại.** Các lần chạy không trạng thái rất khó để phát lại chính xác; Lựa chọn bàn giao của LLM không phải là xác định.

### OpenAI Agents SDK (Tháng Ba 2025)

Người kế nhiệm production cho biết thêm:

- **Session trạng thái.** thread liên tục qua các lần chạy.
- **Guardrails.** Input/output xác thực hooks.
- **Truy kích.** Mọi lệnh gọi và chuyển giao công cụ đều được ghi lại.
- **Bộ lọc bàn giao.** Kiểm soát ngữ cảnh nào được chuyển khi chuyển giao.

Việc bàn giao primitive vẫn tồn tại; production công thái học được thêm vào xung quanh nó.

### Swarm so với GroupChat

Cả hai đều sử dụng định tuyến điều khiển LLM, nhưng chúng khác nhau về **ai chọn tiếp theo**:

- GroupChat: một bộ chọn (chức năng hoặc LLM) chọn người nói tiếp theo từ bên ngoài.
- Swarm: agent hiện tại chọn người kế nhiệm bằng cách gọi công cụ chuyển giao.

Swarm là "agent quyết định điều gì tiếp theo"; GroupChat là "người quản lý quyết định điều gì tiếp theo". Quyết định của Swarm nằm trong cuộc gọi công cụ của agent tích cực; Cuộc sống của GroupChat trong `GroupChatManager`.

## Tự xây dựng

`code/main.py` triển khai Swarm từ đầu: lớp dữ liệu Agent, cơ chế chuyển giao (công cụ trả về Agent) và vòng lặp chạy phát hiện agent chuyển mạch.

Bản demo: phân loại agent các lộ trình đến các chuyên gia hoàn tiền, bán hàng hoặc hỗ trợ. Mỗi chuyên gia có công cụ riêng. Vòng lặp chạy in mỗi lần chuyển giao.

Chạy:

```
python3 code/main.py
```

## Ứng dụng

`outputs/skill-handoff-designer.md` thiết kế cấu trúc liên kết chuyển giao cho một nhiệm vụ nhất định: agents nào tồn tại, chuyển giao nào họ có thể gọi, chuyển ngữ cảnh nào.

## Sản phẩm bàn giao

Danh sách kiểm tra:

- **Ghi nhật ký bàn giao.** Mỗi lần chuyển giao sẽ ghi một sự kiện trace với ảnh chụp nhanh ngữ cảnh từ agent, đến agent.
- **Quy tắc chuyển ngữ cảnh.** Quyết định những gì di chuyển khi chuyển giao: lịch sử đầy đủ (đắt tiền), N tin nhắn cuối cùng hoặc tóm tắt.
- **Guardrail khi bàn giao.** Việc chuyển giao cho một chuyên gia có quyền công cụ khác nhau phải được xác thực - nếu không, việc tiêm prompt có thể buộc phải chuyển giao không mong muốn.
- **Phát hiện vòng lặp.** Hai agents giao qua lại là một lỗi phổ biến; phát hiện bằng cách kiểm tra vòng Last K đơn giản.
- **Dự phòng agent.** Nếu mục tiêu chuyển giao không tồn tại, hãy quay trở lại mặc định an toàn.

## Bài tập

1. Chạy `code/main.py`, phân loại đến agent hoàn tiền. Xác nhận agent hoạt động của lượt thứ hai là hoàn tiền.
2. Thêm quy tắc phát hiện vòng lặp: nếu hai agents giống nhau đã chuyển giao 3 lần liên tiếp, hãy buộc thoát. Thiết kế dự phòng.
3. Đọc tài liệu OpenAI Agents SDK về bộ lọc bàn giao. Triển khai phiên bản "tóm tắt khi bàn giao": agent đi nén ngữ cảnh thành tóm tắt gạch đầu dòng trước khi agent đến tiếp quản.
4. So sánh chuyển giao Swarm với bộ chọn GroupChatManager. Mô hình nào làm cho việc tiêm prompt trở nên tồi tệ hơn và tại sao?
5. Đọc sách dạy nấu ăn Swarm (https://developers.openai.com/cookbook/examples/orchestrating_agents). Xác định một quyết định thiết kế rõ ràng Swarm đưa ra OpenAI Agents SDK thay đổi hoặc giữ lại.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Thói quen | "Người agent prompt" | System prompt + danh sách công cụ. Xác định vai trò và chuyển giao có sẵn. |
| Bàn giao | "Chuyển sang agent khác" | Một công cụ mà agent đang hoạt động có thể gọi trả về một Agent mới. runtime chuyển đổi agent hoạt động. |
| Không trạng thái | "Không có bộ nhớ giữa các lần chạy" | Swarm không tồn tại bất cứ điều gì; Trí nhớ là trách nhiệm của người gọi. |
| Hoạt động agent | "Ai đang nói bây giờ" | Người agent hiện đang tổ chức cuộc trò chuyện. Handoff thay đổi điều này. |
| Chuyển ngữ cảnh | "Điều gì di chuyển khi bàn giao" | Policy lịch sử mà agent sắp tới nhìn thấy: đầy đủ, N cuối cùng hoặc tóm tắt. |
| Vòng lặp bàn giao | "Agents bóng bàn" | Chế độ thất bại trong đó hai agents tiếp tục giao lại cho nhau. |
| OpenAI Agents SDK | "Production Swarm" | Người kế nhiệm tháng 3 năm 2025; thêm sessions, guardrails, theo dõi lên trên primitive bàn giao. |
| Bộ lọc Handoff | "Cổng khi chuyển tiếp" | SDK feature kiểm tra và sửa đổi ngữ cảnh tại ranh giới chuyển giao. |

## Đọc thêm

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — khớp nối tham chiếu
- [OpenAI Swarm repo](https://github.com/openai/swarm) — triển khai ban đầu, được giữ làm tài liệu tham khảo khái niệm
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — production người kế nhiệm với sessions và truy tìm
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — cách Code Claude subagents sử dụng mẫu giống như handoff thông qua `Task`
