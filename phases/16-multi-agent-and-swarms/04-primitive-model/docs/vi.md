# Đa Agent Primitive Model

> Mỗi agent framework shipping đa năng vào năm 2026 - AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Microsoft Agent Framework - là một điểm trong không gian thiết kế bốn chiều. Bốn primitives, không có gì hơn: agent, bàn giao, trạng thái chia sẻ, người điều phối. Bài học này xây dựng chúng từ con số không, chạy một hệ thống đồ chơi trên cả bốn, sau đó ánh xạ mọi framework chính vào cùng một trục để bạn có thể đọc bất kỳ bản phát hành mới nào trong một đoạn văn.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 (Kỹ thuật Agent), Giai đoạn 16 · 01 (Tại sao nên sử dụng Multi-Agent)
**Thời lượng:** ~60 phút

## Vấn đề

Cứ sáu tháng một lần lại có nhiều agent framework ships mới. AutoGen vào năm 2023. CrewAI vào năm 2024. LangGraph và OpenAI Swarm vào năm 2024. Google ADK vào tháng 4 năm 2025. Microsoft Agent Framework RC vào tháng 2 năm 2026. Mỗi thông cáo báo chí đều tuyên bố là "trừu tượng phù hợp".

Nếu bạn cố gắng học từng cái một, bạn sẽ kiệt sức. Các APIs trông khác. Các tài liệu không đồng ý về "agent" là gì. Một framework gọi bộ nhớ được chia sẻ của nó là "bảng đen", một người khác gọi nó là "nhóm tin nhắn", người thứ ba gọi nó là "StateGraph". Bạn bắt đầu nghi ngờ lĩnh vực này chỉ đang xáo trộn.

Không phải vậy. Bên dưới tiếp thị, bốn primitives ổn định. Học chúng một lần, đọc mọi framework mới trong một đoạn văn.

## Khái niệm

### Bốn primitives

1. **Agent** - một system prompt cộng với một danh sách công cụ. Không quốc tịch; Mỗi lần chạy đều bắt đầu từ system prompt và lịch sử tin nhắn hiện tại.
2. **Handoff** — chuyển giao quyền kiểm soát có cấu trúc từ agent này sang  khác. Về mặt cơ học, một lệnh gọi công cụ trả về một agent mới hoặc một cạnh đồ thị tuân theo một điều kiện.
3. **Trạng thái chia sẻ** — bất kỳ cấu trúc dữ liệu nào mà nhiều agent có thể đọc (đôi khi ghi). Nhóm tin nhắn, bảng đen, kho lưu trữ khóa-giá trị vector bộ nhớ.
4. **Orchestrator **- ai quyết định ai sẽ nói tiếp theo. Tùy chọn: biểu đồ rõ ràng (xác định), bộ chọn loa LLM (mềm), cuộc gọi chuyển giao của người nói cuối cùng (OpenAI Swarm) hoặc bộ lập lịch qua hàng đợi (kiến trúc swarm).

Đó là toàn bộ không gian thiết kế. Mỗi framework chọn mặc định cho mỗi trục; Các rest là cú pháp bề mặt.

### Cách mọi framework 2026 ánh xạ đến nó

| Framework | Agent | Bàn giao | Trạng thái được chia sẻ | Điều phối viên |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | công cụ trả về Agent | Vấn đề của người gọi | Cuộc gọi bàn giao tiếp theo của LLM |
| AutoGen v0.4 / AG2 | `ConversableAgent` | bộ chọn loa trên GroupChat | Nhóm tin nhắn | Chức năng chọn (LLM hoặc vòng tròn) |
| Phi hành đoànAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Đầu ra nhiệm vụ được xâu chuỗi | người quản lý LLM hoặc lệnh tĩnh |
| Đồ thị LangGraph | Chức năng nút | Graph Edge + Điều kiện | Bộ giảm tốc `StateGraph` | đồ thị, xác định |
| Microsoft Agent Framework | agent + orchestration mẫu | Mẫu cụ thể | thread / ngữ cảnh | Mẫu cụ thể |
| Quảng cáo Google | Thẻ agent + A2A | A2A nhiệm vụ | A2A artifacts | Người dẫn chương trình quyết định |

Sự khác biệt bề mặt trông rất lớn. Bên dưới: bốn núm giống nhau.

### Tại sao điều này lại quan trọng

Khi bạn nhìn thấy primitives, framework so sánh sẽ trở thành một danh sách kiểm tra ngắn:

- Trình điều phối có tin tưởng LLM định tuyến (Swarm) hay nó ghim định tuyến trong mã (LangGraph)?
- Trạng thái được chia sẻ là lịch sử đầy đủ (GroupChat) hay dự kiến (StateGraph reducer)?
- agents có thể sửa đổi prompts của nhau (người quản lý CrewAI) hay chỉ chuyển giao (Swarm)?

Ba câu hỏi đó trả lời 80% trong số đó framework phù hợp với một vấn đề nhất định. Bạn ngừng mua sắm "đa agent framework tốt nhất" và bắt đầu thiết kế cho trục mà bạn thực sự quan tâm.

### Cái nhìn sâu sắc về người không có quốc tịch

Mọi primitive ngoại trừ trạng thái chia sẻ đều không có trạng thái. Agent là một hàm của (prompt, công cụ). Handoff là một lệnh gọi hàm. Orchestrator là một người lập lịch. **Thứ có trạng thái duy nhất trong hệ thống là trạng thái chia sẻ.** Đó là nơi tất cả các lỗi thú vị tồn tại: nhiễm độc bộ nhớ (Bài 15), sắp xếp thứ tự tin nhắn, tạo phiên bản, tranh chấp ghi.

Frameworks ẩn trạng thái chia sẻ (Swarm) đẩy vấn đề đến người gọi. Frameworks tập trung nó (LangGraph checkpoint, AutoGen pool) làm cho nó có thể kiểm tra được nhưng chuyển chi phí điều phối sang triển khai trạng thái chia sẻ.

### Giải phẫu của một primitive

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

Không có bộ nhớ. Không có tiểu bang. Hai agents có cùng system prompt và công cụ có thể hoán đổi cho nhau. Mọi thứ trông giống như trạng thái mỗi agent thực sự ở trạng thái chia sẻ hoặc giao thức bàn giao.

#### Bàn giao

```
Handoff = (from_agent, to_agent, reason, payload)
```

Ba triển khai chiếm ưu thế:

- **Trả về hàm** — công cụ trả về agent tiếp theo. Đây là mô hình OpenAI Swarm. Agents thực hiện định tuyến trong schemas công cụ của họ.
- **Cạnh đồ thị** — LangGraph. Các cạnh là khai báo. Sự LLM tạo ra một giá trị; Một điều kiện chọn nút tiếp theo.
- **Lựa chọn loa** — AutoGen GroupChat. Một hàm selector (đôi khi bản thân nó là một cuộc gọi LLM) đọc nhóm và chọn ai nói tiếp theo.

#### Trạng thái được chia sẻ

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

Tối thiểu là một danh sách các tin nhắn. Thường nhiều hơn: artifacts có cấu trúc (đầu ra tác vụ CrewAI), ngữ cảnh được nhập (bộ giảm tốc LangGraph), bộ nhớ ngoài (MCP, vector DB).

Hai cấu trúc liên kết: **full pool** (mọi agent nhìn thấy mọi tin nhắn) và **projected** (agents xem chế độ xem phạm vi vai trò). Hồ bơi đầy đủ rất đơn giản và quy mô xấu. Các hồ bơi dự kiến mở rộng quy mô nhưng yêu cầu thiết kế schema trước.

#### Điều phối viên

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

Bốn hương vị:

- **Tĩnh** — biểu đồ được cố định tại thời điểm xây dựng (LangGraph deterministic, CrewAI Sequential).
- **LLM-chọn** — một LLM đọc nhóm và chọn loa tiếp theo (AutoGen, CrewAI Hierarchical).
- **Handoff-driven** — agent hiện tại quyết định bằng cách gọi công cụ handoff (Swarm).
- **Queue-driven** — workers kéo từ hàng đợi được chia sẻ; không có loa tiếp theo rõ ràng (kiến trúc swarm, Ma trận).

### Những thay đổi giữa frameworks

Sau khi các primitives được cố định, các quyết định thiết kế còn lại là:

- **Chiến lược bộ nhớ** — điểm kiểm tra tạm thời và bền bỉ (con trỏ kiểm tra LangGraph).
- **Ranh giới an toàn** — ai có thể phê duyệt bàn giao (human-in-the-loop).
- **Kế toán chi phí** — ngân sách trên mỗi agent token.
- **Observability** - theo dõi các bàn giao, trạng thái dai dẳng để phát lại.

Tất cả đều có thể thực hiện được trên primitives. Không ai trong số họ là primitives mới.

## Tự xây dựng

`code/main.py` thực hiện bốn primitives trong ~150 dòng Python stdlib. Không có LLM thực sự - mỗi agent là một policy có kịch bản nên trọng tâm vẫn là cấu trúc phối hợp.

Tệp xuất:

- `Agent` — một lớp dữ liệu gồm tên, system prompt, công cụ policy hàm.
- `Handoff` — một hàm trả về một agent mới.
- `SharedState` — một nhóm tin nhắn an toàn thread.
- `Orchestrator` — ba biến thể: `StaticOrchestrator`, `HandoffOrchestrator`, `LLMSelectorOrchestrator` (mô phỏng).

Bản demo chạy cùng một ba agent pipeline (nghiên cứu → viết → đánh giá) thông qua cả ba loại trình điều phối và in nhóm thông báo ở cuối. Bạn có thể thấy rằng kết quả đầu ra chỉ khác nhau ở *ai chọn tiếp theo*; trạng thái agents và chia sẻ giống hệt nhau giữa các lần chạy.

Chạy nó:

```
python3 code/main.py
```

Đầu ra dự kiến: ba lần chạy trình điều phối, một lần chạy cho mỗi mẫu. Mỗi bản in nhóm tin nhắn cuối cùng. Quá trình chạy theo hướng bàn giao đạt ít agents hơn nếu nhà nghiên cứu quyết định nó được thực hiện sớm - đó là sự đánh đổi định tuyến LLM trong thu nhỏ.

## Ứng dụng

`outputs/skill-primitive-mapper.md` là một skill đọc bất kỳ cơ sở mã đa agent hoặc tài liệu framework nào và trả về ánh xạ bốn primitive. Chạy nó trên một bản phát hành framework mới để hiểu một đoạn trước khi đọc tài liệu chuyên sâu.

## Sản phẩm bàn giao

Trước khi áp dụng một framework mới, hãy viết ánh xạ primitive cho nó. Nếu bạn không thể, các tài liệu không đầy đủ hoặc framework đang phát minh ra primitive thứ năm (hiếm - kiểm tra hương vị trạng thái chia sẻ mà bạn chưa thấy).

Ghim ánh xạ trong tài liệu kiến trúc. Khi một thành viên mới trong nhóm tham gia, hãy gửi cho họ ánh xạ trước tài liệu API. Khi framework phiên bản thay đổi, hãy phân biệt ánh xạ chứ không phải nhật ký thay đổi.

## Bài tập

1. Chạy `code/main.py` ba lần với các agent policies khác nhau. Quan sát cách lựa chọn của trình điều phối thay đổi agents chạy.
2. Triển khai loại trình điều phối thứ tư: kiểu điều khiển theo hàng đợi trong đó agents thăm dò ý kiến trạng thái được chia sẻ cho công việc. Bế tắc nào có thể xảy ra và làm thế nào để bạn phát hiện ra nó?
3. Lấy khởi động nhanh LangGraph (https://docs.langchain.com/oss/python/langgraph/workflows-agents) và viết lại nó thành bốn primitives. Cái nào trong số các trừu tượng của LangGraph lập bản đồ 1: 1 và đâu là trình bao bọc tiện lợi?
4. Đọc sách dạy nấu ăn OpenAI Swarm (https://developers.openai.com/cookbook/examples/orchestrating_agents). Xác định cái nào trong số bốn primitives Swarm làm cho công thái học nhất và cái nào nó đẩy đến người gọi.
5. Tìm một framework trong bảng này ẩn hoàn toàn trạng thái được chia sẻ. Giải thích những gì bị ngắt khi agents cần phối hợp giữa các lần chuyển giao mà không cần đọc lại lịch sử.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Agent | "Một LLM với các công cụ" | Một `(system_prompt, tools, model)` ba lần. Không quốc tịch. |
| Bàn giao | "Chuyển giao quyền kiểm soát" | Một cuộc gọi có cấu trúc đặt tên cho agent tiếp theo và payload tùy chọn. Ba cách triển khai: trả về hàm, cạnh đồ thị, lựa chọn loa. |
| Trạng thái được chia sẻ | "Ký ức" / "ngữ cảnh" | Phần trạng thái duy nhất của hệ thống đa agent. Nhóm tin nhắn hoặc bảng đen. |
| Điều phối viên | "Điều phối viên" | Ai quyết định ai sẽ tranh cử tiếp theo. Biểu đồ tĩnh, bộ chọn LLM, điều khiển bằng bàn giao hoặc điều khiển hàng đợi. |
| Primitive | "Trừu tượng" | Một trong bốn trục mà mỗi framework tham số hóa. Không phải là một framework feature. |
| Nhóm tin nhắn | "Lịch sử trò chuyện được chia sẻ" | Trạng thái chia sẻ lịch sử đầy đủ. Dễ lý luận, quy mô xấu. |
| Trạng thái dự kiến | "Chế độ xem có phạm vi" | Chế độ xem theo vai trò cụ thể vào trạng thái chia sẻ. Quy mô, yêu cầu thiết kế schema. |
| Lựa chọn loa | "Ai nói tiếp theo" | Orchestrator trong đó một hàm (thường là LLM) chọn agent tiếp theo từ một nhóm. |

## Đọc thêm

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) - cách diễn đạt rõ ràng nhất của orchestration điều khiển bằng bàn giao
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — Lựa chọn GroupChat + loa là tài liệu tham khảo cho orchestration được chọn LLM
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — orchestration cạnh đồ thị và trạng thái chia sẻ dựa trên bộ giảm tốc
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — agents vai trò-mục tiêu-cốt truyện, processes tuần tự / phân cấp
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — dòng AutoGen v0.2 trực tiếp sau khi Microsoft chuyển v0.4 sang bảo trì
