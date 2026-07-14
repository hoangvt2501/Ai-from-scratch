# Đánh đổi Agent Framework - LangGraph vs CrewAI vs AutoGen vs Agno

> Mỗi framework đều bán cùng một bản demo (nghiên cứu agent xây dựng một báo cáo) và ẩn cùng một lỗi (trạng thái schema chiến đấu với lớp orchestration). Chọn framework có trừu tượng phù hợp với hình dạng vấn đề của bạn; mọi thứ khác đều là keo bạn viết hai lần.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 11 · 09 (Function Calling), Giai đoạn 11 · 16 (Đồ thị)
**Thời lượng:** ~45 phút

## Vấn đề

Bạn có một nhiệm vụ cần nhiều hơn một cuộc gọi LLM. Có thể đó là một quy trình nghiên cứu (lập kế hoạch, tìm kiếm, tóm tắt, trích dẫn). Có thể đó là một pipeline đánh giá mã (phân tích cú pháp khác biệt, phê bình, vá lỗi, xác thực). Có thể đó là một trợ lý nhiều lượt đặt vé máy bay, viết email và gửi báo cáo chi phí. Bạn chọn một framework.

Ba ngày sau, bạn phát hiện ra các trừu tượng của framework bị rò rỉ. CrewAI cung cấp cho bạn vai trò nhưng chiến đấu với bạn khi "nhà nghiên cứu" cần giao một kế hoạch có cấu trúc cho "nhà văn". AutoGen cung cấp cho bạn cuộc trò chuyện giữa agents nhưng không có trạng thái class đầu tiên, vì vậy checkpoint của bạn chỉ là một dưa chuột của nhật ký cuộc trò chuyện. LangGraph cung cấp cho bạn một biểu đồ trạng thái nhưng buộc bạn phải đặt tên cho mọi chuyển đổi trước khi bạn biết agent sẽ làm gì. Agno cung cấp cho bạn một trừu tượng một agent hét lên khi bạn cố gắng mở rộng thành ba workers đồng thời.

Cách khắc phục không phải là "chọn framework tốt nhất". Đó là để kết hợp sự trừu tượng cốt lõi của framework với hình dạng của vấn đề của bạn. Bài học này vẽ bản đồ đó.

## Khái niệm

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

Bốn frameworks thống trị bối cảnh năm 2026. Trừu tượng cốt lõi của họ không giống nhau.

| Framework | Trừu tượng cốt lõi | Phù hợp nhất | Phù hợp tồi tệ nhất |
|-----------|------------------|----------|-----------|
| **Đồ thị Lang** | `StateGraph` - trạng thái được nhập, nút, cạnh có điều kiện, con trỏ kiểm tra. | Quy trình làm việc với trạng thái rõ ràng và ngắt con người trong vòng lặp; production agents cần gỡ lỗi du hành thời gian. | Động não lỏng lẻo, theo vai trò mà cấu trúc liên kết không được biết. |
| **Phi hành đoànAI **| `Crew` — vai trò (mục tiêu, cốt truyện), nhiệm vụ process (tuần tự hoặc phân cấp). | Quy trình làm việc nhập vai hoặc dựa trên cá tính với kế hoạch linear/hierarchical ngắn. | Bất cứ điều gì có trạng thái ngoài lịch sử lượt của phi hành đoàn; phân nhánh phức tạp. |
| **Tự động** | `ConversableAgent` cặp — hai hoặc nhiều agents lần lượt nói cho đến khi có điều kiện thoát. | Nhiều agent *đối thoại* (giáo viên-học sinh, người đề xuất-nhà phê bình, diễn viên-nhà phê bình) nơi suy nghĩ xuất hiện từ cuộc trò chuyện. | Quy trình làm việc xác định với DAG đã biết; bất kỳ thứ gì cần trạng thái bền bỉ khi khởi động lại. |
| **Bất khả tri** | `Agent` — một LLM + công cụ + bộ nhớ, có thể kết hợp thành Teams. | Xây dựng nhanh các nhóm agents đơn và nhẹ; trình điều khiển lưu trữ tích hợp và đa phương thức mạnh mẽ. | Biểu đồ sâu, phân nhánh rõ ràng với các bộ giảm tùy chỉnh. |

### "Trừu tượng" thực sự có nghĩa là gì

Trừu tượng cốt lõi của framework là thứ bạn vẽ trên bảng trắng khi bạn giới thiệu kiến trúc.

- **LangGraph** → bạn vẽ một biểu đồ. Các nút là các bước, các cạnh là chuyển tiếp và đối tượng trạng thái tại mọi điểm được nhập. model tinh thần là một cỗ máy trạng thái.
- **Phi hành đoànAI **→ bạn vẽ sơ đồ tổ chức. Mỗi vai trò có một mô tả công việc và một người quản lý định tuyến nhiệm vụ. Người model tinh thần là một nhóm nhỏ các chuyên gia.
- **AutoGen** → bạn vẽ DM Slack. Hai agents nhắn tin cho nhau; người thứ ba tham gia nếu bạn cần người kiểm duyệt. model tinh thần là trò chuyện.
- **Agno** → bạn vẽ một chiếc hộp duy nhất với các công cụ treo trên đó. Đặt các hộp cạnh nhau cho một đội. model tinh thần là "agent có pin đi kèm".

### Câu hỏi về nhà nước

Tiểu bang là nơi hầu hết các lựa chọn framework gặp lỗi ở production.

- **LangGraph.** Trạng thái được gõ (`TypedDict` hoặc Pydantic model), bộ giảm tốc trên mỗi trường, con trỏ kiểm tra class đầu tiên (SQLite/Postgres/Redis). Tiếp tục, ngắt và du hành thời gian là miễn phí. *(Xem Giai đoạn 11 · 16.)*
- **CrewAI.** Trạng thái chảy dưới dạng chuỗi giữa các nhiệm vụ thông qua trường `context` hoặc được cấu trúc thông qua `output_pydantic`. Không có kho lưu trữ bền bỉ cho mỗi phi hành đoàn ra khỏi hộp; bạn tự bắt vít nếu phi hành đoàn phải sống sót sau khi khởi động lại.
- **AutoGen.** Trạng thái là lịch sử trò chuyện và bất kỳ `context` nào do người dùng xác định. Bản ghi cuộc trò chuyện vẫn tồn tại; trạng thái quy trình làm việc tùy ý không tồn tại trừ khi bạn viết bộ điều hợp.
- **Agno.** Trình điều khiển lưu trữ tích hợp (SQLite, Postgres, Mongo, Redis, DynamoDB) được gắn vào `Agent` qua `storage=` — sessions hội thoại và bộ nhớ người dùng sẽ tự động tồn tại. Không phải là một con trỏ kiểm tra biểu đồ đầy đủ; một cửa hàng session.

### Câu hỏi phân nhánh

Mọi agent branches không tầm thường. Ai quyết định branch quan trọng.

- **LangGraph** — bạn quyết định, thông qua các cạnh có điều kiện. Định tuyến là một hàm Python có tên branches. Branches là class đầu tiên trong biểu đồ đã biên dịch; con trỏ kiểm tra ghi lại branch nào đã được thực hiện.
- **CrewAI** — người quản lý quyết định ở chế độ phân cấp; ở chế độ tuần tự, bạn quyết định tại thời điểm xây dựng. Định tuyến được ngầm ẩn trong danh sách nhiệm vụ; không có "nếu" class đầu tiên bên ngoài prompt của người quản lý.
- **AutoGen** — agents quyết định thông qua trò chuyện. Phân nhánh xuất hiện từ người nói tiếp theo. `GroupChatManager` chọn người nói tiếp theo; bạn có thể viết tay một `speaker_selection_method` nhưng mặc định là điều khiển LLM.
- **Agno** — agent quyết định sử dụng công cụ nào sẽ gọi tiếp theo. Các nhóm có chế độ coordinator/router/collaborator; phân nhánh ngoài đó là trách nhiệm của nhà phát triển.

### Câu hỏi observability

- **LangGraph** — OpenTelemetry qua LangSmith hoặc bất kỳ trình xuất OTel nào. Mỗi quá trình chuyển đổi nút là một trace span; checkpoints gấp đôi traces có thể phát lại. LangSmith là tùy chọn của bên thứ nhất; Langfuse/Phoenix cũng có bộ điều hợp.
- **CrewAI** — OpenTelemetry class đầu tiên kể từ cuối năm 2025; tích hợp với Langfuse, Phoenix, Opik, AgentOps.
- **AutoGen** - Tích hợp OpenTelemetry qua `autogen-core`; AgentOps và Opik có các trình kết nối. Độ chi tiết theo dõi là trên mỗi agent tin nhắn, không phải trên mỗi nút.
- **Agno** — cờ `monitoring=True` tích hợp cộng với trình xuất OpenTelemetry; tích hợp chặt chẽ với Langfuse cho session traces.

### Chi phí và độ trễ

Tất cả bốn frameworks đều thêm chi phí cho mỗi cuộc gọi (framework logic, xác thực, tuần tự). Thứ tự tăng chi phí sơ bộ: Agno ≈ LangGraph < CrewAI ≈ AutoGen. Sự khác biệt bị chi phối bởi LLM định tuyến bổ sung mà framework thực hiện. Người quản lý phân cấp của CrewAI dành tokens quyết định ai tiếp theo; `GroupChatManager` của AutoGen cũng vậy. LangGraph chỉ dành tokens nơi bạn viết `llm.invoke`. Đường dẫn một agent của Agno rất mỏng.

Khi chi phí mỗi lần chạy quan trọng, hãy ưu tiên định tuyến rõ ràng (cạnh LangGraph, AutoGen `speaker_selection_method`) hơn định tuyến do LLM chọn.

### Khả năng tương tác

- **LangGraph** ↔ **LangChain** công cụ, chó tha mồi, LLMs. Bộ chuyển đổi class MCP đầu tiên (các công cụ được nhập dưới dạng MCP servers).
- **Các công cụ CrewAI** ↔ kế thừa từ `BaseTool`; Các công cụ LangChain, các công cụ LlamaIndex và các công cụ MCP đều thích ứng. Ủy quyền giữa phi hành đoàn với phi hành đoàn thông qua `allow_delegation=True`.
- **AutoGen** → `FunctionTool` bao bọc bất kỳ Python nào có thể gọi được; MCP bộ chuyển đổi có sẵn. Khớp nối chặt chẽ với hệ sinh thái AG2 cho các mẫu agent-agent.
- **Agno** → `@tool` decorator hoặc lớp con BaseTool; MCP adapter; các công cụ có thể được chia sẻ giữa các agents và nhóm.

## Các Skill

> Bạn có thể giải thích, trong một câu, tại sao một framework nhất định phù hợp với một vấn đề agent nhất định.

Danh sách kiểm tra trước khi xây dựng:

1. **Vẽ hình dạng.** Đây có phải là biểu đồ (trạng thái được nhập, chuyển tiếp được đặt tên) không? Một trò chơi nhập vai (các chuyên gia chuyển giao công việc)? Một cuộc trò chuyện (agents nói chuyện cho đến khi hoàn thành)? Một agent duy nhất với các công cụ?
2. **Quyết định ai branches.** Phân nhánh do nhà phát triển quyết định → LangGraph. Người quản lý agent quyết định → phân cấp CrewAI. Trò chuyện nổi lên → AutoGen. Quyết định cuộc gọi công cụ → Agno.
3. **Kiểm tra ngân sách nhà nước. **Bạn có cần sơ yếu lý lịch từ checkpoint không? Du hành thời gian? Con người làm gián đoạn giữa chừng? Nếu có, LangGraph là mặc định; Agno sessions bao gồm trạng thái phạm vi hội thoại.
4. **Kiểm tra ngân sách chi phí.** Định tuyến đã chọn LLM tốn thêm tokens cho mỗi lượt. Nếu agent chạy hàng nghìn lần một ngày, hãy ưu tiên định tuyến rõ ràng.
5. **Lập ngân sách cho chi phí framework.** Mỗi framework là một phụ thuộc khác. Nếu nhiệm vụ là hai cuộc gọi LLM và một công cụ, hãy viết 30 dòng Python đơn giản; không có framework nào rẻ hơn không có framework.

Từ chối lấy một framework trước khi bạn có thể vẽ biểu đồ, sơ đồ tổ chức, cuộc trò chuyện hoặc hộp agent. Từ chối chọn một cái buộc bạn phải chiến đấu với trạng thái của nó model cho thứ bạn thực sự cần.

## Ma trận quyết định

| Hình dạng vấn đề | Ưu tiên framework | Tại sao |
|---------------|---------------------|-----|
| DAG quy trình làm việc với trạng thái được nhập, phê duyệt của con người, chạy lâu dài | Đồ thị LangGraph | Trạng thái class đầu tiên, kiểm tra, ngắt, du hành thời gian. |
| Nghiên cứu / viết pipeline với vai trò riêng biệt | Biểu đồ con CrewAI (tuần tự) hoặc LangGraph | Vai trò trên mỗi nhiệm vụ rẻ để thể hiện trong CrewAI; mở rộng quy mô với LangGraph khi phân nhánh trở nên phức tạp. |
| Đối thoại người đề xuất-nhà phê bình hoặc giáo viên-học sinh | Tự động | Trò chuyện hai agent là hình dạng gốc của nó. |
| agent đơn với các công cụ, sessions, bộ nhớ | Bất động lực | Thiết lập mỏng nhất, bộ nhớ và bộ nhớ tích hợp. |
| Hàng ngàn fanout song song với bộ giảm tốc | LangGraph + `Send` | Người duy nhất có API điều phối song song class đầu tiên. |
| Nguyên mẫu nhanh, không cần cam kết framework | Đơn giản Python + nhà cung cấp SDK | Không có framework là framework nhanh nhất. |

## Bài tập

1. **Dễ dàng.** Thực hiện nhiệm vụ tương tự - "nghiên cứu trụ sở chính của Anthropic, viết một bản tóm tắt dài 200 từ, trích dẫn nguồn" - và thực hiện nó trong LangGraph (bốn nút: lập kế hoạch, tìm kiếm, viết, trích dẫn) và trong CrewAI (ba vai trò: nhà nghiên cứu, nhà văn, biên tập viên). Báo cáo chi phí token mỗi lần chạy và các dòng mã.
2. **Trung bình.** Xây dựng cùng một tác vụ trong AutoGen (nhà nghiên cứu ↔, người viết, trò chuyện, biên tập viên tham gia qua `GroupChat`) và Agno (một agent duy nhất với `search_tools` và `write_tools`, cộng với một cửa hàng session). Xếp hạng bốn triển khai dựa trên (a) chi phí mỗi lần chạy, (b) khả năng tiếp tục sau sự cố, (c) khả năng đưa sự chấp thuận của con người trước bước ghi.
3. **Khó.** Xây dựng một script `pick_framework.py` cây quyết định lấy một mô tả vấn đề ngắn (JSON: `{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`) và trả về một đề xuất với sự biện minh một câu. Xác minh nó trên sáu trường hợp bạn tự thiết kế.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Orchestration | "Phối hợp agents như thế nào" | Lớp quyết định node/role/agent nào sẽ chạy tiếp theo. |
| Trạng thái bền bỉ | "Tiếp tục sau khi khởi động lại" | Tiểu bang còn sót lại process chết, gắn liền với checkpoint hoặc cửa hàng session. |
| Định tuyến được chọn LLM | "Hãy để model quyết định" | Một người lập kế hoạch LLM chọn bước tiếp theo mỗi lượt; linh hoạt nhưng trả tokens cho mọi quyết định. |
| Định tuyến rõ ràng | "Nhà phát triển quyết định" | Chức năng Python hoặc cạnh tĩnh chọn bước tiếp theo; rẻ và có thể kiểm tra. |
| Phi hành đoàn | "Đội CrewAI" | Vai trò + nhiệm vụ + process (tuần tự hoặc phân cấp) được ràng buộc thành một có thể chạy được. |
| Trò chuyện nhóm | "Trò chuyện nhiều agent của AutoGen" | Một cuộc trò chuyện được quản lý giữa N agents với bộ chọn loa. |
| Đội (Agno) | "Đa agent Agno" | Chế độ định tuyến / phối hợp / cộng tác qua một tập hợp agents. |
| Biểu đồ trạng thái | "Biểu đồ của LangGraph" | Trạng thái gõ, nút, cạnh có điều kiện, trừu tượng hóa con trỏ kiểm tra. |

## Đọc thêm

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) - StateGraph, kiểm tra, ngắt, du hành thời gian.
- [CrewAI documentation](https://docs.crewai.com/) - Phi hành đoàn, Dòng chảy, Agents, Nhiệm vụ, Processes.
- [AutoGen documentation](https://microsoft.github.io/autogen/) — ConversableAgent, GroupChat, nhóm, công cụ.
- [Agno documentation](https://docs.agno.com/) — Agent, Nhóm, Quy trình làm việc, lưu trữ, bộ nhớ.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — thư viện mẫu (prompt chuỗi, định tuyến, song song, điều phối-workers, đánh giá-optimizer) framework-bất khả tri.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) - vòng lặp mà mỗi framework ăn mặc.
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — Giấy thiết kế của AutoGen.
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — nền tảng nhập vai mà nhân vật theo phong cách CrewAI stacks xây dựng trên đó.
- Giai đoạn 11 · 16 (LangGraph) — framework bài học này benchmarks chống lại.
- Giai đoạn 11 · 19 (Phản xạ) — một mẫu ánh xạ rõ ràng với LangGraph nhưng khó xử với CrewAI.
- Giai đoạn 11 · 22 (Production observability) - cách sử dụng bất kỳ framework nào bạn chọn.
