# LangGraph — Máy trạng thái cho Agents

> Vòng lặp ReAct được viết bằng tay là một `while True`. Vòng lặp ReAct được viết bằng LangGraph là một biểu đồ mà bạn có thể checkpoint, ngắt quãng, branch và du hành thời gian. agent không thay đổi. Các harness xung quanh nó có.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 11 · 09 (Function Calling), Giai đoạn 11 · 14 (Model Giao thức ngữ cảnh)
**Thời lượng:** ~75 phút

## Vấn đề

Bạn ship một agent gọi hàm. Nó hoạt động trong ba lượt, sau đó xảy ra sự cố: model thử một công cụ trả về 500, người dùng thay đổi ý định giữa nhiệm vụ hoặc agent quyết định hoàn lại tiền cho đơn đặt hàng mà không có con người đăng ký. Vòng lặp `while True:` không có hooks. Bạn không thể tạm dừng nó, bạn không thể tua lại nó và bạn không thể branch vào "điều gì sẽ xảy ra nếu model đã chọn công cụ khác". Khoảnh khắc bạn ship qua một bản demo, agent sẽ trở thành một hộp đen hoạt động hoặc không.

Bước tiếp theo là rõ ràng khi bạn nhìn thấy nó. agent đã là một cỗ máy trạng thái - system prompt cộng với lịch sử tin nhắn cộng với các lệnh gọi công cụ đang chờ xử lý cộng với hành động tiếp theo. Làm cho máy trạng thái rõ ràng: các nút cho "người model nghĩ", "một công cụ chạy", "một con người phê duyệt" và các cạnh cho các chuyển đổi có điều kiện giữa chúng. Một khi biểu đồ rõ ràng, harness nhận được bốn thứ miễn phí: checkpoint (lưu trạng thái giữa các bước), ngắt (tạm dừng cho con người), streaming (luồng tokens và các sự kiện trung gian) và du hành thời gian (tua lại trạng thái prior và thử một branch khác).

LangGraph là thư viện ships trừu tượng này. Nó không phải là một agent framework theo nghĩa LangChain ("đây là một AgentExecutor, chúc may mắn"). Nó là một runtime đồ thị với các ngắt trạng thái class nhất, class persistence đầu tiên và class đầu tiên. Vòng lặp agent là thứ bạn vẽ, không phải là thứ bạn viết tay.

## Khái niệm

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

Một `StateGraph` có ba điều.

1. **Trạng thái.** Một dict được gõ (TypedDict hoặc Pydantic model) chảy qua biểu đồ. Mỗi nút nhận được trạng thái đầy đủ và trả về một bản cập nhật một phần, mà LangGraph merges sử dụng *reducer* cho mỗi trường — `operator.add` cho các danh sách sẽ tích lũy, ghi đè theo mặc định.
2. **Các nút.** Python chức năng `state -> partial_state`. Mỗi bước là một bước rời rạc: "gọi model", "chạy công cụ", "tóm tắt".
3. **Cạnh.** Chuyển đổi giữa các nút. Các cạnh tĩnh đi một nơi. Các cạnh có điều kiện sử dụng chức năng bộ định tuyến `state -> next_node_name` để biểu đồ có thể branch trên đầu ra model.

Bạn biên dịch biểu đồ. Compile liên kết cấu trúc liên kết, đính kèm một con trỏ kiểm tra (tùy chọn nhưng cần thiết cho production) và trả về một trạng thái có thể chạy được. Bạn gọi nó với trạng thái ban đầu và một `thread_id`. Mỗi bước thực thi vẫn tồn tại một checkpoint được khóa trên `(thread_id, checkpoint_id)`.

### Bốn siêu năng lực

**Điểm kiểm tra.** Mỗi quá trình chuyển đổi nút sẽ ghi trạng thái mới vào một kho lưu trữ (trong bộ nhớ cho các bài kiểm tra, Postgres/Redis/SQLite cho sản phẩm). Tiếp tục bằng cách gọi lại biểu đồ với cùng một `thread_id`. Biểu đồ chọn nơi nó đã tạm dừng.

**Ngắt.** Đánh dấu một nút bằng `interrupt_before=["human_review"]` và quá trình thực thi sẽ dừng trước khi nút đó chạy. Trạng thái vẫn tồn tại. API của bạn phản hồi người dùng bằng "đang chờ phê duyệt". Một yêu cầu sau đó đến cùng một `thread_id` với `Command(resume=...)` sẽ tiếp tục thực thi.

**Streaming.** `graph.stream(state, mode="updates")` mang lại các delta trạng thái khi chúng xảy ra. `mode="messages"` phát trực tuyến LLM tokens bên trong các nút model. `mode="values"` mang lại ảnh chụp nhanh đầy đủ. Bạn chọn nội dung sẽ hiển thị trong giao diện người dùng của mình.

**Du hành thời gian.** `graph.get_state_history(thread_id)` trả về nhật ký checkpoint đầy đủ. Chuyển bất kỳ prior `checkpoint_id` nào cho `graph.invoke` và bạn sẽ phân nhánh từ điểm đó. Tuyệt vời để gỡ lỗi ("điều gì sẽ xảy ra nếu model đã chọn công cụ B thay thế?") và cho các bài kiểm tra hồi quy phát lại production traces.

### Bộ giảm tốc là điểm

Mỗi trường trạng thái đều có một bộ giảm tốc. Hầu hết các giá trị mặc định đều ổn - một giá trị mới ghi đè lên giá trị cũ. Nhưng danh sách thông báo cần `operator.add` để các thông báo mới được thêm vào thay vì thay thế. Các cạnh song song merge cập nhật của chúng thông qua bộ giảm tốc. Nếu hai nút đều cập nhật `messages` và bạn quên `Annotated[list, add_messages]`, nút thứ hai sẽ thắng một cách âm thầm và bạn thua một nửa lượt. Bộ giảm là thứ tinh tế duy nhất trong thư viện; làm đúng và rest soạn thảo.

### Biểu đồ ReAct trong bốn nút

Một agent production ReAct là bốn nút và hai cạnh:

1. `agent` - gọi LLM với lịch sử tin nhắn hiện tại. Trả về tin nhắn trợ lý (có thể chứa tool_calls).
2. `tools` — thực thi bất kỳ tool_calls nào trong thông báo trợ lý cuối cùng, thêm kết quả công cụ dưới dạng thông báo công cụ.
3. Một cạnh có điều kiện từ `agent` định tuyến đến `tools` nếu thông báo cuối cùng có tool_calls, nếu không đến `END`.
4. Một cạnh tĩnh từ `tools` trở lại `agent`.

Đó là nó. Bạn nhận được vòng lặp ReAct đầy đủ (Suy nghĩ → Hành động → Quan sát → Suy nghĩ → ...) với điểm kiểm tra, ngắt và streaming, trong khoảng 40 dòng mã.

### StateGraph so với Send (phân xuất)

`Send(node_name, state)` cho phép một nút gửi các biểu đồ con song song. Ví dụ: agent quyết định truy vấn ba người truy xuất cùng một lúc. Mỗi `Send` tạo ra một thực thi song song của nút đích; đầu ra của chúng merge thông qua bộ giảm trạng thái. Đây là cách LangGraph thể hiện mẫu workers điều phối mà không cần phân luồng primitives.

### Biểu đồ con

Biểu đồ được biên dịch có thể là một nút trong một biểu đồ khác. Biểu đồ bên ngoài nhìn thấy một nút duy nhất; biểu đồ bên trong có trạng thái riêng và checkpoints riêng. Đây là cách các nhóm xây dựng worker agents giám sát: biểu đồ giám sát định tuyến ý định của người dùng đến biểu đồ con worker trên mỗi miền.

## Tự xây dựng

### Bước 1: trạng thái và nút

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` là bộ giảm làm cho danh sách tin nhắn tích lũy thay vì ghi đè. Quên rằng đó là lỗi LangGraph phổ biến nhất.

### Bước 2: chạy với thread

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

Mỗi bản cập nhật là một `{node_name: state_delta}`. Giao diện người dùng của bạn có thể truyền những thứ này đến giao diện người dùng để người dùng thấy "agent đang suy nghĩ... gọi search_web... có kết quả... trả lời."

### Bước 3: thêm ngắt human-in-the-loop

Đánh dấu một nút để quá trình thực thi tạm dừng trước khi chạy.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

Trạng thái, checkpoint và thread đều tồn tại qua ngắt. Không có gì trong bộ nhớ ngoại trừ trong quá trình thực thi.

### Bước 4: du hành thời gian để gỡ lỗi

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

Truyền `None` khi đầu vào phát lại từ checkpoint nhất định; truyền một giá trị sẽ thêm giá trị đó dưới dạng cập nhật cho trạng thái của checkpoint đó trước khi tiếp tục. Đây là cách bạn tái tạo một agent chạy không tốt mà không chạy lại toàn bộ cuộc trò chuyện.

### Bước 5: hoán đổi con trỏ kiểm tra lấy production

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis và Postgres là shipped. `MemorySaver` dành cho các thử nghiệm. Bất cứ thứ gì tồn tại sau khi khởi động lại đều muốn có một cửa hàng thực sự.

## Các Skill

> Bạn xây dựng agents dưới dạng đồ thị, không phải dưới dạng vòng lặp `while True`.

Trước khi bạn sử dụng LangGraph, hãy thiết kế 60 giây:

1. **Đặt tên cho các nút.** Mọi quyết định rời rạc hoặc hành động tác dụng phụ đều là một nút. "Agent nghĩ", "công cụ chạy", "người đánh giá phê duyệt", "luồng phản hồi". Nếu bạn không thể liệt kê chúng, nhiệm vụ chưa được định hình agent.
2. **Khai báo trạng thái.** Minimal TypedDict với một bộ giảm cho mọi trường danh sách. Không nhồi nhét mọi thứ vào `messages`; nâng các trường theo nhiệm vụ cụ thể (`plan` làm việc, bộ đếm `budget`, danh sách `retrieved_docs`) lên cấp cao nhất.
3. **Vẽ các cạnh. **Tĩnh trừ khi bước tiếp theo phụ thuộc vào đầu ra model. Mọi cạnh có điều kiện đều cần một chức năng bộ định tuyến có tên branches.
4. **Chọn một con trỏ kiểm tra trước.** `MemorySaver` cho các bài kiểm tra Postgres/Redis/SQLite cho bất kỳ thứ gì khác. Đừng ship mà không có một con trỏ kiểm tra - không có con trỏ kiểm tra có nghĩa là không có sơ yếu lý lịch, không gián đoạn, không du hành thời gian.
5. **Quyết định ngắt trước khi các công cụ chạy, không phải sau đó. **Phê duyệt đi vào một nút tác dụng phụ để bạn có thể hủy trước khi bị hại; xác thực đi trên biên ra khỏi model để bạn có thể từ chối các cuộc gọi xấu với giá rẻ.
6. **Phát trực tuyến theo mặc định.** `mode="updates"` cho giao diện người dùng, `mode="messages"` cho streaming cấp token bên trong các nút model `mode="values"` cho ảnh chụp nhanh đầy đủ trong quá trình đánh giá.

Từ chối ship một agent LangGraph không có con trỏ kiểm tra. Từ chối ship một cái làm gián đoạn * sau * tác dụng phụ. Từ chối ship một trường `messages` mà không có `add_messages` làm bộ giảm của nó.

## Bài tập

1. **Dễ dàng.** Triển khai biểu đồ ReAct bốn nút ở trên bằng công cụ máy tính và công cụ tìm kiếm web. Xác minh rằng `list(app.get_state_history(config))` trả về ít nhất bốn checkpoints cho cuộc trò chuyện hai lượt.
2. **Trung bình.** Thêm một nút `planner` chạy trước `agent` và ghi một `plan: list[str]` có cấu trúc vào trạng thái. Yêu cầu `agent` đánh dấu các bước kế hoạch là đã hoàn thành. Không đạt kiểm tra nếu `plan` bị mất trong sơ yếu lý lịch checkpoint (giảm sai).
3. **Khó.** Xây dựng một biểu đồ giám sát định tuyến giữa ba biểu đồ con (`researcher`, `writer`, `reviewer`) bằng cách sử dụng `Send`. Mỗi biểu đồ con có trạng thái và điểm kiểm tra riêng. Thêm một `interrupt_before=["writer"]` trên biểu đồ bên ngoài để con người có thể phê duyệt bản tóm tắt nghiên cứu. Xác nhận rằng du hành thời gian từ một prior checkpoint chỉ chạy lại branch đã phân nhánh.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Biểu đồ trạng thái | "Biểu đồ LangGraph" | Đối tượng trình tạo mà bạn thêm các nút và cạnh vào trước khi biên dịch. |
| Giảm tốc | "Lĩnh vực merges như thế nào" | Một hàm `(old, new) -> merged` áp dụng khi một nút trả về bản cập nhật cho trường đó; mặc định là ghi đè `add_messages` thêm vào. |
| Thread | "ID cuộc trò chuyện" | Một chuỗi `thread_id` phạm vi tất cả checkpoints cho một session. |
| Checkpoint | "Trạng thái tạm dừng" | Ảnh chụp nhanh liên tục của trạng thái biểu đồ đầy đủ sau khi chuyển đổi nút, được khóa trên `(thread_id, checkpoint_id)`. |
| Ngắt | "Tạm dừng vì một con người" | `interrupt_before` / `interrupt_after` dừng thực hiện tại ranh giới nút; tiếp tục với `Command(resume=...)`. |
| Du hành thời gian | "Ngã ba từ một bước prior" | `graph.invoke(None, config_with_old_checkpoint_id)` phát lại từ đó checkpoint trở đi. |
| Gửi | "Công văn con song song" | Một hàm khởi tạo một nút có thể quay trở lại để sinh ra N lần thực thi song song của một nút đích. |
| Biểu đồ con | "Một biểu đồ được biên dịch dưới dạng một nút" | Một StateGraph đã biên dịch được sử dụng làm nút trong một biểu đồ khác; giữ nguyên phạm vi trạng thái của riêng nó. |

## Đọc thêm

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — tham chiếu chuẩn cho StateGraph, bộ giảm tốc, điểm kiểm tra và ngắt.
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) - model tinh thần mà bài học này sử dụng, trực tiếp từ nguồn.
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — thông tin chi tiết về các cửa hàng Postgres/SQLite/Redis, không gian tên checkpoint và ID thread.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`, `interrupt_after`, `Command(resume=...)` và mẫu trạng thái chỉnh sửa.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — mẫu mà mọi agent LangGraph triển khai; đọc nó để biết lý do trace lý do.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — hình dạng đồ thị nào (chuỗi, bộ định tuyến, trình điều phối-workers, trình đánh giá-optimizer) để thích và khi nào.
- Giai đoạn 11 · 09 (Function Calling) - lệnh gọi công cụ primitive mọi nút LangGraph agent sử dụng lại.
- Giai đoạn 11 · 14 (Giao thức ngữ cảnh Model) - khám phá công cụ bên ngoài cắm vào `ToolNode` LangGraph thông qua bộ điều hợp MCP.
- Giai đoạn 11 · 17 (Agent framework đánh đổi) — khi nào nên chọn LangGraph thay vì CrewAI, AutoGen hoặc Agno.
