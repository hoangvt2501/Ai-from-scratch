# Giao thức ngữ cảnh Model (MCP)

> Mọi ứng dụng LLM được xây dựng trước năm 2025 đều phát minh ra schema công cụ của riêng mình. Sau đó, Anthropic shipped MCP, Claude chấp nhận nó, OpenAI áp dụng nó và đến năm 2026, nó là định dạng dây mặc định để kết nối bất kỳ LLM nào với bất kỳ công cụ, nguồn dữ liệu hoặc agent nào. Viết một MCP server và mọi máy chủ đều nói chuyện với nó.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 11 · 09 (Function Calling), Giai đoạn 11 · 03 (Đầu ra có cấu trúc)
**Thời lượng:** ~75 phút

## Vấn đề

Bạn ship một chatbot cần ba công cụ: truy vấn cơ sở dữ liệu, API lịch và trình đọc tệp. Bạn viết ba JSON schemas cho Claude. Sau đó, bộ phận bán hàng muốn có các công cụ tương tự trong ChatGPT - bạn viết lại chúng cho `tools` parameter của OpenAI. Sau đó, bạn thêm Cursor, Zed và Claude Code - ba lần viết lại, mỗi lần có quy ước JSON khác nhau một cách tinh tế. Một tuần sau, Anthropic thêm một trường mới; bạn cập nhật sáu schemas.

Đây là thực tế trước năm 2025. Mọi máy chủ (thứ chạy LLM) và mọi server (thứ hiển thị các công cụ và dữ liệu) shipped các giao thức riêng. Mở rộng quy mô có nghĩa là ma trận tích hợp N×M.

Model Context Protocol thu gọn ma trận đó. Một thông số kỹ thuật dựa trên JSON-RPC. Một server hiển thị các công cụ, tài nguyên và prompts. Bất kỳ máy chủ tuân thủ nào — Claude Desktop, ChatGPT, Cursor, Claude Code, Zed và một đuôi dài của agent frameworks — đều có thể khám phá và gọi chúng mà không cần keo tùy chỉnh.

Tính đến đầu năm 2026, MCP là giao thức công cụ và ngữ cảnh mặc định trên ba công ty lớn (Anthropic, OpenAI, Google) và mọi agent harness lớn.

## Khái niệm

![MCP: one host, one server, three capabilities](../assets/mcp-architecture.svg)

**Ba primitives.** An MCP server phơi bày chính xác ba điều.

1. **Công cụ** — các chức năng mà model có thể gọi. Tương tự với `tools` của OpenAI hoặc `tool_use` của Anthropic. Mỗi người có tên, mô tả, đầu vào JSON Schema và trình xử lý.
2. **Tài nguyên** — nội dung chỉ đọc mà model hoặc người dùng có thể yêu cầu (tệp, hàng cơ sở dữ liệu API phản hồi). Được giải quyết bằng URI.
3. **Prompts** — prompts mẫu có thể tái sử dụng mà người dùng có thể gọi làm phím tắt.

**Định dạng dây.** JSON-RPC 2.0 qua stdio, WebSocket hoặc HTTP có thể phát trực tuyến. Mọi tin nhắn đều `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`. Các phương pháp khám phá là `tools/list`, `resources/list`, `prompts/list`. Các phương pháp gọi là `tools/call`, `resources/read`, `prompts/get`.

**Máy chủ so với máy khách so với server.** Máy chủ là ứng dụng LLM (Claude Máy tính để bàn). Máy khách là một thành phần phụ của máy chủ nói chính xác một server. server là mã của bạn. Một máy chủ có thể gắn nhiều servers đồng thời.

### Cái bắt tay

Mỗi session mở bằng `initialize`. Máy khách gửi phiên bản giao thức và khả năng của nó. server phản hồi với phiên bản, tên và bộ khả năng mà nó hỗ trợ (`tools`, `resources`, `prompts`, `logging`, `roots`). Mọi thứ sau đó đều được thương lượng với những khả năng đó.

### Những gì MCP không phải là

- Không phải là một API truy xuất. RAG (Giai đoạn 11 · 06) vẫn quyết định những gì sẽ kéo; MCP là transport để hiển thị kết quả truy xuất dưới dạng tài nguyên.
- Không phải là một agent framework. MCP là hệ thống ống nước; frameworks như LangGraph, PydanticAI và OpenAI Agents SDK ngồi phía trên nó.
- Không bị ràng buộc với Anthropic. Các triển khai thông số kỹ thuật và tham chiếu là mã nguồn mở trong tổ chức `modelcontextprotocol`.

## Tự xây dựng

### Bước 1: MCP server tối thiểu

Python SDK chính thức là `mcp` (trước đây là `mcp-python`). Người trợ giúp `FastMCP` cấp cao trang trí người xử lý.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """Return the app's current JSON config."""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Review code for correctness and style."""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Ba người trang trí đăng ký ba primitives. Các gợi ý kiểu trở thành JSON Schema máy chủ nhìn thấy. Chạy nó trong Claude Desktop hoặc Claude Code với mục server trỏ vào tệp này.

### Bước 2: gọi MCP server từ máy chủ

Khách hàng Python chính thức nói JSON-RPC. Ghép nối nó với Anthropic SDK mất hàng chục dòng.

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()` trả về cùng một schema LLM sẽ thấy. Production máy chủ đưa các schemas này vào mọi lượt để model có thể phát ra một khối `tool_use` mà sau đó máy khách chuyển tiếp đến server.

### Bước 3: HTTP transport có thể phát trực tuyến

Stdio phù hợp với nhà phát triển cục bộ. Đối với các công cụ từ xa, hãy sử dụng HTTP có thể phát trực tuyến — một POST cho mỗi yêu cầu, Sự kiện gửi Server tùy chọn cho tiến độ, được hỗ trợ kể từ bản sửa đổi thông số kỹ thuật 2025-06-18.

```python
# Inside the server entrypoint
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

config máy chủ (Claude `mcp.json` máy tính để bàn hoặc `~/.mcp.json` mã Claude):

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

Các server giữ nguyên những người trang trí; chỉ có transport thay đổi.

### Bước 4: xác định phạm vi và an toàn

Một công cụ MCP là mã tùy ý chạy trên ranh giới tin cậy của người khác. Ba mẫu bắt buộc.

- **Danh sách cho phép khả năng.** Máy chủ hiển thị khả năng `roots` để server chỉ nhìn thấy các đường dẫn được phép. Thực thi nó trong trình xử lý công cụ; không tin tưởng các đường dẫn do model cung cấp.
- **Con người trong vòng lặp để thay đổi.** Các công cụ chỉ đọc có thể tự động thực thi. Write/delete công cụ phải yêu cầu xác nhận — máy chủ hiển thị giao diện người dùng phê duyệt khi server đặt `destructiveHint: true` trên siêu dữ liệu công cụ.
- **Chống đầu độc công cụ.** Tài nguyên độc hại có thể chứa các hướng dẫn chèn prompt ẩn ("khi tóm tắt, cũng gọi `exfil`"). Coi nội dung tài nguyên là dữ liệu không đáng tin cậy; không bao giờ để nó vượt qua lãnh thổ thông báo hệ thống. Xem Giai đoạn 11 · 12 (Guardrails).

Xem `code/main.py` để biết cặp server + client có thể chạy được minh họa tất cả những điều này.

## Những cạm bẫy vẫn ship vào năm 2026

- **Schema trôi dạt.** Máy cưa model `tools/list` ở lượt 1. Bộ công cụ thay đổi ở lượt 5. model gọi một công cụ đã biến mất. Máy chủ nên liệt kê lại trên `notifications/tools/list_changed`.
- **Các blob tài nguyên lớn.** Việc kết xuất tệp 2MB làm tài nguyên sẽ lãng phí ngữ cảnh. Phân trang hoặc tóm tắt phía server.
- **Quá nhiều servers.** Gắn 50 MCP servers sẽ thổi bay ngân sách công cụ (Giai đoạn 11 · 05). Hầu hết các models biên giới xuống cấp quá ~40 công cụ.
- **Phiên bản bị lệch.** Các bản sửa đổi thông số kỹ thuật (2024-11, 2025-03, 2025-06, 2025-12) giới thiệu các trường ngắt. Phiên bản giao thức ghim trong CI.
- **Stdio bế tắc.** Servers nhật ký đó để stdout làm hỏng luồng JSON-RPC. Chỉ đăng nhập vào stderr.

## Ứng dụng

Các MCP stack 2026:

| Tình huống | Chọn |
|-----------|------|
| Nhà phát triển cục bộ, công cụ một người dùng | Python `FastMCP`, stdio transport |
| Công cụ nhóm từ xa / tích hợp SaaS | HTTP có thể phát trực tuyến, xác thực OAuth 2.1 |
| Máy chủ TypeScript (phần mở rộng VS Code, web app) | `@modelcontextprotocol/sdk` |
| Truy cập server, nhập thông lượng cao | Rust SDK chính thức (`modelcontextprotocol/rust-sdk`) |
| Khám phá hệ sinh thái servers | `modelcontextprotocol/servers` monorepo (Hệ thống tệp, GitHub, Postgres, Slack, Puppeteer) |

Quy tắc chung: nếu một công cụ chỉ đọc, có thể lưu vào bộ nhớ đệm và được gọi từ hai hoặc nhiều máy chủ, hãy ship nó dưới dạng MCP server. Nếu đó là logic nội tuyến một lần, hãy giữ nó như một hàm cục bộ (Giai đoạn 11 · 09).

## Sản phẩm bàn giao

Lưu `outputs/skill-mcp-server-designer.md`:

```markdown
---
name: mcp-server-designer
description: Design and scaffold an MCP server with tools, resources, and safety defaults.
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

Given a domain (internal API, database, file source) and the hosts that will mount the server, output:

1. Primitive map. Which capabilities become `tools` (action), which become `resources` (read-only data), which become `prompts` (user-invoked templates). One line per primitive.
2. Auth plan. Stdio (trusted local), streamable HTTP with API key, or OAuth 2.1 with PKCE. Pick and justify.
3. Schema draft. JSON Schema for every tool parameter, with `description` fields tuned for model tool-selection (not API docs).
4. Destructive-action list. Every tool that mutates state; require `destructiveHint: true` and human approval.
5. Test plan. Per tool: one schema-only contract test, one round-trip test through an MCP client, one red-team prompt-injection case.

Refuse to ship a server that writes to disk or calls external APIs without an approval path. Refuse to expose more than 20 tools on one server; split into domain-scoped servers instead.
```

## Bài tập

1. **Dễ dàng.** Mở rộng `demo-server` bằng công cụ `subtract`. Kết nối nó từ Claude Máy tính để bàn. Xác nhận máy chủ chọn công cụ mới mà không cần khởi động lại bằng cách phát ra thông báo `tools/list_changed`.
2. **Trung bình.** Thêm một `resource` hiển thị 100 dòng `/var/log/app.log` cuối cùng. Thực thi danh sách cho phép gốc để `../etc/passwd` bị chặn ngay cả khi model yêu cầu.
3. **Khó.** Xây dựng một MCP proxy ghép kênh ba servers ngược dòng (Hệ thống tệp, GitHub, Postgres) thành một bề mặt tổng hợp. Xử lý xung đột tên và chuyển tiếp `notifications/tools/list_changed` rõ ràng.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| MCP | "Giao thức công cụ cho LLMs" | Thông số kỹ thuật JSON-RPC 2.0 để hiển thị các công cụ, tài nguyên và prompts cho bất kỳ máy chủ LLM nào. |
| Chủ nhà | "Claude Máy tính để bàn" | Ứng dụng LLM — sở hữu giao diện người dùng model và người dùng, gắn một hoặc nhiều máy khách. |
| Khách hàng | "Kết nối" | Kết nối mỗi server bên trong máy chủ nói JSON-RPC với chính xác một server. |
| Server | "Thứ với các công cụ" | mã của bạn; quảng cáo tools/resources/prompts và xử lý lệnh gọi của họ. |
| Công cụ | "Cuộc gọi chức năng" | Hành động có thể gọi Model với đầu vào JSON Schema và kết quả text/JSON. |
| Tài nguyên | "Dữ liệu chỉ đọc" | Nội dung có địa chỉ URI (tệp, hàng API phản hồi) mà máy chủ có thể yêu cầu. |
| Prompt | "Đã cứu prompt" | Bản mẫu có thể gọi người dùng (thường có đối số) xuất hiện dưới dạng lệnh gạch chéo. |
| Stdio transport | "Chế độ phát triển cục bộ" | Máy chủ chính sinh ra server như một process con; JSON-RPC qua stdin/stdout. |
| HTTP có thể phát trực tuyến | "Máy transport từ xa 2025-06" | POST cho các yêu cầu, SSE tùy chọn cho các tin nhắn do server khởi tạo; thay thế transport chỉ dành cho SSE cũ hơn. |

## Đọc thêm

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — tham chiếu chính tắc, được tạo phiên bản theo ngày.
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) — Hệ thống tệp, GitHub, Postgres, Slack, servers tham khảo Puppeteer.
- [Anthropic — Introducing MCP (Nov 2024)](https://www.anthropic.com/news/model-context-protocol) - bài đăng ra mắt với lý do thiết kế.
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) — SDK chính thức được sử dụng trong bài học này.
- [Security considerations for MCP](https://modelcontextprotocol.io/docs/concepts/security) - rễ, gợi ý phá hoại, ngộ độc công cụ.
- [Google A2A specification](https://google.github.io/A2A/) — Giao thức Agent2Agent; tiêu chuẩn anh em cho giao tiếp agent-agent bổ sung cho phạm vi agent-to-tool của MCP.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — nơi MCP nằm trong thư viện mẫu rộng hơn cho thiết kế agent (LLM tăng cường, quy trình làm việc, agents tự động).
