# Claude Agent SDK: Cửa hàng Subagents và Session

> Claude Agent SDK là dạng thư viện của Claude Code harness. Các công cụ tích hợp, subagents để cách ly ngữ cảnh, hooks, lan truyền trace W3C session chẵn lẻ lưu trữ. Claude Managed Agents là giải pháp thay thế được lưu trữ cho công việc không đồng bộ chạy trong thời gian dài.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Vòng lặp Agent), Giai đoạn 14 · 10 (Skill Thư viện)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Giải thích sự khác biệt giữa SDK Anthropic Client (API thô) và Claude Agent SDK (hình dạng harness).
- Mô tả subagents - song song hóa và cô lập ngữ cảnh - và khi nào cần tiếp cận chúng.
- Đặt tên cho bề mặt cửa hàng session của Python SDK (`append`, `load`, `list_sessions`, `delete`, `list_subkeys`) và vai trò của `--session-mirror`.
- Triển khai harness stdlib với các công cụ tích hợp, subagent sinh sản với ngữ cảnh riêng biệt, hooks vòng đời và kho lưu trữ session.

## Vấn đề

Một LLM API thô giúp bạn có một chuyến khứ hồi. Một production agent cần thực thi công cụ, MCP servers, hooks vòng đời, subagent sinh sản, session persistence trace nhân giống. Claude Agent SDK ships hình dạng này dưới dạng thư viện — cùng một harness Claude mà Code sử dụng, hiển thị cho agents tuỳ chỉnh.

## Khái niệm

### SDK khách hàng so với Agent SDK

- **Client SDK (`anthropic`).** Tin nhắn thô API. Bạn sở hữu vòng lặp, các công cụ, trạng thái.
- **Agent SDK (`claude-agent-sdk`).** Thực thi công cụ tích hợp, kết nối MCP, hooks, subagent sinh sản session lưu trữ. Vòng lặp Claude Code dưới dạng thư viện.

### Công cụ tích hợp

SDK ships 10+ công cụ ra khỏi hộp: read/write tệp, shell, grep, glob, tìm nạp web, v.v. Các công cụ tùy chỉnh đăng ký thông qua giao diện schema công cụ tiêu chuẩn.

### Subagents

Hai mục đích được Anthropic ghi lại:

1. **Song song hóa.** Chạy đồng thời công việc độc lập. "Tìm tệp kiểm tra cho mỗi mô-đun trong số 20 mô-đun này" là 20 nhiệm vụ subagent song song.
2. **Cô lập ngữ cảnh.** Subagents sử dụng context window của riêng họ; chỉ có kết quả trả về trình điều phối. Ngân sách của người dàn nhạc được bảo toàn.

Python SDK bổ sung gần đây: `list_subagents()`, `get_subagent_messages()` để đọc subagent bảng điểm.

### Cửa hàng Session

Giao thức ngang bằng với TypeScript:

- `append(session_id, message)` - thêm một lượt.
- `load(session_id)` - khôi phục cuộc trò chuyện.
- `list_sessions()` - liệt kê.
- `delete(session_id)` - với thác đến subagent sessions.
- `list_subkeys(session_id)` - liệt kê subagent phím.

`--session-mirror` (cờ CLI) phản chiếu bản chép lời sang tệp bên ngoài khi phát trực tuyến để gỡ lỗi.

### Hooks

Vòng đời hooks bạn có thể đăng ký:

- `PreToolUse`, `PostToolUse` — lệnh gọi cổng hoặc công cụ kiểm tra.
- `SessionStart`, `SessionEnd` - thiết lập và phá bỏ.
- `UserPromptSubmit` - hành động dựa trên đầu vào của người dùng trước khi model nhìn thấy nó.
- `PreCompact` — chạy trước khi nén ngữ cảnh.
- `Stop` - dọn dẹp khi agent ra.
- `Notification` — cảnh báo kênh bên.

Hooks là cách quy trình làm việc chuyên nghiệp (tham khảo chương trình giảng dạy Giai đoạn 14) và các hệ thống tương tự bổ sung hành vi xuyên suốt.

### Bối cảnh trace W3C

OTel spans hoạt động trên trình gọi truyền vào quy trình con CLI thông qua tiêu đề ngữ cảnh trace W3C. Toàn bộ nhiều process trace hiển thị dưới dạng một trace trong phần phụ trợ của bạn.

### Claude Agents được quản lý

Phương án thay thế được lưu trữ (tiêu đề beta `managed-agents-2026-04-01`). Công việc không đồng bộ chạy lâu, bộ nhớ đệm prompt tích hợp, nén tích hợp. Kiểm soát thương mại cho cơ sở hạ tầng được quản lý.

### Mô hình này sai ở đâu

- **Subagent xuất hiện quá mức.** Sinh sản 100 subagents cho 100 nhiệm vụ nhỏ. Trên cao chiếm ưu thế. Thay vào đó, Batch.
- **Hook creep.** Mỗi đội thêm hooks; bong bóng thời gian khởi động. Xem lại hooks hàng quý.
- **Session chướng bụng.** Sessions tích lũy; kích thước tăng lên. Sử dụng `list_sessions` + policy hết hạn.

## Tự xây dựng

`code/main.py` triển khai hình dạng SDK trong stdlib:

- `Tool`, `ToolRegistry` với `read_file`, `write_file` `list_dir` tích hợp.
- `Subagent` — ngữ cảnh riêng tư, chạy riêng biệt, kết quả được trả về.
- `SessionStore` — thêm, tải, liệt kê, xóa list_subkeys.
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- Một bản demo: agent chính sinh ra 3 subagents song song (mỗi cô lập), tổng hợp kết quả, tồn tại session.

Chạy nó:

```
python3 code/main.py
```

trace hiển thị subagent cách ly ngữ cảnh (kích thước ngữ cảnh của trình điều phối vẫn bị giới hạn), thực thi hook và session persistence.

## Ứng dụng

- **Claude Agent SDK** dành cho các sản phẩm ưu tiên Claude muốn có hình dạng harness Mã Claude.
- **Claude Agents được quản lý** cho công việc không đồng bộ được lưu trữ trong thời gian dài.
- **OpenAI Agents SDK** (Bài 16) dành cho các đối tác OpenAI trước.
- **LangGraph + công cụ tùy chỉnh** nếu bạn muốn thay thế máy trạng thái hình đồ thị.

## Sản phẩm bàn giao

`outputs/skill-claude-agent-scaffold.md` giàn giáo một ứng dụng Claude Agent SDK với subagents, hooks, cửa hàng session, tệp đính kèm MCP server và truyền trace W3C.

## Bài tập

1. Thêm một subagent spawner batches 20 nhiệm vụ thành các nhóm gồm 5 subagents song song. Đo lường kích thước ngữ cảnh của trình điều phối so với một tác vụ.
2. Triển khai một `PreToolUse` hook giới hạn tốc độ `write_file` cuộc gọi (5 cuộc gọi mỗi phút mỗi session). Trace hành vi.
3. Wire `list_subkeys` để hiển thị cây subagent. Làm tổ sâu trông như thế nào?
4. Chuyển đồ chơi vào gói `claude-agent-sdk` Python thật. Những thay đổi nào về đăng ký công cụ?
5. Đọc tài liệu Claude Managed Agents. Khi nào bạn sẽ chuyển từ tự lưu trữ sang quản lý?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Agent SDK | "Claude Code như một thư viện" | Hình dạng Harness: dụng cụ, MCP, hooks, subagents session cửa hàng |
| Subagent | "Trẻ em agent" | Bối cảnh riêng biệt, ngân sách riêng; kết quả bong bóng lên |
| Cửa hàng Session | "Cơ sở dữ liệu hội thoại" | Kiên trì, tải, liệt kê, xóa lượt với subagent tầng |
| Hook | "Vòng đời callback" | Pre/post công cụ, session, prompt gửi, thu gọn, dừng |
| Bối cảnh trace W3C | "Cross-process trace" | span mẹ lan truyền vào quy trình con CLI |
| Quản lý Agents | "Được lưu trữ harness" | Công việc không đồng bộ chạy lâu dài được lưu trữ Anthropic |
| `--session-mirror` | "Gương bảng điểm" | Ghi session chuyển sang tệp bên ngoài khi chúng phát trực tuyến |
| MCP server | "Bề mặt dụng cụ" | Nguồn tool/resource bên ngoài gắn vào agent |

## Đọc thêm

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — dạng thư viện của Claude Code
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) - production mẫu
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — giải pháp thay thế được lưu trữ
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — đối tác
