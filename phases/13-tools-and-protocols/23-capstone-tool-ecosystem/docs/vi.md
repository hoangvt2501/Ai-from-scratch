# Capstone — Xây dựng một hệ sinh thái công cụ hoàn chỉnh

> Giai đoạn 13 dạy từng phần. Capstone này kết nối chúng vào một hệ thống hình production: MCP server với các công cụ + tài nguyên + prompts + tác vụ + giao diện người dùng, OAuth 2.1 ở biên, gateway RBAC, máy khách đa server, lệnh gọi agent phụ A2A, theo dõi OTel vào bộ thu, phát hiện ngộ độc dụng cụ trong CI và gói AGENTS.md + SKILL.md. Cuối cùng, bạn có thể bảo vệ mọi lựa chọn kiến trúc.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, end-to-end ecosystem harness)
**Kiến thức tiên quyết:** Giai đoạn 13 · 01 đến 21
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Soạn một MCP server hiển thị các công cụ, tài nguyên, prompts và tác vụ bằng ứng dụng `ui://`.
- Dẫn đầu server bằng gateway OAuth 2.1 thực thi RBAC và hàm băm được ghim.
- Viết một ứng dụng khách đa server traces với các thuộc tính OTel GenAI từ đầu đến cuối.
- Ủy thác một phần khối lượng công việc cho một agent phụ A2A; Xác minh độ mờ được giữ nguyên.
- Đóng gói toàn bộ stack bằng AGENTS.md + SKILL.md để các agents khác có thể điều khiển.

## Vấn đề

Ship hệ thống "nghiên cứu và báo cáo":

- Người dùng hỏi: "tóm tắt ba bài báo arXiv năm 2026 được trích dẫn nhiều nhất về các giao thức agent."
- Hệ thống: tìm kiếm arXiv qua MCP; ủy quyền tóm tắt bài cho một người viết chuyên ngành agent qua A2A; kết quả tổng hợp; hiển thị báo cáo tương tác dưới dạng tài nguyên `ui://` ứng dụng MCP; đăng nhập từng bước vào OTel.

Tất cả các primitives từ Giai đoạn 13 đều xuất hiện. Đây không phải là một món đồ chơi — production hệ thống trợ lý nghiên cứu shipped vào năm 2026 bởi Anthropic (sản phẩm Claude Research), OpenAI (GPT với Ứng dụng SDK) và các bên thứ ba có hình dạng chính xác này.

## Khái niệm

### Kiến trúc

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### Trace phân cấp

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

Một trace id. Mỗi span đều có các thuộc tính `gen_ai.*` phù hợp.

### Vị thế bảo mật

- OAuth 2.1 + PKCE với chỉ báo tài nguyên ghim đối tượng vào gateway.
- Gateway nắm giữ thông tin đăng nhập thượng nguồn; người dùng không bao giờ nhìn thấy chúng.
- RBAC: `alice` có `research:read`, `research:write`, có thể gọi tất cả các công cụ. `bob` có `research:read`, không thể gọi `generate_report`.
- Tệp kê khai mô tả được ghim: bỏ bất kỳ server nào có hàm băm công cụ thay đổi.
- Kiểm tra Quy tắc Hai: không có công cụ nào kết hợp đầu vào không đáng tin cậy, dữ liệu nhạy cảm và hành động do hậu quả.

### Kết xuất

Tác vụ `generate_report` cuối cùng trả về các khối nội dung cộng với tài nguyên `ui://report/current`. Máy chủ của máy khách (Claude Máy tính để bàn, v.v.) hiển thị bảng điều khiển tương tác trong iframe sandbox. Bảng điều khiển chứa danh sách giấy được sắp xếp, số lượng trích dẫn và nút gọi `host.callTool('summarize_paper', {arxiv_id})` cho bất kỳ bài báo nào mà người dùng nhấp vào.

### Bao bì

Toàn bộ sự việc ships như:

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

Người dùng triển khai với `docker compose up`. Người dùng Claude Code, Cursor, Codex và opencode có thể điều khiển hệ thống bằng cách gọi `run-research` skill.

### Mỗi bài học Giai đoạn 13 đã đóng góp gì

| Bài học | Capstone sử dụng gì |
|--------|------------------------|
| 01-05 | Giao diện công cụ, tính di động của nhà cung cấp, cuộc gọi song song, schemas linting |
| 06-10 | MCP primitives, server, máy khách, transports, tài nguyên + prompts |
| 11-14 | Sampling, roots + elicitation, tác vụ không đồng bộ `ui://` ứng dụng |
| 15-17 | Ngộ độc dụng cụ, OAuth 2.1, gateway + registry |
| 18 | A2A đoàn phụ agent |
| 19 | Truy tìm OTel GenAI |
| 20 | Định tuyến gateway cho lớp LLM |
| 21 | Bao bì SKILL.md + AGENTS.md |

## Ứng dụng

`code/main.py` khâu các mẫu của các bài học trước thành một bản demo có thể chạy được. Tất cả các stdlib, tất cả đều process để bạn có thể đọc nó từ đầu đến cuối. Nó chạy toàn bộ quy trình cho kịch bản nghiên cứu và báo cáo: bắt tay với gateway, mô phỏng OAuth 2.1, tools/list merged, generate_report dưới dạng tác vụ, A2A cuộc gọi đến người viết, ui:// tài nguyên được trả về, OTel spans phát ra.

Những gì cần xem:

- Một trace id trên mỗi bước nhảy.
- Gateway policy chặn người dùng thứ hai viết.
- Vòng đời tác vụ hoạt động → hoàn thành và trả về cả văn bản và nội dung ui://.
- Trạng thái bên trong của A2A gọi là mờ đục đối với người điều phối.
- AGENTS.md và SKILL.md là các tệp duy nhất mà một agent khác cần để tái tạo quy trình làm việc.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-ecosystem-blueprint.md`. Với nhu cầu sản phẩm (nghiên cứu, tóm tắt, tự động hóa), skill tạo ra kiến trúc đầy đủ: MCP primitives nào, gateway kiểm soát, A2A gọi nào, telemetry nào, đóng gói nào.

## Bài tập

1. Chạy `code/main.py`. Lưu ý id trace đơn và cách lồng spans. Đếm xem bản demo chạm vào bao nhiêu primitives từ Giai đoạn 13.

2. Mở rộng bản demo: thêm MCP server phụ trợ thứ hai (ví dụ: `bibliography`) và xác nhận gateway merges các công cụ của nó vào cùng một không gian tên.

3. Thay thế agent người viết A2A giả bằng một người thật đang chạy trên một quy trình con. Sử dụng bài học 19 harness.

4. Thêm bước biên tập PII trong gateway định tuyến giữa trình điều phối và LLM. Xác nhận email trong truy vấn của người dùng sẽ bị xóa.

5. Viết một AGENTS.md cho một đồng đội sẽ duy trì hệ thống này. Sẽ mất chưa đầy năm phút để đọc và cung cấp cho họ mọi thứ họ cần để điều khiển capstone trong Cursor hoặc Codex.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Đá đỉnh | "Bản demo tích hợp giai đoạn 13" | Hệ thống đầu cuối sử dụng mọi primitive |
| Nghiên cứu và báo cáo | "Kịch bản" | Tìm kiếm, tóm tắt, hiển thị mẫu |
| Hệ sinh thái | "Tất cả các mảnh ghép cùng nhau" | Server + khách hàng + gateway + agent phụ + telemetry + gói |
| Trace phân cấp | "ID trace độc thân" | Mỗi span của hoa bia đều chia sẻ trace; Cha mẹ-con qua ID span |
| token do Gateway cấp | "Xác thực chuyển tiếp" | Khách hàng chỉ nhìn thấy token của gateway; gateway nắm giữ uy tín thượng nguồn |
| Không gian tên Merged | "Tất cả các công cụ trong một danh sách phẳng" | Nhiều server merge ở gateway, tiền tố khi va chạm |
| Ranh giới độ mờ | "Cuộc gọi A2A ẩn nội bộ" | Lý luận của Sub-agent vô hình đối với người điều phối |
| stack ba lớp | "AGENTS.md + SKILL.md + MCP" | Ngữ cảnh dự án + quy trình làm việc + công cụ |
| Phòng thủ chuyên sâu | "Nhiều lớp bảo mật" | Hàm băm được ghim, OAuth, RBAC, Quy tắc hai, nhật ký kiểm tra |
| Ma trận tuân thủ thông số kỹ thuật | "Những gì chúng tôi ship rằng thông số kỹ thuật yêu cầu" | Danh sách kiểm tra ánh xạ sản phẩm theo yêu cầu 2025-11-25 |

## Đọc thêm

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — tham chiếu hợp nhất
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — giao thức đang hướng đến đâu
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — Tham khảo A2A v1.0
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — quy ước theo dõi chuẩn
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) - production agent runtime mẫu
