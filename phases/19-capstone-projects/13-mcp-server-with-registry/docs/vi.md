# Capstone 13 — MCP Server với Registry và Quản trị

> Giao thức ngữ cảnh Model không còn là tương lai và trở thành thông số kỹ thuật sử dụng công cụ mặc định vào năm 2026. Anthropic, OpenAI, Google và mọi khách hàng IDE ship MCP lớn. Pinterest đã công bố hệ sinh thái nội bộ của MCP servers. AAIF Registry siêu dữ liệu năng lực được chính thức hóa tại `.well-known`. AWS ECS đã phát hành triển khai không trạng thái tham chiếu. agent ngỗng của Block đặt cùng một giao thức bên trong một trợ lý được lưu trữ. Hình dạng production năm 2026 là: StreamableHTTP transport, phạm vi OAuth 2.1, cổng OPA policy và registry cho phép các nhóm nền tảng khám phá, xác thực và kích hoạt servers. Xây dựng từ đầu đến cuối.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (server, via FastMCP) or TypeScript (@modelcontextprotocol/sdk), Go (registry service)
**Kiến thức tiên quyết:** Giai đoạn 11 (LLM kỹ thuật), Giai đoạn 13 (dụng cụ và MCP), Giai đoạn 14 (agents), Giai đoạn 17 (cơ sở hạ tầng), Giai đoạn 18 (an toàn)
**Các giai đoạn thực hiện: **P11 · P13 · P14 · P17 · Trang 18
**Thời lượng:** 25 giờ

## Vấn đề

MCP trở thành ngôn ngữ chung sử dụng công cụ. Claude Code, Cursor 3, Amp, OpenCode, Gemini CLI và mọi agent được quản lý hiện đang sử dụng MCP servers. Những thách thức production không phải là tạo servers (FastMCP làm cho điều đó dễ dàng) mà là triển khai chúng trên quy mô lớn với các yêu cầu của doanh nghiệp: phạm vi OAuth trên mỗi tenant, OPA policy trên các công cụ phá hủy, mở rộng quy mô không trạng thái StreamableHTTP, registry để khám phá, nhật ký kiểm tra cho mỗi lệnh gọi công cụ. Hệ sinh thái MCP nội bộ của Pinterest và thông số kỹ thuật Registry AAIF đã thiết lập thanh năm 2026.

Bạn sẽ xây dựng một MCP server hiển thị 10 công cụ nội bộ (chỉ đọc Postgres, danh sách S3, Jira, Linear, Datadog, v.v.), giao diện người dùng registry để khám phá nền tảng và cổng phê duyệt của con người cho các công cụ phá hoại. Kiểm tra tải minh họa tỷ lệ ngang StreamableHTTP. Dấu vết kiểm tra đáp ứng đánh giá bảo mật doanh nghiệp.

## Khái niệm

Bản sửa đổi MCP 2026 yêu cầu StreamableHTTP làm transport mặc định. Không giống như hình dạng stdio-and-SSE trước đó, StreamableHTTP không có trạng thái theo mặc định: một HTTP endpoint duy nhất chấp nhận các yêu cầu JSON-RPC, phát trực tuyến phản hồi và hỗ trợ các kết nối lâu dài cho các thông báo. Stateless có nghĩa là có thể mở rộng theo chiều ngang phía sau load balancer.

Authorization là OAuth 2.1 với phạm vi trên mỗi công cụ. Một token mang các phạm vi như `jira:read`, `s3:list` `postgres:query:readonly`. MCP server kiểm tra phạm vi tại thời điểm gọi công cụ, không chỉ session bắt đầu. Đối với các công cụ có rủi ro cao, server từ chối bất kỳ cuộc gọi nào có phạm vi không được nâng lên `approved:by:human` trong vòng N phút cuối cùng - mức độ cao đó đến từ thẻ đánh giá Slack.

registry là một dịch vụ riêng biệt. Mỗi MCP server hiển thị một tài liệu `.well-known/mcp-capabilities` với tệp kê khai công cụ, URL transport, các yêu cầu xác thực. registry thăm dò ý kiến, xác thực và lập chỉ mục. Các nhóm nền tảng sử dụng giao diện người dùng registry để xem những công cụ nào có sẵn, phạm vi nào họ cần và nhóm nào sở hữu chúng.

## Kiến trúc

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## Stack

- Server framework: FastMCP (Python) hoặc `@modelcontextprotocol/sdk` (TypeScript)
- Transport: StreamableHTTP qua HTTPS (không có trạng thái)
- Auth: OAuth 2.1 với nhận dạng khối lượng công việc thông qua SPIFFE / SPIRE
- Policy: Quy tắc OPA / Rego cho mỗi công cụ; policy dịch vụ quyết định theo yêu cầu
- Registry: tự lưu trữ, tiêu thụ tệp kê khai `.well-known/mcp-capabilities`
- Sự chấp thuận của con người: Tin nhắn tương tác Slack cho các công cụ phá hoại
- Triển khai: AWS ECS Fargate hoặc Fly.io, một server mỗi tenant hoặc được chia sẻ với phạm vi tenant
- Kiểm tra: nhóm JSONL trên mỗi tenant có cấu trúc với dòng mỗi cuộc gọi

## Tự xây dựng

1. **Bề mặt công cụ.** Hiển thị 10 công cụ nội bộ: Truy vấn chỉ đọc Postgres, đối tượng danh sách S3, Jira search/fetch, search/fetch tuyến tính, truy vấn chỉ số Datadog, tra cứu theo cuộc gọi PagerDuty, chỉ đọc GitHub, Tìm kiếm khái niệm, Tìm kiếm Slack, Đọc Salesforce. Mỗi công cụ có một schema được gõ và nhãn phạm vi.

2. **FastMCP server.** Gắn các công cụ. Cấu hình transport StreamableHTTP. Thêm middleware cho OAuth token nội tâm và thực thi phạm vi.

3. **OPA policy.** Rego policy cho mỗi công cụ: phạm vi nào cho phép gọi, biên tập PII nào áp dụng, giới hạn kích thước payload nào được áp dụng. Dịch vụ quyết định được gọi trong mọi cuộc gọi công cụ.

4. **Registry dịch vụ.** Dịch vụ Go hoặc TS riêng biệt thăm dò `.well-known/mcp-capabilities` với servers đã đăng ký, xác thực bằng JSON Schema và hiển thị danh sách / tìm kiếm / xác thực / bật-tắt giao diện người dùng.

5. **Tệp kê khai khả năng.** Mỗi server hiển thị `.well-known/mcp-capabilities` với: danh sách công cụ, yêu cầu xác thực, URL transport, nhóm chủ sở hữu, SLO.

6. **Tách công cụ phá hủy.** Các công cụ thay đổi trạng thái (Jira tạo, Tạo tuyến tính, Postgres ghi) hoạt động trên MCP server thứ hai với quy trình xác thực nghiêm ngặt hơn: tokens phải có phạm vi `approved:by:human` được nâng cao thông qua thẻ Slack trong vòng 15 phút.

7. **Nhật ký kiểm tra.** JSONL chỉ thêm mỗi tenant: `{timestamp, user, tool, args_redacted, response_redacted, outcome}`. Biên tập PII qua Presidio trước khi ghi.

8. **Kiểm tra tải.** 100 máy khách đồng thời trên StreamableHTTP. Thể hiện tỷ lệ theo chiều ngang bằng cách thêm bản sao thứ hai; hiển thị load balancer phân phối lại mà không session dính.

9. **Kiểm tra sự phù hợp.** Chạy bộ tuân thủ MCP chính thức đối với cả hai servers. Vượt qua tất cả các phần bắt buộc.

## Ứng dụng

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## Sản phẩm bàn giao

`outputs/skill-mcp-server.md` mô tả sản phẩm. Lớp kiểm tra MCP server + registry + cấp production cho các công cụ nội bộ với phạm vi OAuth 2.1 và cổng OPA.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | Sự phù hợp thông số kỹ thuật | Tệp kê khai khả năng StreamableHTTP + vượt qua các bài kiểm tra tuân thủ MCP |
| 20 | Bảo mật | Thực thi phạm vi, phạm vi OPA trên mọi công cụ, vệ sinh bí mật |
| 20 | Observability | Nhật ký kiểm tra mỗi lệnh gọi công cụ với biên tập PII |
| 20 | Quy mô | Trình diễn quy mô ngang kiểm tra tải 100 máy khách |
| 15 | Registry UX | Khám phá/xác thực/bật-tắt quy trình làm việc |
| **100** |||

## Bài tập

1. Thêm một công cụ mới (Tìm kiếm hợp lưu). Ship nó thông qua quy trình xác thực registry mà không cần chạm vào server cốt lõi.

2. Viết một policy OPA biên tập kết quả truy vấn Postgres có chứa các cột có tên `email`, `ssn` hoặc `phone`. Bài tập với truy vấn thăm dò.

3. Benchmark StreamableHTTP so với stdio về độ trễ cục bộ. Báo cáo p50/p95. cho mỗi cuộc gọi

4. Triển khai hạn ngạch trên mỗi tenant: tối đa N lệnh gọi mỗi phút cho mỗi công cụ mỗi tenant. Thực thi thông qua quy tắc OPA thứ hai.

5. Chạy bộ tuân thủ MCP từ [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) và sửa mọi lỗi.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Có thể phát trực tuyếnHTTP | "2026 MCP transport" | HTTP + streaming không quốc tịch; Thay thế SSE + STDIO cho servers nối mạng |
| Bản kê khai năng lực | "Bác sĩ nổi tiếng" | `.well-known/mcp-capabilities` với danh sách công cụ, xác thực transport URL |
| OPA / Quay lại | "Động cơ Policy" | Mở Policy Agent để cho phép các lệnh gọi công cụ chống lại các quy tắc bên ngoài |
| Độ cao phạm vi | "Được con người chấp thuận" | Phạm vi tồn tại trong thời gian ngắn được cấp thông qua phê duyệt Slack, cần thiết cho các công cụ phá hoại |
| Registry | "Khám phá công cụ" | Dịch vụ lập chỉ mục MCP servers từ các tệp kê khai khả năng của chúng |
| Nhận dạng khối lượng công việc | "SPIFFE / NGỌN THÁP" | Nhận dạng dịch vụ mật mã để phát hành token OAuth |
| Bộ tuân thủ | "Kiểm tra thông số kỹ thuật" | Pin kiểm tra MCP chính thức cho tính đúng đắn của tệp kê khai công cụ StreamableHTTP + |

## Đọc thêm

- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, siêu dữ liệu khả năng registry
- [AAIF MCP Registry spec](https://github.com/modelcontextprotocol/registry) — thông số kỹ thuật registry 2026
- [AWS ECS reference deployment](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — tham khảo production triển khai
- [Pinterest internal MCP ecosystem](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — triển khai nội bộ tham chiếu
- [Block `goose` MCP usage](https://block.github.io/goose/) - tham khảo agent mô hình tiêu dùng
- [FastMCP](https://github.com/jlowin/fastmcp) — Python server framework
- [Open Policy Agent](https://www.openpolicyagent.org/) - Tham khảo động cơ policy
- [SPIFFE / SPIRE](https://spiffe.io) — tham chiếu nhận dạng khối lượng công việc
