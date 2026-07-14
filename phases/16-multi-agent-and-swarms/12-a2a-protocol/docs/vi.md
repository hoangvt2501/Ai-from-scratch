# A2A — Giao thức Agent-Agent

> Google đã công bố A2A vào tháng 4 năm 2025; đến tháng 4 năm 2026, thông số kỹ thuật là https://a2a-protocol.org/latest/specification/ và 150+ tổ chức ủng hộ nó. A2A là bổ sung ngang cho MCP (Bài 13): trong đó MCP là dọc (agent ↔ công cụ), A2A là ngang hàng (agent ↔ agent). Nó xác định Thẻ Agent (khám phá), nhiệm vụ có artifacts (văn bản, dữ liệu có cấu trúc, video), vòng đời tác vụ không rõ ràng và xác thực. Production hệ thống ngày càng ghép nối MCP với A2A. Google Cloud triển khai hỗ trợ A2A vào Vertex AI Agent Builder trong giai đoạn 2025-2026.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib, `http.server`, `json`)
**Kiến thức tiên quyết:** Giai đoạn 16 · 04 (Primitive Model)
**Thời lượng:** ~75 phút

## Vấn đề

agent của bạn cần gọi cho một agent khác trên một hệ thống khác. Làm sao? Bạn có thể phơi bày một HTTP endpoint, xác định một JSON schema riêng và hy vọng phía bên kia nói điều đó. Mỗi cặp agents trở thành một tích hợp tùy chỉnh.

A2A là giao thức dây phổ quát cho cuộc gọi đó. Khám phá tiêu chuẩn, model tác vụ tiêu chuẩn, transport tiêu chuẩn, artifacts tiêu chuẩn. Giống như HTTP+REST nhưng đối với agents với tư cách là công dân class đầu.

## Khái niệm

### Bốn yếu tố

**Agent Thẻ.** Tài liệu JSON ở `/.well-known/agent.json` mô tả agent: tên, skills, endpoints, phương thức được hỗ trợ, yêu cầu xác thực. Khám phá xảy ra bằng cách đọc thẻ.

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Nhiệm vụ.** Đơn vị công việc. Một đối tượng không đồng bộ, có trạng thái với vòng đời: `submitted → working → completed / failed / canceled`. Khách hàng gửi nhiệm vụ, thăm dò ý kiến hoặc đăng ký để cập nhật.

**Artifact.** Loại kết quả được tạo ra bởi một nhiệm vụ. Văn bản, JSON có cấu trúc, hình ảnh, video, âm thanh. Artifacts được gõ để các phương thức khác nhau được class trước.

**Vòng đời mờ đục.** A2A không quy định * cách * agent từ xa giải quyết nhiệm vụ. Khách hàng thấy sự chuyển đổi và artifacts trạng thái; Việc triển khai được miễn phí sử dụng bất kỳ framework nào.

### Sự chia rẽ MCP/A2A

- **MCP** (Bài 13): agent ↔ công cụ. agent reads/writes thông qua JSON-RPC đến một công cụ server. Không có trạng thái theo mặc định.
- **A2A**: agent ↔ agent. Giao thức ngang hàng; Cả hai bên đều có agents với lý do riêng của họ.

Production hệ thống đa agent sử dụng cả hai. Một đồng nghiệp A2A gọi MCP công cụ ở bên cạnh nó. Sự chia rẽ giữ cho hai mối quan tâm trong sạch.

### Quy trình khám phá

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

Hoặc với streaming: SSE đăng ký `/tasks/{id}/events` để cập nhật đẩy.

### Xác thực

A2A hỗ trợ ba mẫu phổ biến:

- **Bearer token** — OAuth2 hoặc mờ đục.
- **mTLS** — TLS lẫn nhau; các tổ chức chứng minh danh tính với nhau.
- **Yêu cầu đã ký** — HMAC qua payload.

Auth được khai báo trong Thẻ Agent; Khách hàng khám phá và tuân thủ.

### 150+ tổ chức vào tháng 4 năm 2026

Việc áp dụng doanh nghiệp đã thúc đẩy quy mô A2A. Tiêu đề: A2A trở thành cách doanh nghiệp agent hệ thống vượt qua ranh giới tin cậy. Google Cloud shipped Vertex AI Agent Builder A2A hỗ trợ; Microsoft Agent Framework hỗ trợ nó; hầu hết các frameworks chính (LangGraph, CrewAI, AutoGen) ship A2A bộ điều hợp.

### Nơi A2A chiến thắng

- **Cuộc gọi liên tổ chức.** Agent tại công ty A gọi agent tại công ty B. Không có A2A, mỗi cặp là một hợp đồng đặt trước.
- **frameworks không đồng nhất.** LangGraph agent gọi CrewAI agent gọi Python agent tùy chỉnh. A2A bình thường hóa.
- **Đã nhập artifacts.** Kết quả video, JSON có cấu trúc, âm thanh — tất cả đều class trước.
- **Các tác vụ chạy dài.** Vòng đời mờ đục + thăm dò giúp các tác vụ kéo dài hàng giờ trở nên đơn giản.

### Nơi A2A đấu tranh

- **Cuộc gọi vi mô nhạy cảm với độ trễ.** Vòng đời của A2A không đồng bộ. agent-to-agent dưới mili giây không vừa; sử dụng RPC trực tiếp.
- **Kết hợp chặt chẽ trong process agents.** Nếu cả hai agents chạy trong cùng một Python process, chuyến đi khứ hồi HTTP của A2A là quá mức cần thiết.
- **Các nhóm nhỏ.** Chi phí thông số kỹ thuật là có thật; agents chỉ nội bộ có thể không cần hình thức.

### A2A so với ACP, ANP, NLIP

Một số thông số kỹ thuật liên quan đã xuất hiện vào năm 2024-2026:

- **ACP** (IBM/Linux Foundation) — tiền thân của A2A, phạm vi hẹp hơn.
- **ANP** (Giao thức mạng Agent) — nặng về khám phá ngang hàng, phi tập trung đầu tiên.
- **NLIP** (Giao thức tương tác ngôn ngữ tự nhiên Ecma, tiêu chuẩn hóa tháng 12 năm 2025) — loại nội dung ngôn ngữ tự nhiên.

A2A là giao thức ngang hàng được áp dụng nhiều nhất tính đến tháng 4 năm 2026. Xem arXiv:2505.02279 (Liu và cộng sự, "Khảo sát các giao thức tương tác Agent") để so sánh.

## Tự xây dựng

`code/main.py` triển khai server và máy khách tối thiểu A2A bằng cách sử dụng `http.server` và JSON. Các server:

- phơi bày `/.well-known/agent.json`,
- chấp nhận `POST /tasks`,
- quản lý trạng thái tác vụ,
- trả về artifacts trên `GET /tasks/{id}`.

Khách hàng:

- lấy Thẻ Agent,
- gửi một nhiệm vụ,
- thăm dò ý kiến cho đến khi hoàn thành,
- đọc artifact.

Chạy:

```
python3 code/main.py
```

script khởi động server trong thread nền, sau đó chạy máy khách chống lại nó. Bạn thấy quy trình hoàn chỉnh: khám phá, gửi, thăm dò ý kiến artifact.

## Ứng dụng

`outputs/skill-a2a-integrator.md` thiết kế tích hợp A2A: Nội dung thẻ Agent, schemas tác vụ, lựa chọn xác thực streaming so với thăm dò ý kiến.

## Sản phẩm bàn giao

Danh sách kiểm tra:

- **Ghim phiên bản thông số kỹ thuật.** A2A vẫn đang phát triển; Thẻ Agent sẽ khai báo phiên bản giao thức.
- **Tạo tác vụ idempotent.** Gửi trùng lặp (thử lại mạng) sẽ tạo ra một tác vụ.
- **Artifact schemas.** Khai báo những hình dạng mà agent trả về; Người tiêu dùng nên xác nhận.
- **Rate limits + auth.** A2A hướng đến công chúng; Áp dụng bảo mật web tiêu chuẩn.
- **Thư chết cho các tác vụ không thành công.** Kiểm tra các mẫu theo thời gian để tìm các loại lỗi định kỳ.

## Bài tập

1. Chạy `code/main.py`. Xác nhận khách hàng phát hiện ra server và nhận được artifact chính xác.
2. Thêm skill thứ hai vào server (ví dụ: "tóm tắt"). Cập nhật Thẻ Agent. Viết một ứng dụng khách chọn skill dựa trên loại tác vụ.
3. Triển khai SSE streaming endpoint: `/tasks/{id}/events` phát ra các thay đổi trạng thái. Khách hàng cần làm gì khác đi?
4. Đọc thông số kỹ thuật A2A (https://a2a-protocol.org/latest/specification/). Xác định ba điều mà thông số kỹ thuật yêu cầu mà bản demo này không thực hiện.
5. So sánh A2A (Khám phá thẻ Agent) với MCP (niêm yết khả năng bên server qua `listTools`). Sự đánh đổi giữa agents tự mô tả và thăm dò khả năng là gì?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| A2A | "Agent agent" | Giao thức ngang hàng để agents gọi các agents khác trên các hệ thống. Google 2025. |
| Thẻ Agent | "Danh thiếp của agent" | JSON ở `/.well-known/agent.json` mô tả skills, endpoints, xác thực. |
| Nhiệm vụ | "Đơn vị công việc" | Đối tượng có trạng thái không đồng bộ với vòng đời; artifacts sản xuất khi hoàn thành. |
| Artifact | "Kết quả" | Đầu ra được nhập: văn bản, JSON có cấu trúc, hình ảnh, video, âm thanh. Phương tiện truyền thông class thứ nhất. |
| Vòng đời mờ đục | "Cách giải quyết là công việc của agent" | Khách hàng thấy chuyển đổi trạng thái; server được tự do lựa chọn framework/tools. |
| Khám phá | "Tìm agent" | `GET /.well-known/agent.json` trả lại thẻ. |
| MCP so với A2A | "Công cụ so với đồng nghiệp" | MCP: công cụ agent ↔ dọc. A2A: ngang agent ↔ agent. |
| ACP / ANP / NLIP | "Giao thức anh chị em" | Thông số kỹ thuật liền kề; A2A là năm 2026 được áp dụng nhiều nhất. |

## Đọc thêm

- [A2A specification](https://a2a-protocol.org/latest/specification/) — thông số kỹ thuật chuẩn
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) - Bài ra mắt tháng 4 năm 2025
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — triển khai tham khảo và SDKs
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — So sánh MCP, ACP, A2A, ANP
