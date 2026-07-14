# Capstone 06 — Agent khắc phục sự cố DevOps cho Kubernetes

> Agent DevOps của AWS đã chuyển sang GA, Resolve AI xuất bản playbook K8s, NeuBird trình diễn giám sát ngữ nghĩa và Metoro gắn AI SRE với SLO cho mỗi dịch vụ. Hình dạng production đã được giải quyết: một cảnh báo webhook bắn, một agent đọc telemetry, đi bộ một biểu đồ các đối tượng K8s, xếp hạng các giả thuyết nguyên nhân gốc rễ và đăng một bản tóm tắt Slack với các nút phê duyệt. Chỉ đọc theo mặc định. Mọi biện pháp khắc phục được kiểm soát bởi một con người. Điểm mấu chốt này là agent, được đánh giá trên 20 sự cố tổng hợp và so sánh với Agent của AWS trên ba trường hợp được chia sẻ.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (agent), TypeScript (Slack integration)
**Kiến thức tiên quyết:** Giai đoạn 11 (LLM kỹ thuật), Giai đoạn 13 (công cụ và MCP), Giai đoạn 14 (agents), Giai đoạn 15 (tự chủ), Giai đoạn 17 (cơ sở hạ tầng), Giai đoạn 18 (an toàn)
**Các giai đoạn thực hiện: **P11 · P13 · P14 · P15 · P17 · Trang 18
**Thời lượng:** 30 giờ

## Vấn đề

Câu chuyện SRE 2025-2026 trở thành: "AI agents các sự cố phân loại, con người phê duyệt các biện pháp khắc phục." AWS DevOps Agent, Resolve AI, NeuBird, Metoro, PagerDuty AIOps tất cả đều ship hình dạng này trong production. agent đọc các chỉ số Prometheus, nhật ký Loki, traces nhịp độ, kube-state-metrics và biểu đồ tri thức của các đối tượng K8s. Nó tạo ra một giả thuyết nguyên nhân gốc rễ được xếp hạng với telemetry trích dẫn trong vòng chưa đầy năm phút. Nó không bao giờ thực hiện các lệnh phá hoại mà không có sự chấp thuận rõ ràng của con người thông qua Slack.

Hầu hết công việc khó khăn là phạm vi và an toàn, không phải lý luận. agent cần bề mặt RBAC chỉ đọc theo mặc định, server công cụ MCP cứng và nhật ký kiểm tra của mọi lệnh được xem xét so với thực thi. Nó cần biết khi nào nó nằm ngoài độ sâu của nó và leo thang. Và nó phải chạy đủ rẻ để các tầng OOM-kill không tạo ra hóa đơn agent 5.000 đô la.

## Khái niệm

agent hoạt động trên biểu đồ tri thức. Các nút là các đối tượng K8s (Pods, Triển khai, Dịch vụ, Nút, HPA, PVC) cộng với các nguồn telemetry (dòng Prometheus, luồng Loki, Tempo traces). Các cạnh mã hóa quyền sở hữu (Pod -> ReplicaSet -> Triển khai), lập lịch (Pod -> Node) và quan sát (Pod -> Prometheus series). Biểu đồ được giữ mới bằng cách đồng bộ hóa kube-state-metrics và lấy mẫu lại trên mọi cảnh báo.

Khi cảnh báo kích hoạt, agent nguyên nhân gốc rễ từ đối tượng bị ảnh hưởng. Nó đi theo các cạnh, kéo các lát telemetry có liên quan (15 phút cuối) và phác thảo một giả thuyết. Giả thuyết được xếp hạng theo bằng chứng: có bao nhiêu trích dẫn telemetry ủng hộ nó, gần đây như thế nào, cụ thể như thế nào. 3 giả thuyết hàng đầu được chuyển đến Slack với trực quan hóa đường dẫn đồ thị và các nút phê duyệt cho các hành động khắc phục.

Khắc phục được kiểm soát. Các hành động mặc định được phép là chỉ đọc. Các hành động phá hoại (thu hẹp quy mô, khôi phục, xóa Pod) cần có sự chấp thuận của Slack; ArgoCD rollback hooks yêu cầu xác thực token agent không bao giờ nắm giữ. Nhật ký kiểm tra ghi lại mọi lệnh mà agent *xem xét* - không chỉ được thực thi - vì vậy việc xem xét process phát hiện các lỗi suýt trượt.

## Kiến trúc

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## Stack

- Nguồn Observability: Prometheus, Loki, Tempo, kube-state-metrics
- Biểu đồ tri thức: Neo4j (được quản lý) hoặc kuzu (nhúng) của các đối tượng K8s + telemetry cạnh
- Agent: LangGraph với danh sách cho phép cho mỗi công cụ, chỉ đọc theo mặc định
- Công cụ transport: FastMCP qua StreamableHTTP; server riêng biệt cho các công cụ phá hủy phía sau cổng phê duyệt
- Models: Claude Sonnet 4.7 để suy luận nguyên nhân gốc rễ, Gemini 2.5 Flash để tóm tắt nhật ký
- Khắc phục: ArgoCD rollback webhook, PagerDuty leo thang, Thẻ phê duyệt Slack
- Kiểm tra: nhật ký có cấu trúc chỉ nối (xem xét, thực thi, phê duyệt, kết quả)
- Triển khai: Triển khai K8 với vai trò RBAC hẹp của riêng nó; Không gian tên riêng biệt

## Tự xây dựng

1. **Nhập đồ thị.** Đồng bộ hóa các chỉ số trạng thái thành Neo4j/kuzu sau mỗi 30 giây. Các nút: Pod, Triển khai, Nút, Dịch vụ, PVC, HPA. Các cạnh: OWNED_BY, SCHEDULED_ON, EXPOSES, MOUNTS, SCALES. Telemetry các cạnh lớp phủ: OBSERVED_BY (một Pod được quan sát bởi một chuỗi Prometheus).

2. **Người nhận cảnh báo.** endpoint FastAPI chấp nhận webhooks PagerDuty hoặc Alertmanager. Trích xuất (các) đối tượng bị ảnh hưởng và vi phạm SLO.

3. **Bề mặt công cụ chỉ đọc.** Wrap kubectl, truy vấn Prometheus, Loki logql, Tempo traceql thông qua FastMCP. Mỗi công cụ đều có một động từ RBAC hẹp ("get", "list", "describe"). Không có "xóa", "exec", "scale" trong server mặc định.

4. **agent nguyên nhân gốc rễ.** LangGraph với ba nút: `sample` kéo lát cắt telemetry trong 15 phút cuối cùng `walk` truy vấn biểu đồ cho các đối tượng lân cận `hypothesize` nháp các ứng cử viên nguyên nhân gốc rễ được xếp hạng với telemetry trích dẫn.

5. **Chấm điểm bằng chứng.** Mỗi giả thuyết có điểm = gần đây * độ đặc hiệu * độ dài đường dẫn đồ thị nghịch đảo * số lần trích dẫn. Trở lại top-3.

6. **Tóm tắt về Slack.** Đăng tệp đính kèm với giả thuyết, trực quan hóa đường dẫn đồ thị (hình ảnh biểu đồ con được hiển thị server bên) và các nút phê duyệt cho tối đa một hành động khắc phục.

7. **Cổng khắc phục.** Các công cụ phá hoại (thu hẹp quy mô, khôi phục, xóa) hoạt động MCP server giây sau token phê duyệt. Người agent chỉ có thể gọi họ sau khi thẻ Slack được con người chấp thuận.

8. **Nhật ký kiểm tra.** JSONL chỉ nối thêm: đối với mọi lệnh ứng viên, hãy ghi lại xem lệnh đó có được xem xét hay không, có được thực thi hay không, ai đã phê duyệt. Ship đến S3 hàng ngày.

9. **Bộ sự cố tổng hợp.** Xây dựng 20 kịch bản: OOMKill cascade, DNS flap, HPA thrash, PVC fill, noisy neighbor, sidecar bị lỗi, ConfigMap rollout xấu, xoay chứng chỉ, backoff kéo hình ảnh, v.v. Chấm điểm agent về accuracy nguyên nhân gốc rễ và thời gian đưa ra giả thuyết.

## Ứng dụng

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## Sản phẩm bàn giao

`outputs/skill-devops-agent.md` là sản phẩm được giao. Với cụm K8s và nguồn cảnh báo, agent tạo ra các giả thuyết nguyên nhân gốc rễ được xếp hạng và quy trình khắc phục được kiểm soát bởi Slack.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | RCA accuracy trên bộ kịch bản | ≥80% nguyên nhân gốc rễ chính xác trong 20 sự cố tổng hợp |
| 20 | Sự An Toàn | Bảo vệ hành động phá hủy không bao giờ kích hoạt mà không có sự chấp thuận của Slack trong nhật ký kiểm tra |
| 20 | Thời gian để đưa ra giả thuyết | p50 dưới 5 phút từ cảnh báo đến Slack brief |
| 20 | Khả năng giải thích | Mọi giả thuyết đều có đường dẫn đồ thị và trích dẫn telemetry |
| 15 | Tích hợp hoàn chỉnh | PagerDuty, Slack, ArgoCD, Prometheus làm việc từ đầu đến cuối |
| **100** |||

## Bài tập

1. Chạy agent của bạn trên cùng ba sự cố mà Agent DevOps của AWS được trình diễn. Xuất bản song song. Báo cáo nơi agent phân kỳ.

2. Thêm một kiểm tra "suýt trượt" gắn cờ bất kỳ lệnh nào mà các agent *xem xét* sẽ phá hoại nếu không có sự chấp thuận. Đo tỷ lệ suýt bỏ lỡ trong một tuần.

3. Hoán đổi giả thuyết model từ Claude Sonnet 4.7 sang Llama 3.3 70B tự lưu trữ. Đo lường RCA accuracy delta và đô la cho mỗi sự cố.

4. Xây dựng bộ lọc nhân quả: phân biệt các gai telemetry tương quan với nguyên nhân gốc rễ thực sự. Huấn luyện một bộ phân loại nhỏ trên nhãn 20 kịch bản.

5. Thêm một rollback chạy khô: ArgoCD rollback chống lại một cụm dàn dựng có cùng tệp kê khai. Xác minh gói rollback trong một cụm trực tiếp trước nút phê duyệt Slack.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Biểu đồ tri thức K8s | "Biểu đồ cụm" | Các nút = đối tượng K8s + chuỗi telemetry; cạnh = quyền sở hữu, lập kế hoạch, quan sát |
| Chỉ đọc theo mặc định | "RBAC có phạm vi" | Tài khoản dịch vụ của Agent chỉ có get/list/describe động từ; Động từ phá hoại sống trong một server riêng biệt đằng sau sự chấp thuận |
| Nhật ký kiểm tra | "Đã xem xét so với đã thực hiện" | Chỉ thêm hồ sơ của mọi lệnh ứng cử viên, cho dù nó chạy, ai đã phê duyệt |
| Xếp hạng giả thuyết | "Điểm bằng chứng" | Tính gần đây × độ đặc hiệu × độ dài đường dẫn đồ thị nghịch đảo × số lần trích dẫn |
| Thẻ phê duyệt Slack | "Cổng HITL" | Tin nhắn Slack tương tác với các nút khắc phục; agent không thể tiếp tục cho đến khi con người nhấp chuột |
| Trích dẫn Telemetry | "Con trỏ bằng chứng" | Truy vấn Prometheus, bộ chọn Loki hoặc URL trace Tempo hỗ trợ xác nhận quyền sở hữu |
| MTTR | "Thời gian giải quyết" | Đồng hồ treo tường từ cảnh báo cháy đến khôi phục SLO |

## Đọc thêm

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — tài liệu tham khảo chính tắc năm 2026
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — tham chiếu đối thủ cạnh tranh
- [NeuBird semantic monitoring](https://www.neubird.ai) — cách tiếp cận đồ thị ngữ nghĩa
- [Metoro AI SRE](https://metoro.io) — Đóng khung production ưu tiên SLO
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — nguồn trạng thái cụm
- [LangGraph](https://langchain-ai.github.io/langgraph/) — tài liệu tham khảo agent người điều phối
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP server framework
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — mục tiêu khắc phục được kiểm soát
