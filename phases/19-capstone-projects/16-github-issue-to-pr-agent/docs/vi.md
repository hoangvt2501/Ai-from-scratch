# Capstone 16 — GitHub Agent tự trị phát hành đến PR

> AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud và Google Jules đều ship cùng một hình dạng sản phẩm năm 2026: gắn nhãn vấn đề, nhận PR. Chạy một agent trong một cloud sandbox, xác minh các bài kiểm tra vượt qua và đăng một PR sẵn sàng xem xét với lý do. Các bộ phận cứng đang tự động tái tạo môi trường xây dựng của repo, ngăn chặn rò rỉ thông tin đăng nhập, thực thi ngân sách cho mỗi repo và đảm bảo agent không thể đẩy mạnh. Capstone này xây dựng phiên bản tự lưu trữ và so sánh nó về chi phí và tỷ lệ vượt qua với các lựa chọn thay thế được lưu trữ.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (agent), TypeScript (GitHub App), YAML (Actions)
**Kiến thức tiên quyết:** Giai đoạn 11 (LLM kỹ thuật), Giai đoạn 13 (công cụ), Giai đoạn 14 (agents), Giai đoạn 15 (tự chủ), Giai đoạn 17 (cơ sở hạ tầng)
**Các giai đoạn thực hiện: **P11 · P13 · P14 · P15 · Trang 17
**Thời lượng:** 30 giờ

## Vấn đề

agent mã hóa cloud không đồng bộ là một danh mục sản phẩm riêng biệt với agents mã hóa tương tác (capstone 01). UX là một nhãn hiệu GitHub. Bạn gắn nhãn một vấn đề `@agent fix this`, một worker quay trong một cloud sandbox, sao chép repo, chạy thử nghiệm, chỉnh sửa tệp, xác minh và mở một PR với lý do của agent trong nội dung. Không có vòng lặp tương tác, không có thiết bị đầu cuối. AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, Google Jules và Factory Droids đều hội tụ về điều này.

Các thách thức về kỹ thuật rất cụ thể: tái tạo môi trường (agent phải xây dựng repo từ đầu mà không có hình ảnh nhà phát triển được lưu trong bộ nhớ đệm), kiểm tra không rõ ràng (phải chạy lại hoặc cô lập), phạm vi thông tin xác thực (Ứng dụng GitHub có quyền chi tiết tối thiểu), thực thi ngân sách mỗi repo mỗi ngày và policy không ép đẩy. Capstone đo lường tỷ lệ vượt qua, chi phí và độ an toàn so với các lựa chọn thay thế được lưu trữ.

## Khái niệm

trigger là một GitHub webhook (nhãn vấn đề hoặc nhận xét PR). Một người điều phối đưa công việc vào ECS Fargate hoặc Lambda. worker kéo repo vào một sandbox Daytona hoặc E2B với một tệp Dockerfile chung được suy ra từ repo (ngôn ngữ, framework). agent chạy một vòng lặp mini-swe-agent hoặc SWE-agent v2 so với Claude Opus 4.7 hoặc GPT-5.4-Codex. Nó lặp lại: đọc mã, đề xuất sửa chữa, áp dụng bản vá, chạy kiểm tra.

Xác minh là bước kiểm soát. CI đầy đủ phải đi qua sandbox trước khi PR mở cửa. Vùng phủ sóng delta được tính toán; Nếu âm vượt quá ngưỡng, PR sẽ mở ra nhưng được gắn nhãn `needs-review`. agent đăng lý do dưới dạng mô tả PR cộng với một `@agent` thread người đánh giá có thể ping để theo dõi.

An toàn được giới hạn thông qua hai bề mặt GitHub khác nhau: Ứng dụng cung cấp token cài đặt ngắn với phạm vi repo contents/PR `workflows: read` và hẹp; branch bảo vệ (không phải quyền ứng dụng) thực thi "không ghi trực tiếp vào `main`" và "không buộc đẩy" - ứng dụng không bao giờ được thêm vào danh sách bỏ qua. Quyền truy cập chỉ đọc trong phạm vi đường dẫn vào `.github/workflows` không phải là một primitive ứng dụng GitHub thực sự, vì vậy danh sách cho phép của agent đối với các chỉnh sửa tệp phải thực thi điều đó ngay worker. Mức trần ngân sách mỗi repo mỗi ngày được thực thi tại người điều phối (ví dụ: tối đa 5 PR mỗi repo mỗi ngày, $20 mỗi PR).

## Kiến trúc

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## Stack

- Trigger: Ứng dụng GitHub với token chi tiết; webhook thu qua Lambda hoặc Fly.io
- Worker: Tác vụ ECS Fargate (hoặc trình chạy tự lưu trữ GitHub Actions)
- Sandbox: Daytona devcontainer hoặc E2B sandbox cho mỗi tác vụ
- Vòng lặp Agent: đường cơ sở mini-swe-agent hoặc SWE-agent v2 qua Claude Opus 4.7 / GPT-5.4-Codex
- Truy cập: tree-sitter repo-map + ripgrep
- Xác minh: đầy đủ CI trong sandbox + cổng delta phủ sóng
- Observability: Langfuse với kho lưu trữ mỗi PR trace được liên kết từ thân PR
- Ngân sách: trần đô la mỗi repo ngày; PR tối đa mỗi repo mỗi ngày

## Tự xây dựng

1. **GitHub Ứng dụng.** token cài đặt chi tiết: vấn đề đọc + ghi, pull_requests ghi, đọc nội dung + ghi, đọc quy trình làm việc. Branch bảo vệ (bề mặt duy nhất có thể làm điều này) thực thi "không đẩy trực tiếp vào `main`" và "không đẩy lực"; Ứng dụng không có trong danh sách bỏ qua. worker thực thi "không ghi theo `.github/workflows`" như một kiểm tra danh sách cho phép đối với sự khác biệt được đề xuất, vì các quyền của Ứng dụng GitHub không nằm trong phạm vi đường dẫn.

2. **Webhook bộ thu.** Hàm Lambda chấp nhận nhãn vấn đề / PR nhận xét webhooks. Lọc theo `@agent fix this` nhãn. Enqueues to SQS.

3. **Người điều phối.** Bật các nhiệm vụ từ SQS. Thực thi ngân sách mỗi repo mỗi ngày. Khởi động tác vụ ECS Fargate với URL repo, nội dung vấn đề và sandbox Daytona mới.

4. **Môi trường inference.** Phát hiện ngôn ngữ (Python, Node, Go, Rust) và trình quản lý gói (uv, pnpm, go mod, cargo). Tạo Dockerfile một cách nhanh chóng nếu không tồn tại.

5. **Agent vòng lặp.** mini-swe-agent hoặc SWE-agent v2 với Claude Opus 4.7. Công cụ: ripgrep, tree-sitter repo-map, read_file, edit_file, run_tests, git. Giới hạn cứng: 20 đô la chi phí, 30 phút đồng hồ treo tường, 30 agent lượt.

6. **Xác minh.** Sau khi vòng lặp kết thúc, hãy chạy bộ kiểm tra đầy đủ trong sandbox. Tính toán vùng phủ sóng delta thông qua jacoco / coverage.py. Nếu CI màu đỏ: dừng lại, không mở PR. Nếu độ che phủ giảm hơn 2%: mở PR có nhãn `needs-review`.

7. **PR đăng.** Đẩy agent branch. Mở PR qua GitHub API với: tiêu đề, lý do, tóm tắt khác biệt, URL trace, chi phí, lượt.

8. **Vệ sinh thông tin xác thực.** Worker chạy với token cài đặt Ứng dụng GitHub tồn tại trong thời gian ngắn. Nhật ký được dọn dẹp để tìm bí mật trước khi lưu trữ.

9. **Eval.** 30 vấn đề nội bộ có độ khó khác nhau. Đo tốc độ vượt qua, chất lượng PR (kích thước khác biệt, kiểu dáng, phạm vi), chi phí, độ trễ. So sánh với Cursor Background Agents và AWS Remote SWE Agents về cùng một vấn đề.

## Ứng dụng

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## Sản phẩm bàn giao

`outputs/skill-issue-to-pr.md` là sản phẩm được giao. Một cloud worker không đồng bộ GitHub App + biến các vấn đề được gắn nhãn thành PR sẵn sàng xem xét với chi phí giới hạn và thông tin xác thực có phạm vi.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | Tỷ lệ đậu trong 30 vấn đề | Thành công từ đầu đến cuối (CI màu xanh lá cây + phạm vi phủ sóng OK) |
| 20 | Chất lượng PR | Kích thước khác biệt, delta phạm vi, sự phù hợp kiểu |
| 20 | Chi phí và độ trễ cho mỗi vấn đề đã giải quyết | $ và đồng hồ treo tường mỗi PR |
| 20 | Sự An Toàn | Phạm vi token, ngân sách theo repo, không ép buộc, vệ sinh thông tin xác thực |
| 15 | UX của nhà điều hành | Nhận xét lý do, thử lại khả năng chi trả, @-đề cập theo dõi |
| **100** |||

## Bài tập

1. Thêm chế độ "sửa thử nghiệm bong tróc": nhãn `@agent stabilize-flake TestX` chạy thử nghiệm 50 lần trong sandbox và đề xuất một thay đổi tối thiểu để ổn định nó.

2. So sánh chi phí và Agents nền con trỏ trên ba vấn đề được chia sẻ. Báo cáo công cụ nào chiến thắng ở đâu.

3. Triển khai bảng điều khiển ngân sách: chi phí mỗi repo mỗi ngày, chi phí cho mỗi người dùng. Cảnh báo về sự bất thường.

4. Xây dựng chế độ "chạy khô" mở PR nháp mà không cần chạy CI để người đánh giá có thể kiểm tra kế hoạch với giá rẻ.

5. Thêm policy lưu giữ: PR branches cũ hơn 7 ngày không có merge sẽ tự động bị xóa.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Ứng dụng GitHub | "Nhận dạng bot có phạm vi" | Ứng dụng có quyền chi tiết + token cài đặt trong thời gian ngắn |
| cloud agent không đồng bộ | "Bối cảnh agent" | worker không tương tác chạy trong cloud sandbox chứ không phải thiết bị đầu cuối |
| Môi trường inference | "Tổng hợp tệp Docker" | Phát hiện ngôn ngữ + trình quản lý gói, tạo Dockerfile nếu không có |
| Xác minh | "CI trong sandbox" | Chạy bộ thử nghiệm đầy đủ bên trong worker trước khi mở PR |
| Phạm vi phủ sóng delta | "Bảo toàn vùng phủ sóng" | Thay đổi % độ bao phủ xét nghiệm từ cơ sở đến agent branch |
| Ngân sách mỗi repo | "Trần hàng ngày" | Giới hạn đô la và PR đếm được thực thi tại người điều phối |
| Cơ sở lý luận | "Giải thích cơ thể PR" | Agent tóm tắt về những gì đã thay đổi và tại sao; yêu cầu trong cơ thể PR |

## Đọc thêm

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — tham chiếu cloud agent không đồng bộ chính tắc
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI tham khảo
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — giải pháp thay thế thương mại
- [OpenAI Codex (cloud)](https://openai.com/codex) — đối thủ cạnh tranh được lưu trữ
- [Google Jules](https://jules.google) — Phiên bản lưu trữ của Google
- [Factory Droids](https://www.factory.ai) — tham chiếu thương mại thay thế
- [GitHub App documentation](https://docs.github.com/en/apps) — danh tính bot có phạm vi
- [Daytona cloud sandboxes](https://daytona.io) — sandbox tham khảo
