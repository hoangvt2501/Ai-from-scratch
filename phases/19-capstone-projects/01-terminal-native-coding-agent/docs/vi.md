# Capstone 01 — Agent mã hóa gốc đầu cuối

> Đến năm 2026, hình dạng của một agent mã hóa được giải quyết. Một harness TUI, một kế hoạch trạng thái, một bề mặt công cụ sandbox, một vòng lặp lập kế hoạch, hành động, quan sát, phục hồi. Claude Code, Cursor 3 và OpenCode đều trông giống nhau từ 50 feet. Capstone này yêu cầu bạn xây dựng một đầu đến cuối - CLI vào, pull request ra - và đo lường nó so với mini-swe-agent và Live-SWE-agent trên SWE-bench Pro. Bạn sẽ tìm hiểu lý do tại sao phần khó không phải là cuộc gọi model mà là vòng lặp công cụ, sandbox và trần chi phí khi chạy 50 lượt.

**Loại:** Đá đỉnh
**Ngôn ngữ:** TypeScript / Bun (harness), Python (eval scripts)
**Kiến thức tiên quyết:** Giai đoạn 11 (LLM kỹ thuật), Giai đoạn 13 (công cụ và giao thức), Giai đoạn 14 (agents), Giai đoạn 15 (hệ thống tự trị), Giai đoạn 17 (cơ sở hạ tầng)
**Các giai đoạn thực hiện: **P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · Trang 18
**Thời lượng:** 35 giờ

## Vấn đề

Mã hóa agents trở thành danh mục ứng dụng AI thống trị vào năm 2026. Claude Code (Anthropic), Cursor 3 với Composer 2 và Agent Tabs (Cursor), Amp (Sourcegraph), OpenCode (112 nghìn sao), Factory Droids và Google Jules đều ship biến thể của cùng một kiến trúc: harness đầu cuối, bề mặt công cụ được phép, sandbox và vòng lặp kế hoạch-hành động-quan sát được xây dựng xung quanh model biên giới. Biên giới hẹp - Live-SWE-agent đạt 79,2% trên SWE-bench Verified with Opus 4.5 - nhưng kỹ thuật rất rộng. Hầu hết các chế độ thất bại không phải là lỗi model. Chúng là sự không ổn định của vòng lặp công cụ, đầu độc ngữ cảnh, chi phí token chạy trốn và các hoạt động phá hoại hệ thống tệp.

Bạn không thể lý luận về những agents này từ bên ngoài. Bạn phải xây dựng một cú sút, xem vòng lặp gặp sự cố ở lượt 47 khi ripgrep trả về 8MB kết quả phù hợp và xây dựng lại lớp cắt bớt. Đó là điểm của capstone này.

## Khái niệm

harness có bốn bề mặt. **Plan** duy trì một đối tượng trạng thái kiểu TodoWrite mà model viết lại mỗi lượt. **Act** gửi các lệnh gọi công cụ (đọc, chỉnh sửa, chạy, tìm kiếm, git). **Observe** nắm bắt mã stdout / stderr / exit, truncates và cung cấp lại bản tóm tắt. **Khôi phục** xử lý các lỗi công cụ mà không thổi context window hoặc lặp lại mãi mãi. Hình dạng năm 2026 bổ sung thêm một điều: **hooks**. `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `Notification`, `Stop` và `PreCompact` — các điểm mở rộng có thể định cấu hình nơi người vận hành chèn policy, telemetry và guardrails.

sandbox là E2B hoặc Daytona. Mỗi tác vụ chạy trong một devcontainer mới với một cây làm việc git được gắn đọc-ghi. harness không bao giờ chạm vào hệ thống tệp máy chủ. Cây làm việc bị xé bỏ dựa trên thành công hay thất bại. Kiểm soát chi phí được thực thi ở ba lớp: trần token mỗi lượt, ngân sách mỗi session đô la và giới hạn vòng quay cứng (thường là 50). Lớp observability là OpenTelemetry spans với các quy ước ngữ nghĩa GenAI, shipped Langfuse tự lưu trữ.

## Kiến trúc

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## Stack

- Harness runtime: Bun 1.2 + Ink 5 (React-in-terminal)
- Truy cập Model: OpenRouter hợp nhất API với Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5 (dành cho các tác vụ khó nhất)
- Công cụ transport: Giao thức ngữ cảnh Model StreamableHTTP (bản sửa đổi MCP 2026)
- Sandbox: E2B sandboxes (JS SDK) hoặc Daytona devcontainers
- Tìm kiếm mã: quy trình con ripgrep, trình phân tích cú pháp tree-sitter cho 17 ngôn ngữ (biên dịch sẵn)
- Cách ly: `git worktree add` cho mỗi nhiệm vụ, dọn dẹp thành công / thất bại
- harness đánh giá: SWE-bench Pro (tập hợp con đã được xác minh) + Terminal-Bench 2.0 + 30 nhiệm vụ của riêng bạn
- Observability: OpenTelemetry SDK với `gen_ai.*` semconv → Langfuse tự lưu trữ
- Đăng PR: Ứng dụng GitHub với token chi tiết, phạm vi giới hạn ở mục tiêu repo

## Tự xây dựng

1. **TUI và vòng lặp lệnh.** Giàn giáo một dự án Bun với Ink. Chấp nhận `agent run <repo> "<task>"`. In chế độ xem phân tách: ngăn kế hoạch (trên cùng), luồng lệnh gọi công cụ (giữa) token ngân sách (dưới cùng). Thêm hủy trên Ctrl-C kích hoạt `SessionEnd` hook trước khi thoát.

2. **Trạng thái kế hoạch.** Xác định schema TodoWrite đã nhập (các mục đang chờ xử lý / in_progress / đã hoàn thành với ghi chú). Model viết lại trạng thái đầy đủ mỗi lượt dưới dạng lệnh gọi công cụ - không để nó thay đổi dần dần. Kiên trì kế hoạch `.agent/state.json` để các vụ tai nạn có thể tiếp tục.

3. **Bề mặt công cụ.** Xác định sáu công cụ: `read_file`, `edit_file` (với bản xem trước khác biệt), `ripgrep`, `tree_sitter_symbols`, `run_shell` (với timeout), `git` (trạng thái / khác biệt / commit / đẩy). Hiển thị qua MCP StreamableHTTP để harness không phụ thuộc vào transport. Mọi công cụ trả về đầu ra bị cắt ngắn (giới hạn ở mức 4k tokens mỗi cuộc gọi).

4. **Sandbox bao bọc.** Mỗi nhiệm vụ tạo ra một sandbox E2B. `git worktree add -b agent/$TASK_ID` một branch mới. Tất cả các lệnh gọi công cụ được thực hiện bên trong sandbox. Không thể truy cập được hệ thống tệp máy chủ.

5. **Hooks.** Triển khai tất cả tám loại hook 2026. Kết nối ít nhất bốn hooks do người dùng tạo ra: (a) `PreToolUse` bảo vệ lệnh phá hủy chặn `rm -rf` bên ngoài cây làm việc, (b) kế toán `PostToolUse` token, (c) khởi tạo ngân sách `SessionStart`, (d) `Stop` ghi gói trace cuối cùng.

6. **Vòng lặp đánh giá.** Sao chép một tập hợp con 30 số của SWE-bench Pro Python. Chạy harness của bạn chống lại từng người. So sánh với agent swe-mini (đường cơ sở tối thiểu) về pass@1, lượt mỗi tác vụ và $-per-task. Viết kết quả cho `eval/results.jsonl`.

7. **Kiểm soát chi phí.** Ngưỡng khó: 50 lượt, 200k ngữ cảnh, 5 đô la cho mỗi nhiệm vụ. `PreCompact` hook tóm tắt các khúc cua cũ hơn thành một khối prior ở mốc 150km, giải phóng không gian cho các quan sát mới mà không làm mất kế hoạch.

8. **PR đăng bài.** Khi thành công, bước cuối cùng là `git push` + một cuộc gọi GitHub API mở ra một PR với kế hoạch và tóm tắt khác biệt trong nội dung.

## Ứng dụng

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## Sản phẩm bàn giao

Sản phẩm skill tồn tại ở `outputs/skill-terminal-coding-agent.md`. Với một đường dẫn repo và mô tả tác vụ, nó chạy vòng lặp plan-act-observe đầy đủ trong một sandbox và trả về một URL PR cộng với một gói trace. Bảng đánh giá cho capstone này:

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | SWE-bench Pro pass@1 so với đường cơ sở | harness của bạn so với agent nhỏ trên 30 nhiệm vụ Python phù hợp |
| 20 | Kiến trúc rõ ràng | Plan/act/observe tách, bề mặt hook, công cụ schema - được xem xét dựa trên bố cục Live-SWE-agent |
| 20 | Sự An Toàn | Sandbox các bài kiểm tra trốn thoát, prompts cho phép, bảo vệ chỉ huy phá hoại vượt qua đội đỏ |
| 20 | Observability | Trace tính đầy đủ (100% lệnh gọi công cụ kéo dài) token tính toán mỗi lượt |
| 15 | UX dành cho nhà phát triển | Khởi động nguội < 2 giây, khôi phục sự cố tiếp tục kế hoạch, Ctrl-C hủy giữa công cụ một cách sạch sẽ |
| **100** |||

## Bài tập

1. Hoán đổi model hỗ trợ từ Claude Sonnet 4.7 sang Qwen3-Coder-30B được phục vụ trên vLLM. So sánh pass@1 và $-per-task. Báo cáo nơi model mở hoạt động kém hiệu quả.

2. Thêm một agent phụ `reviewer` đọc sự khác biệt trước khi PR đăng và có thể yêu cầu vòng lặp sửa đổi. Đo lường xem các đánh giá dương tính giả có làm giảm tỷ lệ đậu SWE-bench xuống dưới đường cơ sở một agent hay không (gợi ý: thường là có).

3. Kiểm tra căng thẳng sandbox: viết một tác vụ cố gắng `curl` một URL bên ngoài và một tác vụ viết bên ngoài cây làm việc. Xác nhận cả hai đều bị chặn bởi hook PreToolUse. Ghi lại các lần thử.

4. Thực hiện tóm tắt `PreCompact` với một model nhỏ hơn (Haiku 4.5). Đo mức độ trung thực của kế hoạch bị mất khi nén 3x.

5. Hoán đổi MCP StreamableHTTP transport lấy stdio. Benchmark độ trễ khởi động nguội và mỗi cuộc gọi. Chọn người chiến thắng chỉ sử dụng tại địa phương.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Harness | "Vòng lặp agent" | Mã xung quanh model gửi công cụ, duy trì trạng thái kế hoạch và thực thi ngân sách |
| Hook | "Agent người nghe sự kiện" | Một script do người dùng tạo chạy trên một trong tám sự kiện vòng đời của harness |
| Cây làm việc | "Git sandbox" | Thanh toán git được liên kết tại một đường dẫn riêng; dùng một lần mà không cần chạm vào bản sao chính |
| TodoWrite | "Trạng thái kế hoạch" | Một danh sách các mục pending/in-progress/done được nhập mà model viết lại mỗi lượt |
| Có thể phát trực tuyếnHTTP | "MCP transport" | Bản sửa đổi MCP năm 2026: kết nối HTTP lâu dài với streaming hai chiều; thay thế SSE |
| Trần Token | "Ngân sách ngữ cảnh" | Giới hạn mỗi lượt hoặc mỗi session trên tokens đầu vào + đầu ra; triggers nén hoặc chấm dứt |
| pass@1 | "Tỷ lệ vượt qua một lần thử" | Một phần các tác vụ SWE-bench được giải quyết trong lần chạy đầu tiên mà không cần thử lại hoặc xem qua bộ thử nghiệm |

## Đọc thêm

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — harness tham khảo từ Anthropic
- [Cursor 3 changelog](https://cursor.com/changelog) — Ghi chú sản phẩm Agent Tabs và Composer 2
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) - đường cơ sở tối thiểu để so sánh harness SWE-bench
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) - 79.2% SWE-bench được xác minh với Opus 4.5
- [OpenCode](https://opencode.ai) - harness mở, 112k sao
- [SWE-bench Pro leaderboard](https://www.swebench.com) - đánh giá mà capstone này nhắm mục tiêu
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, siêu dữ liệu khả năng
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — span schema để gọi công cụ và sử dụng token
