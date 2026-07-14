# Skills và Agent SDKs — Anthropic Skills, AGENTS.md OpenAI Ứng dụng SDK

> MCP nói "những công cụ nào tồn tại". Skills nói "làm thế nào để thực hiện một nhiệm vụ". 2026 stack lớp cả hai. Agent Skills của Anthropic (tiêu chuẩn mở, tháng 12 năm 2025) ship là SKILL.md với công bố thông tin lũy tiến. Ứng dụng SDK của OpenAI là MCP siêu dữ liệu tiện ích cộng với tiện ích. AGENTS.md (hiện có 60.000+ repos) nằm ở gốc repo dưới dạng ngữ cảnh agent cấp dự án. Bài học này đặt tên cho những gì mỗi gói bao gồm và xây dựng một gói SKILL.md + AGENTS.md tối thiểu di chuyển qua agents.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, SKILL.md parser and loader)
**Kiến thức tiên quyết:** Giai đoạn 13 · 07 (MCP server)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Phân biệt ba lớp: AGENTS.md (bối cảnh dự án), SKILL.md (bí quyết có thể tái sử dụng), MCP (công cụ).
- Viết một SKILL.md với YAML mặt trước và tiết lộ tiến bộ.
- Tải kiểu hệ thống tệp skills vào một agent runtime.
- Soạn một skill với một MCP server và một AGENTS.md để một gói hoạt động trong Mã, Con trỏ và Codex Claude.

## Vấn đề

Một kỹ sư chắt lọc quy trình viết ghi chú phát hành thành một prompt nhiều bước: "Đọc PR merged mới nhất. Nhóm theo khu vực. Tóm tắt từng cái. Viết một mục nhật ký thay đổi theo phong cách của nhóm. Đăng lên bản nháp Slack." Họ đưa nó vào tài liệu Notion cho nhóm của họ.

Bây giờ họ muốn sử dụng quy trình làm việc này từ Claude Code, Cursor và Codex CLI. Mỗi agent có một cách tải hướng dẫn khác nhau: Claude Lệnh gạch chéo Code, Quy tắc con trỏ, Codex `.codex.md`. Kỹ sư sao chép quy trình làm việc ba lần và duy trì ba bản sao.

AGENTS.md và SKILL.md cùng nhau khắc phục vấn đề này:

- **AGENTS.md** nằm ở gốc repo. Mọi agent tương thích đều đọc nó khi session đầu. "Dự án này hoạt động như thế nào? Các quy ước là gì? Lệnh nào chạy thử nghiệm?"
- **SKILL.md** là một gói di động: YAML frontmatter (tên, mô tả) + nội dung đánh dấu + tài nguyên tùy chọn. Agents hỗ trợ đó skills tải chúng theo tên theo yêu cầu.
- **MCP** (Giai đoạn 13 · 06-14) xử lý các công cụ mà skill cần gọi.

Ba lớp, một artifact di động.

## Khái niệm

### AGENTS.md (agents.md)

Ra mắt vào cuối năm 2025, được 60.000+ repos chấp nhận vào tháng 4 năm 2026. Một tệp ở repo gốc. Định dạng:

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

Agents đọc điều này khi bắt đầu session và sử dụng nó để hiệu chỉnh hành vi của họ cho dự án đó. Mọi agent lập trình vào năm 2026 đều hỗ trợ AGENTS.md: Claude Code, Cursor, Codex, Copilot Workspace, opencode, Windsurf, Zed.

### Định dạng SKILL.md

Agent Skills của Anthropic (phát hành dưới dạng tiêu chuẩn mở tháng 12 năm 2025):

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter tuyên bố danh tính của skill. Thân máy là prompt hiển thị cho model khi skill tải.

### Tiết lộ lũy tiến

Skills có thể tham chiếu các tài nguyên phụ mà agent chỉ tìm nạp khi cần. Ví dụ:

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md nói "xem style-guide.md để biết các quy tắc về phong cách". agent chỉ kéo style-guide.md khi skill đang hoạt động. Điều này tránh làm đầy prompt với chi tiết mà model có thể không cần.

### Khám phá hệ thống tệp

Agent runtimes quét các thư mục đã biết để tìm các tệp SKILL.md:

- `~/.anthropic/skills/*/SKILL.md`
- `./skills/*/SKILL.md` dự án
- `~/.claude/skills/*/SKILL.md`

Tải theo tên thư mục và `name` frontmatter. Claude Code, Anthropic Claude Agent SDK và SkillKit (cross-agent) đều tuân theo mẫu này.

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk` (TypeScript) và `claude-agent-sdk` (Python) tải skills khi session khởi động, hãy hiển thị chúng dưới dạng "agents" có thể gọi bên trong runtime. Vòng lặp agent gửi đến một skill khi người dùng gọi nó.

### OpenAI Ứng dụng SDK

Ra mắt Tháng Mười 2025; được xây dựng trực tiếp trên MCP. Hợp nhất các Trình kết nối prior và Hành động GPT tùy chỉnh của OpenAI dưới một giao diện dành cho nhà phát triển duy nhất. Ứng dụng Apps SDK là:

- Một MCP server (công cụ, tài nguyên, prompts).
- Cộng với siêu dữ liệu tiện ích cho giao diện người dùng của ChatGPT.
- Cộng với tài nguyên MCP Apps `ui://` tùy chọn cho các nền tảng tương tác.

Cùng một giao thức, UX phong phú hơn.

### Tính di động agent chéo qua SkillKit

Các công cụ như SkillKit và các lớp phân phối chéo agent tương tự dịch một SKILL.md duy nhất sang định dạng gốc của mỗi AI agents 32+ (Claude Code, Cursor, Codex, Gemini CLI, OpenCode, v.v.). Một nguồn sự thật; nhiều người tiêu dùng.

### stack ba lớp

| Lớp | Tập tin | Tải khi | Mục đích |
|-------|------|-------------|---------|
| AGENTS.md | repo rễ | session bắt đầu | Quy ước cấp dự án |
| SKILL.md | skills thư mục | skill được gọi | Quy trình làm việc có thể tái sử dụng |
| MCP server | process bên ngoài | Công cụ cần thiết | Hành động có thể gọi |

Cả ba đều soạn thảo: agent đọc AGENTS.md khi session bắt đầu, người dùng gọi một skill, hướng dẫn của skill bao gồm các lệnh gọi công cụ MCP, agent gửi qua ứng dụng khách MCP.

## Ứng dụng

`code/main.py` ships trình phân tích cú pháp và trình tải SKILL.md stdlib. Nó phát hiện ra skills dưới `./skills/`, phân tích phần trước YAML cộng với nội dung đánh dấu và tạo ra một dict được khóa theo tên skill. Sau đó, nó mô phỏng một vòng lặp agent gọi `release-notes-writer` theo tên.

Những gì cần xem:

- YAML frontmatter được phân tích cú pháp với trình phân tích cú pháp stdlib tối thiểu (không phụ thuộc `pyyaml`).
- Skill được lưu trữ nguyên văn; agent thêm nó vào system prompt về lời kêu gọi.
- Tiết lộ liên tục được trình diễn thông qua chức năng `read_subresource` lấy các tệp được tham chiếu theo yêu cầu.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-agent-bundle.md`. Với quy trình làm việc, skill tạo ra gói bản thiết kế SKILL.md + AGENTS.md + MCP server kết hợp, có thể di động trên agents.

## Bài tập

1. Chạy `code/main.py`. Thêm skill thứ hai dưới `skills/` và xác nhận bộ nạp đã nhặt nó.

2. Viết một AGENTS.md cho khóa học này repo. Bao gồm các lệnh thử nghiệm, quy ước phong cách và model tinh thần Giai đoạn 13.

3. Chuyển quy trình làm việc nhiều bước từ tài liệu nội bộ của nhóm vào SKILL.md. Xác minh xem nó tải trong Claude Code.

4. Dịch skill sang định dạng quy tắc gốc của Cursor và Codex bằng tay. Đếm sự khác biệt giữa các định dạng - đây là bề mặt dịch mà SkillKit tự động hóa.

5. Đọc bài đăng trên blog Anthropic Agent Skills. Xác định một feature trong Claude Agent SDK mà trình tải bài học này không đề cập. (Gợi ý: agent gọi phụ.)

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| SKILL.md | "Tập tin skill" | YAML phần trước cộng với thân máy đánh dấu, được tải bởi agent runtime |
| AGENTS.md | "Ngữ cảnh agent gốc Repo" | Tệp quy ước cấp dự án được đọc khi bắt đầu session |
| Tiết lộ lũy tiến | "Tải tài nguyên phụ lười biếng" | Skill tệp tham chiếu nội dung chỉ được kéo khi cần thiết |
| Mặt trận | "YAML khối ở trên cùng" | Siêu dữ liệu (tên, mô tả) trong dấu phân cách `---` |
| Claude Agent SDK | "Anthropic skill runtime" | `@anthropic-ai/claude-agent-sdk`, tải skills và tuyến đường |
| OpenAI Ứng dụng SDK | "MCP + meta tiện ích" | Bề mặt phát triển của OpenAI được xây dựng trên MCP cộng với hooks giao diện người dùng ChatGPT |
| Khám phá Skill | "Quét hệ thống tệp" | Đi bộ các dirs được biết đến cho SKILL.md, chìa khóa theo tên |
| Tính di động agent chéo | "Một skill nhiều agents" | Dịch một SKILL.md sang 32+ agents thông qua các công cụ kiểu SkillKit |
| Agent Skill | "Bí quyết di động" | Mẫu tác vụ có thể tái sử dụng bên ngoài khái niệm công cụ của MCP |
| Ứng dụng SDK | "MCP cộng với giao diện người dùng ChatGPT" | Trình kết nối và GPT tùy chỉnh được thống nhất trên MCP |

## Đọc thêm

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) - Ra mắt tháng 12 năm 2025
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — tham chiếu định dạng SKILL.md
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — Nền tảng dành cho nhà phát triển dựa trên MCP dành cho ChatGPT
- [agents.md](https://agents.md/) — định dạng AGENTS.md và danh sách áp dụng
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — ví dụ skill chính thức
