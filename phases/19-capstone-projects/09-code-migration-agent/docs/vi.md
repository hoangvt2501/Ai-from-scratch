# Capstone 09 — Di chuyển mã Agent (Ngôn ngữ cấp Repo / Nâng cấp Runtime)

> MigrationBench của Amazon (Java 8 đến 17) và trình di chuyển Py2-to-Py3 của App Engine của Google đã đặt ra tiêu chuẩn năm 2026. OpenRewrite của Moderne thực hiện viết lại AST xác định trên quy mô lớn. Grit nhắm mục tiêu vào vấn đề tương tự với DSL kiểu codemod. Mẫu production kết hợp cả hai: một chất nền xác định để viết lại an toàn cộng với một lớp agent cho các trường hợp mơ hồ, một sandbox cho các bản dựng mỗi branch và một harness thử nghiệm lật màu xanh lá cây trước khi PR mở ra. Điểm mấu chốt là di chuyển 50 repos thực và công bố tỷ lệ đậu với phân loại thất bại.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (agent), Java / Python (targets), TypeScript (dashboard)
**Kiến thức tiên quyết:** Giai đoạn 5 (NLP), Giai đoạn 7 (transformers), Giai đoạn 11 (LLM kỹ thuật), Giai đoạn 13 (công cụ), Giai đoạn 14 (agents), Giai đoạn 15 (tự chủ), Giai đoạn 17 (cơ sở hạ tầng)
**Các giai đoạn thực hiện: **P5 · P7 · P11 · P13 · P14 · P15 · Trang 17
**Thời lượng:** 30 giờ

## Vấn đề

Di chuyển mã quy mô lớn là một trong những ứng dụng production sạch nhất của agents mã hóa 2026. ground truth là rõ ràng (bộ thử nghiệm có vượt qua sau khi di chuyển không?), phần thưởng là có thật (di chuyển hạm đội Java-8 là một dự án quy mô số lượng nhân viên) và benchmarks là công khai (tập hợp con MigrationBench 50-repo). OpenRewrite của Moderne xử lý khía cạnh xác định. Lớp agent xử lý mọi thứ mà các công thức OpenRewrite không thể làm được: viết lại mơ hồ, trôi dạt hệ thống xây dựng, cú pháp đuôi dài, phá vỡ phần phụ thuộc chuyển tiếp.

Bạn sẽ xây dựng một agent sử dụng Java 8 repo (hoặc Python 2 repo) và tạo ra một branch di chuyển CI xanh. Bạn sẽ đo lường tỷ lệ đỗ, bảo toàn phạm vi kiểm tra, chi phí mỗi repo và xây dựng phân loại thất bại. Song song với đường cơ sở chỉ xác định cho bạn biết giá trị của agent thực sự sống ở đâu.

## Khái niệm

pipeline có hai lớp. **Deterministic substrate** (OpenRewrite cho Java, libcst cho Python) chạy phần lớn các lần viết lại cơ học một cách an toàn: imports, chữ ký phương thức, chỉnh sửa an toàn rỗng, thử với tài nguyên, thay thế API không dùng nữa. Nó nhanh và tạo ra sự khác biệt có thể kiểm tra. Lớp **agent** (OpenAI Agents SDK hoặc LangGraph qua Claude Opus 4.7 và GPT-5.4-Codex) xử lý các trường hợp mà các công thức không thể: nâng cấp tệp xây dựng (Maven/Gradle/pyproject), xung đột phụ thuộc chuyển tiếp, mảnh thử nghiệm, chú thích tùy chỉnh.

Mỗi repo nhận được một sandbox Daytona với mục tiêu runtime cài đặt sẵn. Các agent lặp lại: chạy bản dựng, phân loại lỗi, áp dụng sửa lỗi, chạy lại. Giới hạn cứng: 30 phút mỗi repo, 8 đô la mỗi repo, 20 agent lượt. Nếu tất cả các bài kiểm tra đều vượt qua và delta phạm vi phủ sóng không âm tính, branch sẽ mở ra một PR. Nếu không, repo sẽ được nộp theo class không có bằng chứng.

Phân loại thất bại là sản phẩm phân phối. Trong 50 repos, điều gì đã bị hỏng? Deps chuyển tiếp? Chú thích tùy chỉnh? Xây dựng phiên bản công cụ? Các mảnh thử nghiệm không liên quan đến di cư? Mỗi class nhận được một số đếm và một sự khác biệt mẫu mực. Các tác giả công thức trong tương lai có thể nhắm mục tiêu ba người hàng đầu.

## Kiến trúc

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## Stack

- Chất nền xác định: OpenRewrite (Java) hoặc libcst (Python)
- Agent: OpenAI Agents SDK hoặc LangGraph qua Claude Opus 4.7 + GPT-5.4-Codex
- Sandbox: Daytona devcontainers per branch, runtime mục tiêu được cài đặt sẵn (Java 17 / Python 3.12)
- Xây dựng hệ thống: Maven, Gradle uv (Python)
- Benchmarks: Tập hợp con Amazon MigrationBench 50-repo (Java 8 đến 17), Google App Engine Py2-to-Py3 repos
- harness thử nghiệm: chạy song song, phủ sóng qua Jacoco (Java) hoặc coverage.py (Python)
- Observability: Langfuse + trace gói mỗi repo với mỗi đoạn khác biệt
- Bảng điều khiển: bảng điều khiển phân loại thất bại với số lượng trên mỗi class và sự khác biệt mẫu

## Tự xây dựng

1. **Công thức vượt qua.** Chạy công thức OpenRewrite (Java) hoặc libcst (Python) trước. Nắm bắt 70-80% di chuyển máy móc. Commit là "công thức" commit.

2. **Bản dùng thử bản dựng.** Daytona sandbox: cài đặt runtime mục tiêu, chạy bản dựng. Nếu màu xanh lá cây, hãy chuyển sang bài kiểm tra. Nếu màu đỏ, hãy giao cho agent.

3. **Agent vòng lặp.** LangGraph với các công cụ: `run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`. Agent phân loại lỗi (dep, syntax, test, build-tool) và áp dụng bản sửa lỗi có mục tiêu. Chạy lại.

4. **Giới hạn ngân sách.** Đồng hồ treo tường 30 phút mỗi repo, chi phí 8 đô la, 20 agent lượt. Mọi vi phạm sẽ tạm dừng và files trong "budget_exhausted" với diff hiện tại.

5. **Cổng kiểm tra + vùng phủ sóng.** Sau khi bản dựng chuyển sang màu xanh lá cây, hãy chạy bộ thử nghiệm. So sánh phạm vi bảo hiểm với repo cơ sở. Nếu phạm vi bảo hiểm giảm hơn 2%, hãy nộp hồ sơ dưới "coverage_regression".

6. **PR mở.** Khi thành công, hãy đẩy branch, mở PR với sự khác biệt và tóm tắt công thức nấu ăn nào được áp dụng và công thức nào commits agent tác giả.

7. **Phân loại thất bại.** Đối với mỗi repo thất bại, hãy gắn thẻ bằng class: `dep_upgrade_required`, `build_tool_drift`, `custom_annotation`, `test_flake`, `syntax_edge_case`, `budget_exhausted`. Xây dựng bảng điều khiển.

8. **Chạy 50 repo.** Thực thi trên tập hợp con MigrationBench. Báo cáo tỷ lệ đậu trên mỗi class, chi phí mỗi repo, bảo toàn phạm vi và đường cơ sở chỉ so sánh và xác định.

## Ứng dụng

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## Sản phẩm bàn giao

`outputs/skill-migration-agent.md` là sản phẩm được giao. Cho một repo, nó thực hiện các công thức xác định sau đó là một vòng lặp agent để tạo ra một branch di chuyển màu xanh lá cây hoặc nộp repo theo class phân loại.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | MigrationTỷ lệ vượt qua băng ghế dự bị | Tập con 50 repo pass@1 |
| 20 | Bảo quản phạm vi xét nghiệm | Phạm vi phủ sóng trung bình delta so với cơ sở |
| 20 | Chi phí cho mỗi repo di chuyển | $/repo khi vượt qua |
| 20 | Tích hợp Agent / công cụ xác định | Phần các bản sửa lỗi mà OpenRewrite đã xử lý so với agent tác giả |
| 15 | Viết phân tích lỗi | Tính hoàn chỉnh của phân loại với các ví dụ |
| **100** |||

## Bài tập

1. Chạy pipeline di chuyển chỉ với OpenRewrite (không có agent). So sánh tỷ lệ đậu với toàn bộ pipeline. Xác định các trường hợp mà chỉ riêng agent là sự khác biệt.

2. Thực hiện kiểm tra "lint-sạch": sau khi di chuyển, hãy chạy một linter kiểu (spotless cho Java, ruff cho Python). Không đạt PR nếu lỗi lint mới xuất hiện. Đo lường tỷ lệ bảo toàn nhưng thoái lui kiểu.

3. Thêm optimizer "khác biệt tối thiểu": sau khi branch của agent vượt qua các bài kiểm tra, hãy cắt bớt các thay đổi không cần thiết bằng lần thứ hai. Báo cáo giảm kích thước chênh lệch.

4. Mở rộng sang lần di chuyển thứ ba: Nút 18 đến Nút 22. Tái sử dụng bao bì sandbox; Hoán đổi lớp công thức cho một codemod tùy chỉnh.

5. Đo lường thời gian xây dựng xanh đầu tiên (TTFGB) dưới dạng chỉ số trải nghiệm người dùng. Mục tiêu: p50 dưới 10 phút.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Chất nền xác định | "Công cụ công thức" | OpenRewrite / libcst: viết lại AST khai báo với đảm bảo an toàn |
| Mã mod | "Chương trình sửa đổi mã" | Quy tắc viết lại thay đổi mã nguồn một cách cơ học |
| Xây dựng drift | "Phiên bản công cụ bị lệch" | Thay đổi hành vi của Maven / Gradle / uv tinh tế giữa các phiên bản chính |
| Thất bại class | "Nhóm phân loại" | Một lý do được gắn nhãn khiến repo không di chuyển: dep, cú pháp, kiểm tra, công cụ xây dựng, ngân sách |
| Phạm vi phủ sóng delta | "Bảo toàn vùng phủ sóng" | Thay đổi % phạm vi xét nghiệm từ cơ sở sang branch di chuyển |
| Agent lượt | "Vòng gọi công cụ" | Một kế hoạch -> hành động -> quan sát chu kỳ trong vòng lặp agent |
| Cạn kiệt ngân sách | "Chạm trần nhà" | repo đã tiêu tốn giới hạn 30 phút / 8 đô la / 20 lượt mà không vượt qua |

## Đọc thêm

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) - benchmark kinh điển năm 2026
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — tham chiếu chất nền xác định
- [OpenRewrite documentation](https://docs.openrewrite.org) - tác giả công thức
- [Grit.io](https://www.grit.io) - DSL codemod thay thế
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — tài liệu tham khảo Agents SDK
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — benchmark di chuyển thay thế
- [libcst](https://github.com/Instagram/LibCST) — Python chất nền xác định
- [Daytona sandboxes](https://daytona.io) — tham khảo trên mỗi branch sandbox
