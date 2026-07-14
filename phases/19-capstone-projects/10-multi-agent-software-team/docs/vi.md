# Capstone 10 — Nhóm kỹ thuật phần mềm đa Agent

> Kiến trúc nhà máy của SWE-AF, prompting dựa trên vai trò của MetaGPT, biểu đồ diễn viên được đánh máy của AutoGen 0.4, Devin của Cognition và Droids của Factory đều hội tụ trên cùng một hình dạng năm 2026: một kiến trúc sư lập kế hoạch, N lập trình viên làm việc trong các cây làm việc song song, một cổng đánh giá, một người thử nghiệm xác minh. Cây làm việc song song chuyển đổi đồng hồ treo tường thành thông lượng. Các giao thức trạng thái và chuyển giao được chia sẻ trở thành bề mặt lỗi. Điểm mấu chốt là xây dựng đội, đánh giá trên SWE-bench Pro và báo cáo những bàn giao nào bị hỏng và tần suất như thế nào.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python / TypeScript (agents), Shell (worktree scripts)
**Kiến thức tiên quyết:** Giai đoạn 11 (LLM kỹ thuật), Giai đoạn 13 (công cụ), Giai đoạn 14 (agents), Giai đoạn 15 (tự chủ), Giai đoạn 16 (đa agent), Giai đoạn 17 (cơ sở hạ tầng)
**Các giai đoạn thực hiện: **P11 · P13 · P14 · P15 · P16 · Trang 17
**Thời lượng:** 40 giờ

## Vấn đề

Mã hóa một agent harnesses đạt mức trần đối với các tác vụ lớn. Không phải vì bất kỳ agent riêng lẻ nào yếu, mà vì ngữ cảnh 200k token không thể chứa một kế hoạch kiến trúc cộng với bốn lát cắt cơ sở mã song song cộng với bình luận của người đánh giá cộng với đầu ra thử nghiệm. Các nhà máy đa agent phân chia vấn đề: một kiến trúc sư sở hữu kế hoạch, một lập trình viên sở hữu việc triển khai trong các cây làm việc song song, một người đánh giá cổng, một người kiểm tra xác minh. Kiến trúc "nhà máy" của SWE-AF, vai trò của MetaGPT, biểu đồ diễn viên được đánh máy của AutoGen — cả ba khung đều mô tả cùng một hình dạng.

Bề mặt hỏng hóc là bàn giao. Kiến trúc sư lập kế hoạch một cái gì đó mà các lập trình viên không thể thực hiện. Coder tạo ra các diff mâu thuẫn. Người đánh giá phê duyệt một bản sửa lỗi ảo giác. Tester chạy đua với một lập trình viên vẫn đang viết. Bạn sẽ xây dựng một trong những đội này, điều hành nó trên 50 vấn đề SWE-bench Pro, theo dõi mọi bàn giao và xuất bản kết quả khám nghiệm.

## Khái niệm

Vai trò được gõ agents. **Architect** (Claude Opus 4.7) đọc vấn đề, viết kế hoạch và chia nó thành các nhiệm vụ con với các giao diện rõ ràng. **Lập trình viên** (Claude Sonnet 4.7, N phiên bản song song, mỗi phiên bản trong `git worktree` + Daytona sandbox) thực hiện các tác vụ con một cách độc lập. **Người đánh giá** (GPT-5.4) đọc sự khác biệt merged và phê duyệt hoặc yêu cầu các thay đổi cụ thể. **Tester **(Gemini 2.5 Pro) chạy bộ kiểm tra một cách riêng biệt và báo cáo pass/fail với artifacts.

Giao tiếp thông qua bảng tác vụ được chia sẻ (được hỗ trợ bằng tệp hoặc Redis). Mỗi vai trò tiêu thụ các tác vụ mà nó được phép xử lý. Handoff là các tin nhắn kiểu giao thức A2A. Mối quan tâm về điều phối: giải quyết merge xung đột (vai trò điều phối viên hoặc merge ba chiều tự động), đồng bộ hóa trạng thái chia sẻ (kế hoạch bị đóng băng khi lập trình viên bắt đầu; lập kế hoạch lại là các sự kiện riêng biệt) và người đánh giá (người đánh giá không thể phê duyệt các thay đổi của chính mình hoặc các thay đổi mà họ đề xuất).

Token khuếch đại là chi phí ẩn. Mỗi ranh giới vai trò đều thêm prompts tóm tắt và ngữ cảnh bàn giao. Một lần chạy một agent 40 lượt trở thành tổng cộng 160 lượt trên bốn vai trò. Bảng đánh giá đặc biệt cân nhắc hiệu quả token so với đường cơ sở một agent vì câu hỏi không phải là "nhiều agent có hoạt động không" mà là "nó có thắng trên mỗi đô la không".

## Kiến trúc

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## Stack

- Orchestration: LangGraph với trạng thái được chia sẻ + biểu đồ con trên agent
- Nhắn tin: Giao thức A2A (Google 2025) cho các tin nhắn liên agent được nhập
- Models: Opus 4.7 (kiến trúc sư), Sonnet 4.7 (lập trình viên), GPT-5.4 (người đánh giá), Gemini 2.5 Pro (người kiểm tra)
- Cách ly Worktree: `git worktree add` cho mỗi lập trình viên + Daytona sandbox
- Điều phối viên Merge: merge ba chiều tùy chỉnh + giải quyết xung đột qua trung gian LLM
- Đánh giá: SWE-bench Pro (50 số), kịch bản SWE-AF, HumanEval++ để kiểm tra đơn vị
- Observability: Langfuse với tính năng spans được gắn thẻ vai trò, kế toán theo agent token
- Triển khai: K8 với mỗi vai trò là Triển khai + HPA riêng biệt trên backlog

## Tự xây dựng

1. **Bảng tác vụ.** JSONL được hỗ trợ bằng tệp với các thông báo được nhập: `plan_request`, `subtask`, `diff_ready`, `review_needed`, `test_needed`, `approved`, `rejected`, `replan_needed`. Agents đăng ký thẻ.

2. **Architect.** Đọc vấn đề GitHub, chạy Opus 4.7 với một mẫu kế hoạch yêu cầu các giao diện tác vụ con rõ ràng (các tệp được chạm, chức năng công khai, tác động thử nghiệm). Phát ra một `plan_request` với DAG của các nhiệm vụ phụ.

3. **Lập trình viên.** N workers song song, mỗi người yêu cầu một nhiệm vụ con từ bảng. Mỗi con sinh ra một `git worktree add` branch tươi cộng với một sandbox Daytona. Thực hiện nhiệm vụ con. Phát ra `diff_ready` với miếng dán + thử nghiệm deltas.

4. **Merge điều phối viên.** Trên tất cả các lập trình viên hoàn thành, ba chiều merges N branches thành một branch dàn dựng. Giải quyết xung đột qua trung gian LLM chỉ khi tồn tại chồng chéo cấp tệp.

5. **Người đánh giá.** GPT-5.4 đọc sự khác biệt merged. Không thể phê duyệt các diff mà nó là tác giả. Phát ra `approved` (no-op) hoặc `review_feedback` với các yêu cầu thay đổi cụ thể được định tuyến trở lại bộ lập trình có liên quan.

6. **Tester.** Gemini 2.5 Pro chạy bộ thử nghiệm trong một sandbox sạch. Chụp artifacts. Phát ra `test_passed` hoặc `test_failed` với các dấu vết ngăn xếp. Các bài kiểm tra không thành công sẽ lặp lại với lập trình viên sở hữu tác vụ con không thành công.

7. **Kế toán bàn giao.** Mỗi tin nhắn vượt qua ranh giới vai trò sẽ nhận được span trong Langfuse với payload kích thước và model sử dụng. Tính toán trên mỗi tác vụ con token khuếch đại (coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens).

8. **Đánh giá.** Chạy trên 50 vấn đề SWE-bench Pro. So sánh pass@1 và $-per-solved-issue với đường cơ sở một agent (một Sonnet 4.7 trong một worktree duy nhất).

9. **Xử lý.** Đối với mỗi vấn đề không thành công, hãy xác định bàn giao bị hỏng (kế hoạch quá mơ hồ, merge xung đột, người đánh giá phê duyệt sai, vảy thử nghiệm). Tạo biểu đồ bàn giao-thất bại.

## Ứng dụng

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## Sản phẩm bàn giao

`outputs/skill-multi-agent-team.md` là sản phẩm được giao. Với URL vấn đề và mức độ song song, nhóm tạo ra một PR sẵn sàng cho merge với kế toán token theo từng vai trò.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | SWE-bench Pro pass@1 | Tập hợp con 50 số phù hợp, pass@1 |
| 20 | Tăng tốc song song | Đồng hồ treo tường so với đường cơ sở một agent |
| 20 | Chất lượng đánh giá | Tỷ lệ phê duyệt sai trên đầu dò lỗi được tiêm |
| 20 | Token hiệu quả | Tổng tokens cho mỗi vấn đề đã giải quyết so với một agent |
| 15 | Kỹ thuật điều phối | Giải quyết Merge xung đột, biểu đồ bàn giao-thất bại |
| **100** |||

## Bài tập

1. Chèn một lỗi rõ ràng vào một diff giữa chừng (thêm `return None` trước phần thân chính). Đo lường tỷ lệ phê duyệt sai của người đánh giá. Điều chỉnh prompt của người đánh giá cho đến khi phê duyệt sai dưới 5%.

2. Giảm xuống còn hai lập trình viên (kiến trúc sư + lập trình viên + người đánh giá + người kiểm tra, lập trình viên chạy hai tác vụ con tuần tự). So sánh đồng hồ treo tường và tỷ lệ vượt qua.

3. Thay thế bộ điều phối merge bằng ràng buộc một người ghi (nhiệm vụ con chạm vào các tập hợp tệp rời rạc). Đo lường gánh nặng quy hoạch cho kiến trúc sư.

4. Hoán đổi người đánh giá từ GPT-5.4 sang Claude Opus 4.7. Đo lường tỷ lệ phê duyệt sai và token chênh lệch chi phí.

5. Thêm vai trò thứ năm: tài liệu (Haiku 4.5). Sau khi xem xét, nó sẽ tạo ra một mục nhật ký thay đổi. Đo lường xem chất lượng tài liệu có phù hợp với việc chi tiêu thêm token hay không.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Cây làm việc song song | "Cô lập branch" | `git worktree add` tạo ra một cây làm việc mới cho mỗi mã hóa |
| Bảng tác vụ | "Bus tin nhắn được chia sẻ" | Tệp hoặc Redis lưu trữ các tin nhắn đã nhập agents đăng ký |
| Bàn giao | "Ranh giới vai trò" | Bất kỳ tin nhắn nào chuyển từ ngữ cảnh của vai trò này sang ngữ cảnh của vai trò khác |
| Token khuếch đại | "Chi phí trên nhiều agent" | Tổng tokens trên các vai trò / một agent tokens cho cùng một nhiệm vụ |
| Giao thức A2A | "Agent agent" | Thông số kỹ thuật năm 2025 của Google cho tin nhắn liên agent được nhập |
| Điều phối viên Merge | "Nhà tích hợp" | Thành phần chạy merge ba chiều và hòa giải xung đột |
| Phê duyệt sai | "Ảo giác của người đánh giá" | Người đánh giá phê duyệt sự khác biệt với các lỗi đã biết |

## Đọc thêm

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) - Nhà máy đa agent tham chiếu 2026
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — đa agent framework dựa trên vai trò
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Diễn viên đánh máy của Microsoft framework
- [Cognition AI (Devin)](https://cognition.ai) - sản phẩm tham khảo
- [Factory Droids](https://www.factory.ai) - sản phẩm tham chiếu thay thế
- [Google A2A protocol](https://developers.google.com/agent-to-agent) — thông số kỹ thuật nhắn tin giữa các agent
- [git worktree documentation](https://git-scm.com/docs/git-worktree) - chất nền cách ly
- [SWE-bench Pro](https://www.swebench.com) — mục tiêu đánh giá
