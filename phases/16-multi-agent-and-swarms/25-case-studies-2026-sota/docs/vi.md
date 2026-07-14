# Nghiên cứu điển hình và tình trạng nghệ thuật năm 2026

> Ba tài liệu tham khảo cấp production để nghiên cứu từ đầu đến cuối, mỗi tài liệu minh họa một phần khác nhau của kỹ thuật đa agent. **Hệ thống Nghiên cứu của Anthropic** (orchestrator-worker, 15x tokens, +90,2% so với một agent Opus 4, triển khai cầu vồng) là trường hợp giám sát chính tắc. **MetaGPT / ChatDev** (chuyên môn hóa vai trò được mã hóa SOP cho kỹ thuật phần mềm; "Ảo giác giao tiếp" của ChatDev; Phần mở rộng MacNet đến >1000 agents thông qua DAG, arXiv:2406.07155) là trường hợp phân tách vai trò chính tắc. **OpenClaw / Moltbook** (ban đầu là Clawdbot của Peter Steinberger, tháng 11 năm 2025; đổi tên hai lần; 247 nghìn GitHub sao vào tháng 3 năm 2026; agents vòng lặp ReAct; Moltbook là một mạng xã hội chỉ dành cho agent với ~2,3 triệu tài khoản agent trong vòng vài ngày sau khi ra mắt, được Meta mua lại 2026-03-10) minh họa những gì xảy ra ở quy mô dân số: hoạt động kinh tế mới nổi, rủi ro tiêm prompt, quy định cấp tiểu bang (Trung Quốc hạn chế OpenClaw trên máy tính của chính phủ, tháng 3 năm 2026). **Framework bối cảnh tháng 4 năm 2026: **LangGraph và CrewAI dẫn đầu production; AG2 là sự tiếp nối của cộng đồng AutoGen; Microsoft AutoGen đang ở chế độ bảo trì (merged vào Microsoft Agent Framework, RC Feb 2026); OpenAI Agents SDK là người kế nhiệm production Swarm; Google ADK (tháng 4 năm 2025) là người tham gia A2A bản địa. Mọi framework chính hiện ships MCP hỗ trợ; ship A2A nhất. Bài học này đọc từng trường hợp từ đầu đến cuối và chắt lọc các mẫu phổ biến để bạn có thể chọn tài liệu tham khảo phù hợp cho hệ thống production tiếp theo của mình.

**Loại:** Tìm hiểu (capstone)
**Ngôn ngữ:** —
**Kiến thức tiên quyết:** tất cả Giai đoạn 16 (Bài 01-24)
**Thời lượng:** ~90 phút

## Vấn đề

Kỹ thuật đa agent là một ngành học trẻ. Các tài liệu tham khảo production rất ít và mỗi tài liệu tham khảo bao gồm một phần khác nhau của không gian. Đọc từng cái một rất hữu ích; So sánh chúng như một tập hợp sẽ hữu ích hơn. Bài học này coi ba nghiên cứu điển hình kinh điển năm 2026 như một danh sách đọc từ đầu đến cuối, ghim các mẫu phổ biến và lập bản đồ bối cảnh framework để bạn có thể đưa ra lựa chọn framework từ kiến thức chứ không phải tiếp thị.

## Khái niệm

### Anthropic Hệ thống nghiên cứu

Trường hợp worker giám sát production. Claude Opus 4 lập kế hoạch và tổng hợp; Claude Sonnet 4 subagents nghiên cứu song song. Bài đăng kỹ thuật đã xuất bản: https://www.anthropic.com/engineering/multi-agent-research-system.

Kết quả đo lường chính:

- **+90,2%** cải thiện so với Opus 4 một agent về đánh giá nghiên cứu nội bộ.
- **80% BrowseComp variance** được giải thích bởi **token cách sử dụng** - nhiều agent chiến thắng phần lớn vì mỗi subagent đều có một context window mới.
- **15x tokens cho mỗi truy vấn** so với một agent.
- **Triển khai cầu vồng** vì agents chạy lâu dài và có trạng thái.

Các bài học thiết kế được hệ thống hóa:

1. **Mở rộng quy mô nỗ lực để truy vấn độ phức tạp.** Đơn giản → 1 agent với 3-10 lệnh gọi công cụ. Trung bình → 3 agents. Nghiên cứu phức tạp → 10+ subagents.
2. **Mở rộng trước, sau đó thu hẹp.** Subagents thực hiện tìm kiếm rộng; chì tổng hợp; theo dõi subagents thực hiện độ sâu có mục tiêu.
3. **Cầu vồng triển khai.** Giữ cho các phiên bản runtime cũ tồn tại cho đến khi agents trên chuyến bay của chúng kết thúc.
4. **Xác minh không phải là tùy chọn.** Hệ thống được quan sát thấy bị ảo giác mà không có vai trò xác minh rõ ràng.

Đây là trường hợp tham chiếu cho cấu trúc liên kết worker giám sát (Giai đoạn 16 · 05) ở tỷ lệ production.

### MetaGPT / ChatDev

Trường hợp production SOP-role-debreak. Cover arXiv:2308.00352 (MetaGPT) và arXiv:2307.07924 (ChatDev).

MetaGPT mã hóa SOP kỹ thuật phần mềm dưới dạng prompts vai trò: Giám đốc sản phẩm, Kiến trúc sư, Quản lý dự án, Kỹ sư, Kỹ sư QA. Khung của tờ báo: `Code = SOP(Team)`. Mỗi vai trò có prompt hẹp, chuyên biệt; chuyển giao giữa các vai trò mang artifacts có cấu trúc (tài liệu PRD, tài liệu kiến trúc, mã).

Đóng góp của ChatDev: **ảo giác giao tiếp**. Agents yêu cầu chi tiết cụ thể trước khi trả lời - một nhà thiết kế agent hỏi lập trình viên ngôn ngữ nào trước khi phác thảo giao diện người dùng, thay vì đoán. Bài báo báo cáo điều này làm giảm ảo giác trong nhiều agent pipelines một cách có thể đo lường được.

MacNet (arXiv:2406.07155) mở rộng ChatDev lên **>1000 agents thông qua DAGs**. Mỗi nút DAG là một chuyên môn hóa vai trò; Các cạnh mã hóa hợp đồng bàn giao. Tỷ lệ này có thể thực hiện được vì định tuyến rõ ràng và có thể tính toán ngoại tuyến.

Bài học thiết kế:

1. **Cấu trúc quan trọng hơn quy mô.** Một nhóm SOP 5 vai trò chặt chẽ đánh bại một nhóm không có cấu trúc 50 agent.
2. **Hợp đồng bàn giao bằng văn bản.** Artifacts chuyển giao giữa các vai trò theo một schema.
3. **Ảo giác giao tiếp **là một mô hình chịu tải rẻ.
4. **DAG mở rộng quy mô hơn trò chuyện.** Khi có thể biết được quy trình đó, hãy mã hóa quy trình đó.

Đây là trường hợp tham khảo cho chuyên môn hóa vai trò (Giai đoạn 16 · 08) và cấu trúc liên kết (Giai đoạn 16 · 15).

### Hệ sinh thái OpenClaw / Moltbook

Trường hợp quy mô dân số production. Dòng thời gian:

- **Tháng 11 năm 2025: **Clawdbot (agent mã hóa vòng lặp ReAct cục bộ của Peter Steinberger) ships.
- **Tháng 12 năm 2025 - Tháng 3 năm 2026: **đổi tên hai lần (Clawdbot → OpenClaw → tiếp tục dưới OpenClaw).
- **Tháng 2 năm 2026:** Moltbook ra mắt dưới dạng mạng xã hội chỉ dành cho agent trên cùng một primitives; ~2,3 triệu tài khoản agent trong vòng vài ngày.
- **Mar 2026 (2026-03-10):** Meta mua lại Moltbook.
- **Tháng 3 năm 2026:** Trung Quốc hạn chế OpenClaw trên máy tính của chính phủ.
- **Tháng 3 năm 2026: **OpenClaw vượt qua 247 nghìn GitHub sao.

Đây là những gì đa agent trông như thế nào khi bạn đặt hàng triệu agents trên một chất nền được chia sẻ:

- **Hoạt động kinh tế mới nổi.** Agents mua, bán và phục vụ lẫn nhau bằng cách sử dụng thanh toán token.
- **Rủi ro tiêm Prompt ở quy mô dân số.** Một prompt độc hại trong hồ sơ agent của virus lan truyền đến hàng nghìn tương tác agent-agent trong vài giờ.
- **Phản hồi quy định cấp tiểu bang.** Trong vòng vài tuần kể từ khi ra mắt, quy định sẽ đến với hệ sinh thái.

Bài học thiết kế từ trường hợp này một phần là kỹ thuật, một phần là quản trị:

1. **Đa agent ở quy mô dân số là một chế độ mới.** Các phương pháp hay nhất về hệ thống cá nhân (xác minh, làm rõ vai trò) vẫn được áp dụng nhưng chưa đủ.
2. **Prompt chèn là XSS mới.** Coi cấu hình agent và thông báo chéo agent là đầu vào không đáng tin cậy theo mặc định.
3. **Quy định nhanh hơn chu kỳ thiết kế.** Lập kế hoạch cho nó.
4. **Mã nguồn mở + hợp chất quy mô virus.** 247 nghìn ngôi sao trong ~4 tháng là bất thường; thiết kế để triển khai-burst-load.

Xem báo cáo của [OpenClaw Wikipedia](https://en.wikipedia.org/wiki/OpenClaw) và CNBC / Palo Alto Networks để biết chi tiết về hệ sinh thái. Đối với các nền tảng kỹ thuật, Clawdbot / OpenClaw repos hiển thị vòng lặp ReAct cục bộ; Các bài đăng công khai của Moltbook tiết lộ kiến trúc đồ thị xã hội ở trên cùng.

### Framework cảnh quan Tháng Tư 2026

| Framework | Trạng thái | Tốt nhất cho | Ghi chú |
|---|---|---|---|
| **LangGraph** (LangChain) | Production lãnh đạo | Đồ thị có cấu trúc + điểm kiểm tra + con người trong vòng lặp | Mặc định được đề xuất cho production |
| **Phi hành đoànAI **| Production lãnh đạo | Đội ngũ dựa trên vai trò với Sequential/Hierarchical processes | mạnh mẽ cho sự phân hủy vai trò |
| **AG2** | Cộng đồng được duy trì | GroupChat + lựa chọn loa | Tiếp tục AutoGen v0.2 |
| **Tự động của Microsoft** | Chế độ bảo trì (Tháng 2 năm 2026) | — | merged vào Microsoft Agent Framework RC |
| **Microsoft Agent Framework** | RC (Tháng Hai 2026) | orchestration mẫu + tích hợp doanh nghiệp | người mới nhập cảnh; Xem |
| **OpenAI Agents SDK** | Production | Swarm kế nhiệm | Mẫu chuyển giao trả lại công cụ |
| **Google ADK** | Production (Tháng Tư 2025) | A2A bản địa | Tích hợp Google Cloud |
| **Anthropic Claude Agent SDK** | Production | Một agent + Mở rộng nghiên cứu | xem bài đăng trên hệ thống nghiên cứu |

Mọi framework lớn hiện ships hỗ trợ **MCP**; hầu hết ship **A2A**. Khả năng tương thích giao thức không còn là một điểm khác biệt.

### Các mô hình phổ biến trên cả ba trường hợp

1. **Orchestrator + workers** (Anthropic giám sát rõ ràng, MetaGPT PM-as-supervisor, OpenClaw individual agents + hiệu ứng mạng).
2. **Hợp đồng bàn giao có cấu trúc** (mô tả nhiệm vụ Anthropic subagent, tài liệu MetaGPT PRD/architecture, OpenClaw A2A artifacts).
3. **Xác minh với tư cách là vai trò class thứ nhất** (người xác minh của Anthropic, Kỹ sư QA của MetaGPT, trình xác thực trong mạng của OpenClaw).
4. **Tỷ lệ là cấu trúc liên kết + chất nền, không chỉ agents hơn **(triển khai cầu vồng, MacNet DAG, chất nền quy mô dân số).
5. **Chi phí là trọng yếu và được tiết lộ** (15x tokens, ngân sách cho mỗi vai trò trong MetaGPT, định giá cho mỗi tương tác trong Moltbook).
6. **Tình hình bảo mật rõ ràng** (sandbox của Anthropic, hạn chế vai trò của MetaGPT, prompt-injection của OpenClaw làm bề mặt tấn công đã biết).

### Chọn tài liệu tham khảo cho dự án tiếp theo của bạn

- **Production nhiệm vụ nghiên cứu / kiến thức → Anthropic Nghiên cứu.** Bối cảnh mới subagents chiến thắng.
- **Quy trình làm việc kỹ thuật / chuỗi công cụ → MetaGPT / ChatDev.** Vai trò + SOP + hợp đồng bàn giao.
- **Sản phẩm xã hội hiệu ứng mạng → OpenClaw / Moltbook.** Chất nền + nền kinh tế mới nổi.
- **Tự động hóa doanh nghiệp cổ điển → CrewAI hoặc LangGraph** (production dẫn đầu, runtime ổn định).

### Tóm tắt hiện đại năm 2026

Trường ở đâu vào tháng 4 năm 2026:

- **Frameworks đang hội tụ.** Hỗ trợ MCP + A2A là tiền đặt cược trên bàn. Ngữ nghĩa bàn giao là lựa chọn thiết kế còn lại.
- **Đánh giá đang cứng hơn.** SWE-bench Pro, MARBLE, STRATUS giảm thiểu benchmarks. Pro là kiểm tra thực tế chống ô nhiễm hiện tại.
- **Tỷ lệ thất bại Production có thể đo lường được **(Cemri 2025 MAST; 41-86.7% trên MAS thực). Lĩnh vực này đã ra khỏi kỷ nguyên "trông tuyệt vời trong bản demo".
- **Chi phí là ràng buộc kỹ thuật trung tâm.** Token chi phí cho mỗi tác vụ, đồng hồ treo tường cho mỗi tương tác, chi phí triển khai cầu vồng. Nhiều agent thắng accuracy nhưng thua về chi phí - và thương mại đó là quyết định kinh doanh.
- **Quy định là đầu vào ngắn hạn, không phải là mối quan tâm nền.** Các khu vực pháp lý đang di chuyển nhanh hơn các chu kỳ triển khai riêng lẻ.

## Ứng dụng

`outputs/skill-case-study-mapper.md` là một skill đọc thiết kế hệ thống đa agent được đề xuất và ánh xạ nó với nghiên cứu điển hình gần nhất, hiển thị các quyết định thiết kế mà nghiên cứu điển hình đã thử nghiệm.

## Sản phẩm bàn giao

Quy tắc dành cho người mới bắt đầu cho production nhiều agent vào năm 2026:

- **Bắt đầu từ một nghiên cứu điển hình, không phải từ đầu.** Chọn nghiên cứu gần nhất của Anthropic Nghiên cứu / MetaGPT / OpenClaw và thích ứng.
- **Áp dụng MCP + A2A.** Tính di động trên frameworks là có giá trị; Hỗ trợ giao thức là miễn phí.
- **Đo lường so với SWE-bench Pro hoặc Pro tương đương bên trong của bạn.** Đã xác minh bị nhiễm bẩn.
- **Trả thuế xác minh.** Người xác minh độc lập tốn ~20-30% ngân sách token của bạn và mua tính chính xác có thể đo lường được.
- **Rainbow triển khai agents chạy trong thời gian dài.** Dự kiến các agent chạy nhiều giờ sẽ là thường xuyên.
- **Đọc WMAC 2026 và các phần tiếp theo của MAST.** Kỷ luật đang phát triển nhanh chóng.

## Bài tập

1. Đọc bài đăng từ đầu đến cuối hệ thống Nghiên cứu Anthropic. Xác định ba quyết định thiết kế sẽ thay đổi nếu bạn thay thế Opus 4 bằng một model nhỏ hơn (ví dụ: Haiku 4).
2. Đọc MetaGPT Phần 3-4 (arXiv:2308.00352). Mã hóa một SOP từ miền của riêng bạn (không phải phần mềm) dưới dạng vai trò prompts. SOP ngụ ý bao nhiêu vai trò?
3. Đọc ChatDev (arXiv: 2307.07924). Xác định cơ chế của "ảo giác giao tiếp". Triển khai nó trong một trong các hệ thống đa agent hiện có của bạn.
4. Đọc về OpenClaw và Moltbook. Chọn một chế độ lỗi cụ thể xuất hiện ở quy mô dân số sẽ không xuất hiện trong hệ thống 5 agent. Bạn sẽ thiết kế chống lại nó như thế nào?
5. Chọn dự án nhiều agent hiện tại của bạn. Nghiên cứu điển hình nào trong ba trường hợp là tài liệu tham khảo gần nhất? Những quyết định thiết kế nào từ nghiên cứu điển hình đó mà bạn CHƯA áp dụng? Viết ra một trong những bạn sẽ áp dụng trong quý này.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Nghiên cứu Anthropic | "Tài liệu tham khảo của người giám sát" | Claude Opus 4 + Sonnet 4 subagents; 15x tokens; +90,2% so với một agent. |
| Siêu GPT | "SOP như prompts" | Phân tích vai trò cho kỹ thuật phần mềm; `Code = SOP(Team)`. |
| ChatDev | "Agents như vai trò" | Nhà thiết kế / lập trình viên / người đánh giá / người thử nghiệm; Ảo giác giao tiếp. |
| Máy MacNet | "Mở rộng quy mô ChatDev qua DAG" | arXiv: 2406.07155; 1000+ agents thông qua định tuyến DAG rõ ràng. |
| Móng vuốt mở | "Vòng lặp ReAct cục bộ agents" | Dự án của Steinberger; 247 nghìn sao vào tháng 3 năm 2026. |
| Sổ lột xác | "Mạng xã hội chỉ dành cho Agent" | 2,3 triệu tài khoản agent; được Meta mua lại vào tháng 3 năm 2026. |
| Triển khai cầu vồng | "Nhiều phiên bản đồng thời" | Giữ cho các phiên bản runtime cũ tồn tại cho các agents chạy dài trên chuyến bay. |
| Ảo giác giao tiếp | "Hỏi trước khi trả lời" | Agents yêu cầu chi tiết cụ thể từ các đồng nghiệp thay vì đoán. |
| WMAC 2026 | "Hội thảo AAAI" | Tháng Tư 2026 đầu mối cộng đồng về điều phối đa agent. |

## Đọc thêm

- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — tài liệu tham khảo worker production giám sát
- [MetaGPT — Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352) — Phân tách vai trò SOP
- [ChatDev — Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924) - ảo giác giao tiếp
- [MacNet — scaling role-based agents to 1000+](https://arxiv.org/abs/2406.07155) — Thang đo dựa trên DAG
- [OpenClaw on Wikipedia](https://en.wikipedia.org/wiki/OpenClaw) — Tổng quan về hệ sinh thái
- [WMAC 2026](https://multiagents.org/2026/) - Hội thảo Chương trình Cầu nối AAAI 2026 về Điều phối đa Agent
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — production lãnh đạo
- [CrewAI docs](https://docs.crewai.com/en/introduction) — framework dựa trên vai trò
