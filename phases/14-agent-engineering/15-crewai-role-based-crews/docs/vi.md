# CrewAI: Phi hành đoàn và luồng dựa trên vai trò

> CrewAI là đa agent framework dựa trên vai trò năm 2026. Bốn primitives: Agent, Nhiệm vụ, Phi hành đoàn, Process. Hai hình dạng cấp cao nhất: Nhóm (cộng tác tự động, dựa trên vai trò) và Dòng (theo sự kiện, xác định). Các tài liệu rất thẳng thừng: "đối với bất kỳ ứng dụng sẵn sàng production nào, hãy bắt đầu với Flow."

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 12 (Mẫu quy trình làm việc), Giai đoạn 14 · 14 (Diễn viên Model)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Kể tên bốn primitives của CrewAI (Agent, Nhiệm vụ, Phi hành đoàn, Process) và những gì mỗi người sở hữu.
- Phân biệt tuần tự, phân cấp và process đồng thuận đã lên kế hoạch; Chọn một cho mỗi khối lượng công việc.
- Phân biệt Nhóm (dựa trên vai trò tự trị) với Quy trình (xác định theo hướng sự kiện) và giải thích đề xuất production của tài liệu.
- Dây các công cụ với bộ trang trí `@tool` và lớp con `BaseTool`; lý do về đầu ra có cấu trúc so với văn bản tự do.
- Đặt tên cho bốn loại bộ nhớ CrewAI và thời điểm mỗi loại được đền đáp.
- Triển khai một nhóm ba agent (nhà nghiên cứu, nhà văn, biên tập viên) để tạo ra một bản tóm tắt.
- Phát hiện ba chế độ thất bại của CrewAI: prompt-cồng kềnh, thuế LLM người quản lý, bàn giao giòn.

## Vấn đề

Các đội áp dụng nhiều agent frameworks gặp phải cùng một bức tường. "Cộng tác tự động" nghe có vẻ tuyệt vời trong một bản demo. Sau đó, khách hàng gửi lỗi và bạn cần phát lại xác định. Hoặc tài chính hỏi chi phí cho một phi hành đoàn LLM tuyến bao nhiêu cho mỗi lần chạy. Hoặc cuộc gọi cần biết agent nào bị đình trệ lúc 3 giờ sáng.

Các phi hành đoàn định tuyến LLM dạng tự do không trả lời rõ ràng. DAG thuần túy trả lời tất cả nhưng mất đi hình dạng khám phá mà agent động não cần.

Sự chia tách của CrewAI là trung thực về giao dịch. Phi hành đoàn cho công việc hợp tác, dựa trên vai trò, khám phá. Quy trình dành cho production theo hướng sự kiện, sở hữu mã, có thể kiểm tra. Cùng một framework, hai hình dạng, chọn trên mỗi bề mặt.

## Khái niệm

### Bốn primitives

Bề mặt của CrewAI nhỏ. Ghi nhớ điều này và rest là config.

- **Agent.** `role + goal + backstory + tools + (optional) llm`. Cốt truyện là chịu tải. Nó định hình giọng điệu, phán đoán, khi agent dừng lại. Công cụ là các chức năng mà agent có thể gọi (thêm bên dưới).
- **Nhiệm vụ.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`. Một đơn vị công việc có thể tái sử dụng. `expected_output` là hợp đồng. `context` liệt kê các tác vụ ngược dòng có đầu ra được truyền vào. `output_pydantic` buộc một hình dạng có cấu trúc.
- **Phi hành đoàn.** Container. Sở hữu danh sách `agents`, danh sách `tasks`, `process` và cài đặt `memory` + `verbose` + `manager_llm` tùy chọn.
- **Process.** Chiến lược thực hiện. Tuần tự, Phân cấp, Đồng thuận (theo kế hoạch). Chọn hình dạng của đường chạy.

Agents không nhìn thấy nhau trực tiếp. Nhiệm vụ tham khảo agents. Phi hành đoàn sắp xếp các nhiệm vụ. Người Process quyết định ai chọn nhiệm vụ tiếp theo. Đó là toàn bộ model tinh thần.

> **Đã được xác thực chống lại **CrewAI 0.86 (2026-05). Các phiên bản mới hơn có thể đổi tên hoặc merge process loại; Kiểm tra [CrewAI Processes docs](https://docs.crewai.com/concepts/processes) trước khi dựa vào một hình dạng cụ thể.

### Tuần tự vs Phân cấp vs Đồng thuận

- **Tuần tự.** Nhiệm vụ chạy theo thứ tự khai báo. Đầu ra của nhiệm vụ N có sẵn dưới dạng `context` cho nhiệm vụ N + 1. Chi phí thấp nhất. Dễ đoán nhất. Sử dụng khi đơn đặt hàng được cố định.
- **Phân cấp.** Người quản lý Agent (cuộc gọi LLM riêng biệt) giữa các chuyên gia. CrewAI sinh ra trình quản lý từ `manager_llm` config của bạn hoặc mặc định. Người quản lý chọn nhiệm vụ tiếp theo mỗi vòng và có thể từ chối hoặc định tuyến lại. Sử dụng khi bạn có bốn bác sĩ chuyên khoa trở lên và đặt hàng thực sự phụ thuộc vào đầu ra prior.
- **Đồng thuận.** Đã lên kế hoạch, hiện chưa được thực hiện trong API công cộng. Các tài liệu dành tên cho một process dựa trên bỏ phiếu trong tương lai. Đừng dựa vào nó ngày hôm nay.

Phân cấp thêm một cuộc gọi LLM mỗi vòng (người quản lý) trên mỗi cuộc gọi chuyên gia. Chi phí Token có thể tăng gấp ba lần trong một lần chạy năm bước. Chỉ thanh toán khi bạn cần định tuyến.

### Phi hành đoàn vs Dòng chảy

Đây là khung mà các tài liệu dẫn đầu vào năm 2026.

- **Phi hành đoàn.** Quyền tự chủ LLM. framework chọn hình dạng ở runtime. Tốt cho: nghiên cứu, động não, bản nháp đầu tiên, bất cứ nơi nào con đường là một phần của câu trả lời. Khó phát lại. Khó kiểm tra. Giá rẻ để tạo nguyên mẫu.
- **Flow.** Biểu đồ theo sự kiện mà bạn sở hữu. `@start` đánh dấu mục nhập. `@listen(topic)` đánh dấu một bước kích hoạt khi một bước khác phát ra chủ đề đó. Mỗi bước đều Python đơn giản (có thể gọi Kíp lái nội bộ). Phù hợp với: production. Có thể quan sát được. Có thể kiểm tra. Xác định.

Khuyến nghị production năm 2026 của tài liệu: bắt đầu với Flow. Gấp Phi hành đoàn khi `Crew.kickoff()` cuộc gọi từ bên trong Flow bước khi quyền tự chủ kiếm được chi phí. Flow cung cấp cho bạn dấu vết kiểm tra, Phi hành đoàn cung cấp cho bạn khám phá. Soạn thảo, không chọn.

### Tích hợp công cụ

Ba cách để cung cấp cho Agent một công cụ. Chọn một trong những đơn giản nhất phù hợp.

1. **`@tool` trang trí.** Các chức năng thuần túy trở thành công cụ. Chữ ký là schema; docstring là mô tả mà LLM nhìn thấy. Tốt nhất cho những người trợ giúp một lần.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` lớp con.** Công cụ dựa trên Class với schema đối số rõ ràng, hỗ trợ không đồng bộ, thử lại. Sử dụng khi công cụ có trạng thái (máy khách, bộ nhớ đệm) hoặc cần các đối số có cấu trúc.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Bộ công cụ tích hợp.** CrewAI ships bộ điều hợp của bên thứ nhất: `SerperDevTool`, `FileReadTool`, `DirectoryReadTool`, `CodeInterpreterTool`, `RagTool`, `WebsiteSearchTool`. Có dây với một import.

Đầu ra có cấu trúc sử dụng Pydantic. Chuyển `output_pydantic=MyModel` vào nhiệm vụ. CrewAI xác nhận phản ứng LLM chống lại model và ép buộc hoặc thử lại. Ghép nối điều này với một chuỗi `expected_output` chặt chẽ. Đầu ra văn bản tự do phù hợp với bản nháp; đầu ra có cấu trúc là những gì Luồng xuôi dòng có thể tiêu thụ.

### Bộ nhớ hooks

CrewAI ships bốn loại bộ nhớ ra khỏi hộp. Họ soạn thảo: một Phi hành đoàn có thể kích hoạt cả bốn cùng một lúc.

> **Đã được xác thực chống lại **CrewAI 0.86 (2026-05). Các bản phát hành gần đây định tuyến mọi thứ thông qua một hệ thống `Memory` thống nhất bao bọc bốn cửa hàng này. model khái niệm dưới đây vẫn giữ nguyên, nhưng bề mặt class công khai có thể sụp đổ thành một điểm vào `Memory` duy nhất trong các phiên bản mới hơn; Kiểm tra [CrewAI memory docs](https://docs.crewai.com/concepts/memory) để biết API hiện tại.

- **Ngắn hạn.** Bộ đệm hội thoại trong một lần chạy. Xóa ở cuối.
- **Dài hạn.** Kiên trì qua các lần chạy. Được lưu trữ trong vector DB (Chroma theo mặc định, có thể hoán đổi). Được truy xuất bởi sự tương tự với nhiệm vụ hiện tại.
- **Thực thể.** Sự kiện cho mỗi thực thể. "Khách hàng X đang sử dụng gói doanh nghiệp." Khóa theo thực thể, không phải bởi sự tương đồng. Sống sót qua các lần chạy.
- **Theo ngữ cảnh.** Truy xuất thời gian lắp ráp. Kéo bộ nhớ liên quan tại thời điểm Agent cần, không được tải trước.

Bật trên Phi hành đoàn với config `memory=True` hoặc từng loại. Được hỗ trợ bởi nhà cung cấp embeddings mà bạn định cấu hình (mặc định là OpenAI, có thể hoán đổi thành cục bộ). Bộ nhớ là một trong những nơi mà CrewAI kiếm được để chống lại frameworks mỏng hơn; LangGraph thuần túy yêu cầu bạn tự kết nối từng loại này.

### Khi CrewAI phù hợp

- Ba đến sáu agents với các vai trò được đặt tên và quy trình làm việc cộng tác. Soạn thảo, rà soát, lập kế hoạch, động não.
- Định tuyến mà phán đoán của LLM về bước tiếp theo là một phần của giá trị (Phân cấp).
- Bất cứ nơi nào nhóm hạnh phúc hơn khi đọc `role + goal + backstory` hơn là đọc định nghĩa đồ thị.

### Khi CrewAI không phù hợp

- DAG xác định với thứ tự nghiêm ngặt. Sử dụng LangGraph (Bài 13). Hình dạng đồ thị là trừu tượng phù hợp; Đóng khung vai trò của CrewAI là ma sát.
- Ngân sách độ trễ dưới giây. Phân cấp thêm các chuyến đi khứ hồi. Ngay cả Sequential cũng prompts nối tiếp bao gồm cốt truyện và đầu ra prior.
- Vòng lặp một agent. Bỏ qua framework; một vòng lặp agent (Bài 1) cộng với một registry công cụ ngắn hơn.

Bài 17 (Agent Framework Đánh đổi) trình bày điều này trong một ma trận. Phiên bản ngắn: CrewAI nằm ở góc "dựa trên vai trò hợp tác".

### Hình dạng phần phụ thuộc

Độc lập với LangChain. Python 3.10 đến 3.13. Sử dụng `uv`. Số sao: xem [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) (ảnh chụp nhanh tính đến năm 2026-05). Tích hợp AWS Bedrock được ghi lại; nhà cung cấp benchmarks báo cáo tốc độ đáng kể so với LangGraph về khối lượng công việc QA, nhưng phương pháp luận (dataset, phần cứng, chỉ số đánh giá) không được công bố, vì vậy chỉ coi số lượng nhà cung cấp framework là định hướng.

### Mô hình này sai ở đâu

- **Prompt phình to từ những câu chuyện nền.** Một cốt truyện dài 2000 từ mỗi agent và một nhóm năm agent đốt cháy ngân sách ngữ cảnh trước khi gọi công cụ đầu tiên. Giữ cốt truyện dưới 200 từ. Sử dụng lại các cụm từ trên agents; Không lặp lại phong cách nhà năm lần.
- **Thuế LLM token người quản lý.** process phân cấp thêm một người quản lý LLM gọi trước mỗi cuộc gọi của chuyên gia. Trên một nhóm năm nhiệm vụ, đó là sáu cuộc gọi LLM thay vì năm cuộc gọi và cuộc gọi của người quản lý mang danh sách nhiệm vụ đầy đủ cộng với prior đầu ra. Chuyển sang tuần tự trừ khi định tuyến phụ thuộc vào đầu ra.
- **Chuyển giao giòn.** `expected_output` của Nhiệm vụ N là "một dàn ý". Nhiệm vụ N+1 đọc nó là `context` và cố gắng phân tích cú pháp ba phần. LLM sản xuất bốn. Hạ lưu Agent ad-libs. Sửa lỗi với `output_pydantic` trên Tác vụ N để Tác vụ N + 1 đọc một đối tượng được nhập, không phải văn bản tự do.
- **Crew-as-prod.** Phi hành đoàn dạng tự do shipped production mà không cần bao bọc Flow. Sự thay đổi đầu ra cao; phát lại là không thể; On-Call không thể phân biệt một cuộc chạy tồi tệ với một cuộc chạy tốt. Bọc bằng một dòng chảy.

## Tự xây dựng

`code/main.py` triển khai các phiên bản stdlib của cả hai hình dạng cộng với một phi hành đoàn ba agent.

Hình dạng:

- `Agent`, `Task` lớp dữ liệu phù hợp với bề mặt của CrewAI.
- `SequentialCrew.kickoff(inputs)` chạy các tác vụ theo thứ tự khai báo, phân luồng đầu ra dưới dạng `context`.
- `HierarchicalCrew.kickoff(topic)` thêm một người quản lý Agent chọn chuyên gia tiếp theo mỗi vòng, dừng lại ở "xong".
- `Flow` với `@start` và `@listen(topic)` trang trí, một vòng lặp sự kiện nhỏ và một trace.
- `tool(name)` trang trí phản chiếu hình dạng `@tool` của CrewAI.
- `Memory` với các cửa hàng `short_term`, `long_term` `entity`; Mô phỏng tương tự sử dụng numpy.
- Phản hồi LLM giả là các chuỗi được mã hóa cứng được khóa khỏi vai trò cộng với tiền tố đầu vào. Không có mạng. Xác định.

Bản demo cụ thể: nhà nghiên cứu, nhà văn, biên tập viên nhóm sản xuất một bản tóm tắt về "agent kỹ thuật 2026". Nhà nghiên cứu kéo (chế giễu) nguồn. Bản nháp của nhà văn. Trình chỉnh sửa thắt chặt. Cùng một phi hành đoàn chạy qua một Flow để hiển thị hình dạng xác định.

Chạy nó:

```bash
python3 code/main.py
```

Trace bao gồm: đầu ra luồng tuần tự của nhóm thông qua `context`, nhóm phân cấp với các lựa chọn của người quản lý (nhà nghiên cứu, nhà văn, biên tập viên, sau đó là "xong"), dòng chạy ba bước giống nhau với các chủ đề rõ ràng (`researched`, `drafted`, `edited`), các lệnh gọi công cụ được định tuyến qua `@tool` và bộ nhớ dài hạn tồn tại qua hai lần khởi động.

Phi hành đoàn trace rất linh hoạt; Người quản lý về nguyên tắc có thể sắp xếp lại. Flow trace đã được cố định. Sự lựa chọn đó là bài học.

## Ứng dụng

- **CrewAI Flow** cho production. Ngay cả khi Flow là một bước gọi `Crew.kickoff()`. Quy trình đưa ra ranh giới kiểm toán.
- **CrewAI Crew (Tuần tự)** dành cho công việc hợp tác theo thứ tự rõ ràng, đặc biệt là bản nháp đầu tiên và vòng lặp đánh giá.
- **CrewAI Crew (Phân cấp)** khi định tuyến phụ thuộc vào đầu ra và bạn có bốn chuyên gia trở lên.
- **LangGraph** (Bài 13) dành cho các máy trạng thái rõ ràng, sơ yếu lý lịch bền, thứ tự nghiêm ngặt.
- **AutoGen v0.4** (Bài 14) cho tính đồng thời model diễn viên và cách ly lỗi.
- **OpenAI Agents SDK** (Bài 16) dành cho các sản phẩm ưu tiên OpenAI có bàn giao và guardrails.
- **Claude Agent SDK** (Bài 17) đối với các sản phẩm ưu tiên Claude với cửa hàng subagents và session.

## Sản phẩm bàn giao

`outputs/skill-crew-or-flow.md` chọn Crew vs Flow cho một nhiệm vụ và thực hiện tối thiểu. Từ chối cứng rắn đối với Phi hành đoàn không có cốt truyện, Dòng chảy không có chủ đề rõ ràng, Phân cấp với dưới ba chuyên gia.

## Cạm bẫy

- **Cốt truyện như hương vị.** Nó định hình đầu ra. Thử nghiệm ba biến thể mỗi agent; variance là có thật. Chọn một cái, đông lạnh nó.
- **Bỏ qua `expected_output`.** Nếu không có hợp đồng cho mỗi nhiệm vụ, các nhiệm vụ xuôi dòng sẽ chọn bất kỳ LLM nào được tạo ra. Phi hành đoàn chạy; kiểm tra không thành công.
- **Bộ nhớ luôn bật.** Ghi dài hạn mỗi lần chạy. Vector DB phát triển. Truy xuất trở nên ồn ào. Phạm vi ghi vào các nhiệm vụ mà thực tế là dai dẳng.
- **Người quản lý prompt trôi dạt.** Người quản lý của Hierarchical prompt là ngầm. Nếu định tuyến trở nên kỳ lạ, hãy kết xuất nó ở chế độ chi tiết và đọc.
- **Tác dụng phụ của công cụ trong Phi hành đoàn.** Phi hành đoàn có thể gọi một công cụ nhiều lần hơn dự kiến. POST, DELETE, thanh toán thuộc về bước Flow, không bao giờ là công cụ Crew.

## Bài tập

1. Chuyển đổi nhóm tuần tự thành một Flow. Đếm các điểm tiếp xúc mà sự thay đổi giảm xuống. Lưu ý nơi khả năng đọc giảm.
2. Thêm bộ nhớ thực thể cho phi hành đoàn: sự thật về khách hàng vẫn tồn tại trong suốt các lần khởi động. Xác minh truy xuất kéo đúng thực thể.
3. Thực hiện process phân cấp trong đó người quản lý từ chối định tuyến đến trình soạn thảo cho đến khi đầu ra của người viết có ít nhất ba đoạn. Trace thử lại.
4. Kết nối một lớp con `BaseTool` cho một tìm kiếm web (giả định). So sánh hình dạng trace với phiên bản trang trí `@tool`.
5. Thêm `output_pydantic=Brief` vào tác vụ trình chỉnh sửa, nơi `Brief` có `title`, `summary` `sections`. Làm cho đầu ra tác vụ của người viết bị sai định dạng JSON một lần; xác minh hành vi thử lại của CrewAI trong trace.
6. Đọc phần giới thiệu tài liệu của CrewAI. Chuyển đồ chơi vào `crewai` API thật. Phiên bản stdlib đã bỏ qua những đảm bảo nào?
7. Wire AgentOps hoặc Langfuse (Bài 24) để chạy thực sự. Bạn đã bỏ lỡ traces nào trong phiên bản stdlib?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Agent | "Tính cách" | Vai trò + mục tiêu + cốt truyện + công cụ |
| Nhiệm vụ | "Đơn vị công việc" | Mô tả + đầu ra dự kiến + người được chuyển nhượng + đầu ra có cấu trúc tùy chọn |
| Phi hành đoàn | "Đội Agent" | Container cho Agents + Nhiệm vụ + Process |
| Process | "Chiến lược thực hiện" | Tuần tự / Phân cấp / Đồng thuận (theo kế hoạch) |
| Dòng chảy | "Quy trình làm việc xác định" | Theo hướng sự kiện, sở hữu mã, có thể kiểm tra |
| Cốt truyện | "Persona prompt" | Định hình giọng điệu và phán đoán cho Agent |
| `@tool` | "Công cụ chức năng" | Trình trang trí biến một hàm thành một công cụ mà Agent có thể gọi |
| `BaseTool` | "Công cụ Class" | Công cụ dựa trên Class với schema arg, thử lại, hỗ trợ không đồng bộ |
| Bộ nhớ thực thể | "Sự kiện cho mỗi thực thể" | Bộ nhớ được giới hạn cho khách hàng / tài khoản / vấn đề |
| Trí nhớ dài hạn | "Bộ nhớ chạy chéo" | Bộ nhớ được hỗ trợ Vector tồn tại giữa các lần khởi động |
| Bộ nhớ ngữ cảnh | "Truy xuất kịp thời" | Bộ nhớ được kéo vào thời điểm Agent cần nó |
| Quản lý LLM | "Bộ định tuyến agent" | Thêm LLM trong process phân cấp để chọn nhiệm vụ tiếp theo |
| `expected_output` | "Hợp đồng nhiệm vụ" | Chuỗi cho Agent biết (và kiểm tra) hình dạng nào sẽ trả về |

## Đọc thêm

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): khái niệm và lộ trình production được đề xuất
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows): hình dạng theo sự kiện, `@start`, `@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools): `@tool`, `BaseTool`, bộ công cụ tích hợp
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory): ngắn hạn, dài hạn, thực thể, ngữ cảnh
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): khi đa agent giúp ích và khi nào không
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): giải pháp thay thế máy trạng thái
