# SGLang và RadixChú ý đến khối lượng công việc nặng tiền tố

> SGLang coi KV cache như một tài nguyên class đầu tiên, có thể tái sử dụng được lưu trữ trong cây cơ sở. Trong trường hợp vLLM lên lịch yêu cầu FCFS (ai đến trước được phục vụ trước), bộ lập lịch nhận biết bộ nhớ cache của SGLang ưu tiên các yêu cầu có tiền tố được chia sẻ dài hơn — hiệu quả là một truy cập cơ số ưu tiên độ sâu để nóng branches vẫn nằm trong HBM. Trên Llama 3.1 8B với prompts 1K giống ShareGPT, SGLang đạt ~16.200 tok/s so với ~12.500 của vLLM, lợi nhuận ~29%. Trên khối lượng công việc RAG nhiều tiền tố, lợi thế đạt 6,4 lần. Trên khối lượng công việc hình dạng sao chép giọng nói, tỷ lệ truy cập bộ nhớ đệm đã xóa 86%. Được triển khai trên 400.000+ GPUs vào năm 2026 trên xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS. Vấn đề là số 6,4x bốc hơi khi thứ tự tiền tố không nhất quán - thứ tự là đòn bẩy của kỹ sư.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy radix-tree cache + cache-aware scheduler)
**Kiến thức tiên quyết:** Giai đoạn 17 · 04 (vLLM Phục vụ Nội bộ), Giai đoạn 14 (Agentic RAG)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Sơ đồ RadixChú ý: cách các tiền tố được lưu trữ trong cây cơ số và cách các khối KV được chia sẻ trên các chuỗi bắt nguồn từ cùng một branch.
- Giải thích lập lịch nhận biết bộ nhớ cache và lý do tại sao FCFS sai đối với lưu lượng truy cập nhiều tiền tố.
- Tính toán tốc độ dự kiến cho khối lượng công việc với tỷ lệ truy cập bộ nhớ đệm tiền tố và phân bố độ dài prompt.
- Đặt tên cho kỷ luật sắp xếp thứ tự prompt làm cho con số 6,4x trở thành thật so với một mặt tăng bị mất.

## Vấn đề

Phục vụ cổ điển coi prompt của mỗi yêu cầu là mờ đục. Ngay cả khi 5.000 yêu cầu RAG đều bắt đầu với cùng một 2.000 token system prompt cộng với cùng một lời mở đầu truy xuất, vLLM sẽ điền tiền tố 2.000 token đó 5.000 lần. Các GPU làm đi làm lại cùng một công việc.

Quan sát: prompts trong agentic và RAG khối lượng công việc hầu như luôn chia sẻ tiền tố dài. System prompt, schemas công cụ, ví dụ few-shot, tiêu đề truy xuất, lịch sử cuộc trò chuyện — tất cả đều lặp lại trên các yêu cầu. Nếu bạn lưu trữ KV cache cho tiền tố đó một lần và sử dụng lại, bạn sẽ không điền trước.

RadixAttention thực hiện chính xác điều này. Tokens được lập chỉ mục trong cây cơ sở; mỗi nút sở hữu các khối KV cho chuỗi token trên đường đi của nó từ gốc. Một yêu cầu mới đi theo cây: bất kỳ nút nào có token khớp sẽ sử dụng lại các khối KV của nút đó. Chi phí điền trước tỷ lệ thuận với hậu tố "mới", không phải toàn bộ prompt.

Thách thức là lên lịch. Nếu hai yêu cầu chia sẻ tiền tố 2.000 token và yêu cầu thứ ba chỉ chia sẻ 200 tokens của cùng một tiền tố, bạn muốn phục vụ hai yêu cầu được chia sẻ dài cùng nhau để tiền tố dài vẫn ở trong HBM. FCFS làm ngược lại - nó phục vụ bất kỳ ai đến trước, có khả năng loại bỏ branch nóng trước khi yêu cầu tiền tố dài tiếp theo xuất hiện.

## Khái niệm

### Cây cơ số dưới dạng chỉ số KV

Một cây cơ số (trie nhỏ gọn) lưu trữ các chuỗi token. Mỗi nút sở hữu một phạm vi token và các khối KV được tính toán cho phạm vi đó. Trẻ em kéo dài trình tự một hoặc nhiều tokens.

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

Một yêu cầu mới đi kèm với system prompt + "Ngữ cảnh: <doc A>" + "Câu hỏi: Carol". Bộ lập lịch trình thực hiện: tiền tố hệ thống khớp (124 khối được sử dụng lại), doc-A branch khớp (31 khối được sử dụng lại), sau đó chỉ phân bổ các khối mới cho "Câu hỏi: Carol" (4 khối). Chi phí điền trước: 4 khối tokens mới. Không có cây: 160 khối. Tiết kiệm ~40 lần khi nạp trước.

### Lập lịch nhận biết bộ nhớ cache

Việc sử dụng lại được hỗ trợ bởi cây cơ bản là vô nghĩa nếu bộ nhớ đệm bị xáo trộn. Hai policies chính:

1. **Công văn ưu tiên độ sâu**. Khi chọn yêu cầu tiếp theo từ hàng đợi, hãy ưu tiên các yêu cầu được root ở cùng branch với tập hợp đang chạy hiện tại. Điều này giữ cho branch nóng được ghim.
2. **LRU ở cấp branch, không phải cấp khối**. Loại bỏ toàn bộ branches (bắt đầu từ những chiếc lá được sử dụng ngắn nhất) thay vì các khối riêng lẻ, vì vậy hình dạng bộ nhớ đệm phù hợp với hình dạng cơ sở.

FCFS vi phạm cả hai. Một yêu cầu chia sẻ 2.000 tokens nằm đằng sau yêu cầu chia sẻ 50, sau đó 2.000 token branch bị đuổi ra khỏi nhà để thừa nhận 50 token.

### Benchmark số bạn nên ghi nhớ

- Llama 3.1 8B, H100, ShareGPT 1K prompts: SGLang ~16.200 tok/s so với vLLM ~12.500 (~29% lợi nhuận).
- RAG nặng tiền tố (cùng hệ thống + cùng tài liệu, câu hỏi khác nhau): lên đến 6,4x trên SGLang.
- Khối lượng công việc sao chép giọng nói: 86,4% tỷ lệ truy cập vào bộ nhớ đệm tiền tố.
- Tỷ lệ đạt Production trên toàn bộ khách hàng của SGLang: 50-99% tùy thuộc vào prompt ngành.
- Triển khai trên 400,000+ GPUs vào năm 2026.

### Yêu cầu đặt hàng

Con số 6,4x dựa trên thứ tự mẫu prompt nhất quán. Nếu máy khách của bạn xây dựng prompts dưới dạng `[system, tools, context, history, question]` trong một số yêu cầu và `[system, context, tools, history, question]` trong các yêu cầu khác, cây không thể tìm thấy tiền tố được chia sẻ. Những gì trông giống như một tiền tố được chia sẻ cho một con người là hai chuỗi riêng biệt của cây cơ sở.

Đòn bẩy của kỹ sư: mẫu prompt của bạn là một khóa bộ nhớ cache. Sửa lệnh. Đặt mọi thứ bất biến (hệ thống, công cụ, schemas) lên hàng đầu. Đặt ngữ cảnh truy xuất tiếp theo. Đặt câu hỏi của người dùng cuối cùng. Không xen kẽ nội dung động vào tiền tố.

Trường hợp thực tế từ nghiên cứu: di chuyển nội dung động ra khỏi tiền tố có thể lưu vào bộ nhớ đệm đã mất một lần triển khai từ 7% đến 74% tỷ lệ truy cập bộ nhớ đệm trong một lần thay đổi.

### RadixAttention thắng thua ở đâu

Chiến thắng:
- RAG (cùng một lời mở đầu truy xuất, câu hỏi khác nhau).
- Agents (cùng một công cụ schemas, truy vấn khác nhau).
- Trò chuyện với system prompt dài.
- Khối lượng công việc giọng nói / thị giác với lời mở đầu lặp đi lặp lại.

Thua (trở về thông lượng cấp vLLM):
- Tạo một lần với prompts độc đáo (hoàn thành mã, trò chuyện mở không cần system prompt).
- prompts động trong đó mọi yêu cầu xen kẽ nội dung duy nhất vào tiền tố.

### Tại sao đây là vấn đề của bộ lập lịch, không chỉ là vấn đề hạt nhân

Bạn có thể thực hiện tái sử dụng KV như một thủ thuật hạt nhân. Cái nhìn sâu sắc của SGLang là việc sử dụng lại chỉ trả tiền nếu người lập lịch giữ cho cư dân nóng branch. Một policy "tái sử dụng nếu có" ngây thơ sẽ làm khuấy bộ nhớ đệm dưới tải hỗn hợp. Bộ lập lịch lập chỉ mục cây cơ sở là thứ biến thủ thuật hạt nhân thành lợi production 29%.

### Tương tác với vLLM

Hai hệ thống này không phải là đối thủ cạnh tranh gay gắt. Vào năm 2026, vLLM đã thêm bộ nhớ đệm tiền tố (`--enable-prefix-caching`) và bộ định tuyến nhận biết bộ nhớ cache (Bộ định tuyến vLLM ở Rust). Khoảng cách đã khép lại nhưng không hoàn toàn biến mất - toàn bộ stack của SGLang là cơ sở đầu tiên; vLLM đã ghép nó vào. Đối với khối lượng công việc bị chi phối bởi việc sử dụng lại tiền tố, SGLang vẫn là mặc định. Đối với phân phát mục đích chung mà không có các mẫu tiền tố mạnh, vLLM vẫn bằng hoặc tốt hơn.

```figure
roofline
```

## Ứng dụng

`code/main.py` triển khai một KV cache cây cơ sở đồ chơi cộng với một bộ lập lịch với hai policies: FCFS và nhận biết bộ nhớ cache. Chạy cùng một khối lượng công việc thông qua cả hai, báo cáo tỷ lệ truy cập bộ nhớ đệm tiền tố và delta thông lượng. Sau đó, chạy khối lượng công việc "sắp xếp thứ tự xáo trộn" để hiển thị sự sụp đổ 6,4x.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-radix-scheduler-advisor.md`. Với mô tả khối lượng công việc (hình dạng mẫu prompt, mẫu truy xuất, số lượng tenants đồng thời), nó tạo ra đơn thuốc đặt hàng prompt và go/no-go cho việc áp dụng SGLang.

## Bài tập

1. Chạy `code/main.py`. So sánh FCFS và nhận biết bộ nhớ cache trên cùng một khối lượng công việc. Delta đến từ đâu - tiết kiệm điền trước, tiết kiệm giải mã hoặc độ trễ hàng đợi?
2. Sửa đổi khối lượng công việc để prompts hoán vị ngẫu nhiên `[system, tools, context]`. Chạy lại. Điều gì xảy ra với tỷ lệ trúng đích? Tại sao?
3. Tính toán chi phí HBM để giữ một cư dân 2.000 token system prompt dưới dạng một cơ số branch trên Llama 3.1 8B. So sánh với chi phí của một batch 16 trình tự không sử dụng lại tiền tố.
4. Đọc bài SGLang RadixAttention. Giải thích trong ba câu lý do tại sao việc trục xuất LRU hình cây lại đánh bại LRU hình khối dưới tải nặng tiền tố.
5. Một khách hàng báo cáo chỉ có 8% tỷ lệ truy cập vào bộ nhớ cache. Kể tên ba nguyên nhân có thể xảy ra và chẩn đoán bạn sẽ thực hiện cho từng nguyên nhân.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| RadixChú ý | "điều SGLang" | KV cache được lập chỉ mục dưới dạng cây cơ số để các tiền tố được chia sẻ sử dụng lại các khối |
| Cây cơ số | "Trie nhỏ gọn" | Cây trong đó mỗi nút sở hữu một phạm vi token và các khối KV của nó |
| Bộ lập lịch nhận biết bộ nhớ cache | "Nóng-branch-First" | Trình lập lịch ưu tiên các yêu cầu chia sẻ branch cư trú |
| Tỷ lệ truy cập vào bộ nhớ đệm tiền tố | "Bao nhiêu prompt của bạn là miễn phí" | Phần prompt tokens được phục vụ từ các khối KV tái sử dụng |
| FCFS | "ai đến trước được phục vụ trước" | Lập lịch mặc định phá vỡ cục bộ tiền tố |
| LRU cấp Branch | "Đuổi chiếc lá" | Trục xuất policy phù hợp với hình dạng cơ sở |
| Prompt sắp xếp mẫu | "Khóa bộ nhớ đệm" | Thứ tự thành phần của prompt xác định những gì cây có thể chia sẻ |
| Ghim System prompt | "Tiền tố cư trú" | Giữ phần hệ thống bất biến được ghim để tránh bị trục xuất |

## Đọc thêm

- [SGLang GitHub](https://github.com/sgl-project/sglang) — nguồn và tài liệu.
- [SGLang documentation](https://sgl-project.github.io/) — RadixChú ý và chi tiết lên lịch.
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) - tham chiếu thiết kế.
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) - benchmark số và lý do lập lịch.
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — triển khai giống như cơ số của riêng vLLM, để so sánh.
