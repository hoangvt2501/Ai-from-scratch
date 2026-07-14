# Lựa chọn phục vụ tự tổ chức — llama.cpp, Ollama, TGI, vLLM, SGLang

> Bốn động cơ thống trị inference tự lưu trữ vào năm 2026. Chọn dựa trên phần cứng, quy mô và hệ sinh thái. **llama.cpp** nhanh nhất trên CPU - hỗ trợ model rộng nhất, kiểm soát hoàn toàn quantization và phân luồng. **Ollama **là cài đặt một lệnh dành cho nhà phát triển máy tính xách tay, chậm hơn ~15-30% so với llama.cpp (Go + CGo + HTTP serialization), khoảng cách thông lượng gấp 3 lần khi tải giống như sản phẩm. **TGI đã vào chế độ bảo trì ngày 11 tháng 12 năm 2025** — chỉ sửa lỗi, thông lượng thô chậm hơn ~10% so với vLLM nhưng tích hợp hệ sinh thái HF và observability hàng đầu trong lịch sử. Tình trạng bảo trì đó khiến nó trở thành một vụ đặt cược dài hạn rủi ro - SGLang hoặc vLLM là mặc định an toàn hơn cho các dự án mới. **vLLM** là production mặc định có mục đích chung — v0.15.1 (Tháng 2 năm 2026) bổ sung tối ưu hóa PyTorch 2.10, RTX Blackwell SM120, H200. **SGLang** là chuyên gia nhiều lượt / tiền tố nặng agentic - 400.000+ GPUs trong production (xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS). Hạn chế phần cứng: Chỉ → llama.cpp CPU. Chỉ AMD / không NVIDIA → vLLM (TRT-LLM bị khóa NVIDIA). Mô hình pipeline 2026: dev = Ollama, staging = llama.cpp, prod = vLLM hoặc SGLang. Cùng GGUF/HF trọng lượng xuyên suốt.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, engine-decision tree walker)
**Kiến thức tiên quyết:** Tất cả các bài học Giai đoạn 17 bao gồm động cơ (04, 06, 07, 09, 18)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Chọn một công cụ phần cứng nhất định (CPU / AMD / NVIDIA Hopper / Blackwell), quy mô (1 người dùng / 100 / 10.000) và khối lượng công việc (trò chuyện chung / agent / ngữ cảnh dài).
- Kể tên trạng thái chế độ bảo trì TGI năm 2026 (ngày 11 tháng 12 năm 2025) và lý do tại sao nó thiên về các dự án mới về vLLM hoặc SGLang.
- Mô tả dev/staging/prod pipeline sử dụng cùng một trọng số GGUF hoặc HF xuyên suốt.
- Giải thích lý do tại sao "chỉ CPU" buộc llama.cpp và "AMD" loại trừ TRT-LLM.

## Vấn đề

Nhóm của bạn bắt đầu một dự án LLM tự lưu trữ mới. Một kỹ sư nói Ollama, một kỹ sư khác nói vLLM, một kỹ sư thứ ba nói "không phải TGI chỉ hoạt động ngay lập tức sao?" Cả ba đều phù hợp với các bối cảnh khác nhau. Không có gì phù hợp với tất cả.

Vào năm 2026, cây lựa chọn quan trọng: phần cứng đầu tiên, quy mô thứ hai, khối lượng công việc thứ ba. Và một sự kiện cụ thể vào năm 2025 - TGI chuyển sang chế độ bảo trì vào ngày 11 tháng 12 - thay đổi mặc định cho các dự án mới.

## Khái niệm

### Năm động cơ

| Động cơ | Tốt nhất cho | Ghi chú |
|--------|----------|-------|
| **llama.cpp** | CPU / cạnh / deps tối thiểu / hỗ trợ model rộng nhất | Nhanh nhất trên CPU, kiểm soát hoàn toàn |
| **Ollama** | Máy tính xách tay dành cho nhà phát triển, một người dùng, cài đặt một lệnh | chậm hơn 15-30% so với llama.cpp; Khoảng cách thông lượng sản phẩm gấp 3 lần |
| **TGI** | Hệ sinh thái HF, các ngành được quản lý | **Chế độ bảo trì Dec 11, 2025** |
| **vLLM** | production đa năng, 100+ người dùng | Mặc định production rộng; v0.15.1 Tháng Hai 2026 |
| **SGLang** | Agentic khối lượng công việc nhiều lượt, nhiều tiền tố | 400.000+ GPUs trong production |

### Quyết định ưu tiên phần cứng

**Chỉ CPU **→ llama.cpp. Ollama cũng hoạt động nhưng chậm hơn. Không có động cơ nào khác cạnh tranh trên CPU.

**AMD GPU** → vLLM (hỗ trợ AMD ROCm). SGLang cũng hoạt động. TRT-LLM bị khóa NVIDIA, vì vậy nó đã hết.

**NVIDIA Phễu (H100 / H200) **→ vLLM hoặc SGLang hoặc TRT-LLM. Cả ba hạng cao nhất.

**NVIDIA Blackwell (B200 / GB200) **→ TRT-LLM là công ty dẫn đầu thông lượng (Giai đoạn 17 · 07). vLLM và SGLang theo sát.

**Apple Silicon (dòng M)** → llama.cpp (Kim loại). Ollama bọc điều này.

### Quyết định tỷ lệ giây

**1 người dùng / nhà phát triển cục bộ **→ Ollama. Một lệnh, token đầu tiên trong vài giây.

**10-100 người dùng / nhóm nhỏ** → vLLM một GPU.

**100-10 nghìn người dùng / production** → vLLM production-stack (Giai đoạn 17 · 18) hoặc SGLang.

**10k + người dùng / doanh nghiệp **→ vLLM production-stack + phân tách (Giai đoạn 17 · 17) + LMCache (Giai đoạn 17 · 18).

### Quyết định thứ ba về khối lượng công việc

**Trò chuyện chung / Hỏi & đáp **→ vLLM chiến thắng trên mặc định rộng.

**Agentic nhiều lượt (công cụ, lập kế hoạch, bộ nhớ)** → RadixAttention của SGLang (Giai đoạn 17 · 06) chiếm ưu thế.

**RAG với việc sử dụng lại tiền tố nặng** → SGLang.

**Tạo mã** → vLLM tốt; SGLang tốt hơn một chút trên bộ nhớ cache.

**Ngữ cảnh dài (128K+)** → vLLM + điền trước theo đoạn; SGLang + KV phân tầng.

### Bẫy bảo trì TGI

Hugging Face TGI vào chế độ bảo trì vào ngày 11 tháng 12 năm 2025 - chỉ sửa lỗi trong tương lai. Trong lịch sử: observability hàng đầu, tích hợp hệ sinh thái HF tốt nhất trong class (thẻ model, công cụ an toàn), kém một chút so với vLLM về thông lượng thô.

Đối với các dự án mới trong năm 2026: vỡ nợ khỏi TGI. Các triển khai TGI hiện tại có thể tiếp tục nhưng cuối cùng sẽ di chuyển. SGLang và vLLM là mặc định an toàn hơn.

### Mô hình pipeline

Dev (Ollama) → dàn dựng (llama.cpp) → prod (vLLM). Cùng trọng lượng GGUF hoặc HF xuyên suốt. Các kỹ sư lặp lại nhanh chóng trên máy tính xách tay; gương dàn dựng production quantization; PROD là mục tiêu phục vụ.

### Cảnh báo Ollama

Ollama rất tốt cho nhà phát triển. Nó không tuyệt vời cho các production được chia sẻ: Đi HTTP tuần tự hóa thêm chi phí, quản lý đồng thời đơn giản hơn vLLM, hỗ trợ OpenTelemetry bị trễ. Sử dụng Ollama ở nơi nó tỏa sáng - một người dùng, một lệnh - và chuyển sang vLLM để chia sẻ.

### Tự lưu trữ và quản lý là một quyết định riêng biệt

Giai đoạn 17 · 01 (siêu quy mô được quản lý), · 02 (inference bệ) bìa được quản lý. Bài học này giả định rằng bạn đã quyết định tự tổ chức. Lý do để tự lưu trữ: nơi lưu trữ dữ liệu, fine-tune tùy chỉnh, tổng chi phí sở hữu trên quy mô lớn, tên miền model không khả dụng trên lưu trữ.

### Những con số bạn nên nhớ

- Chế độ bảo trì TGI: Tháng Mười Hai 11, 2025.
- vLLM v0.15.1: Tháng Hai 2026; PyTorch 2.10; Hỗ trợ Blackwell SM120.
- Dấu chân production SGLang: 400.000+ GPUs.
- Khoảng cách thông lượng Ollama so với llama.cpp: chậm hơn 15-30%; 3x dưới tải prod.

```figure
data-parallel
```

## Ứng dụng

`code/main.py` là một người đi bộ cây quyết định: cho phần cứng + quy mô + khối lượng công việc, chọn một công cụ và giải thích lý do tại sao.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-engine-picker.md`. Với các ràng buộc, chọn một công cụ và viết kế hoạch di chuyển.

## Bài tập

1. Chạy `code/main.py` với phần cứng / quy mô / khối lượng công việc của bạn. Đầu ra có phù hợp với trực giác của bạn không?
2. Cơ sở hạ tầng của bạn là 12 H100 và 8 MI300X AMD. Động cơ gì? Tại sao TRT-LLM lại ra khỏi bàn?
3. Một đội muốn sử dụng TGI vào năm 2026 vì "đó là những gì chúng tôi biết". Tranh luận về trường hợp di cư.
4. Ollama dev to vLLM prod: những thay đổi nào trong quantization, configuration và observability?
5. RAG sản phẩm có tiền tố P99 dài 8K và khả năng tái sử dụng cao trên tenants. Chọn một động cơ và stack nó với Giai đoạn 17 · 11 + 18.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| llama.cpp | "Người CPU" | Hỗ trợ model rộng nhất, nhanh nhất trên CPU |
| Ollama | "Máy tính xách tay" | Cài đặt một lệnh, thông lượng cấp nhà phát triển |
| TGI | "Phục vụ của HF" | Chế độ bảo trì từ tháng 12 năm 2025 |
| vLLM | "mặc định" | Đường cơ sở production rộng 2026 |
| SGLang | "Người agentic" | Tiền tố nặng, RadixAttention |
| TRT-LLM | "Khóa NVIDIA" | Chỉ dẫn đầu thông lượng Blackwell, NVIDIA |
| GGUF | "Định dạng llama.cpp" | Các biến thể K-quant đi kèm |
| Production-stack | "vLLM K8s" | Giai đoạn 17 · 18 Triển khai tham khảo |
| Pipeline mẫu | "dev→stage→prod" | Ollama → llama.cpp → vLLM trên cùng trọng lượng |

## Đọc thêm

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — ghi chú phát hành.
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
