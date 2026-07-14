# Production Quantization — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

> Định dạng Quantization không phải là một lựa chọn phổ biến - nó là một chức năng của phần cứng, công cụ phục vụ và khối lượng công việc. GGUF Q4_K_M hoặc Q5_K_M sở hữu CPU và cạnh, được phân phối thông qua llama.cpp và Ollama. GPTQ chiến thắng bên trong vLLM khi bạn cần nhiều LoRA trên cùng một cơ sở. AWQ với nhân Marlin-AWQ cung cấp ~741 tok/s trên class model 7B với Pass@1 tốt nhất tại INT4 - mặc định năm 2026 cho production trung tâm dữ liệu. FP8 vẫn ở vị trí trung gian trên Hopper, Ada và Blackwell - gần như không mất dữ liệu và được hỗ trợ rộng rãi. NVFP4 và MXFP4 (Blackwell microscaling) rất tích cực và yêu cầu xác thực trên mỗi khối. Hai bẫy cắn đội: hiệu chuẩn dataset phải phù hợp với miền triển khai và KV cache tách biệt với trọng lượng quantization - bài học AWQ "model của tôi bây giờ là 4 GB" quên KV cache 10-30 GB ở production batch kích thước.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy memory and throughput comparison across formats)
**Kiến thức tiên quyết:** Giai đoạn 10 · 13 (Quantization nền tảng), Giai đoạn 17 · 04 (vLLM phục vụ nội bộ)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Kể tên sáu định dạng production quantization và điểm hấp dẫn của chúng vào năm 2026.
- Chọn một định dạng phần cứng nhất định (CPU so với GPU, Hopper vs Blackwell), công cụ (vLLM, TRT-LLM, llama.cpp) và khối lượng công việc (trò chuyện thông thường, lý luận, đa LoRA).
- Tính toán bộ nhớ trọng lượng đã lưu và KV cache còn nguyên cho định dạng đã chọn.
- Đặt tên cho cạm bẫy dataset hiệu chuẩn làm giảm models lượng tử hóa trên lưu lượng truy cập tên miền.

## Vấn đề

Quantization làm giảm bộ nhớ và băng thông HBM, đó chính xác là những gì giải mã cần. Một model FP16 70B có trọng lượng 140 GB. Lượng tử hóa trọng số thành INT4 (AWQ hoặc GPTQ) và model là 35 GB - phù hợp với một H100 có chỗ cho KV cache, điều này rất quan trọng vì ở 128 chuỗi đồng thời với ngữ cảnh 2k, chỉ riêng KV cache là 20-30 GB.

Nhưng quantization không miễn phí. Sự quantization tích cực làm giảm chất lượng, đặc biệt là đối với các nhiệm vụ nặng về lý luận. Các định dạng khác nhau hoạt động với các công cụ khác nhau. Phần cứng khác nhau hỗ trợ các độ chính xác khác nhau nguyên bản. Vườn thú định dạng năm 2026 là có thật và bạn không thể sao chép sự lựa chọn của người khác - bạn phải chọn dựa trên stack của mình.

## Khái niệm

### Sáu định dạng

| Định dạng | Bit | Điểm ngọt ngào | Động cơ |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU, cạnh, máy tính xách tay | llama.cpp, Ollama |
| GPTQ | 4-8 | Multi-LoRA trên vLLM | vLLM, TGI |
| AWQ | 4 | Trung tâm dữ liệu GPU production | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Trung tâm dữ liệu Hopper/Ada/Blackwell | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell nhiều người dùng | TRT-LLM |
| NVFP4 | 4 | Blackwell nhiều người dùng | TRT-LLM |

### GGUF — mặc định CPU/edge

GGUF là một định dạng tệp, không phải sơ đồ quantization - nó gói các biến thể K-quant (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0) trong một container. Q4_K_M và Q5_K_M là mặc định production - chất lượng gần BF16 ở 4-5 bit. Sự lựa chọn tốt nhất cho CPU hoặc giao bóng cạnh vì llama.cpp cho đến nay là động cơ CPU inference nhanh nhất.

Hình phạt thông lượng trong vLLM: ~93 tok/s trên 7B — định dạng không được tối ưu hóa cho GPU nhân. Sử dụng GGUF khi mục tiêu triển khai là CPU/edge. Không khác.

### GPTQ — đa LoRA trong vLLM

GPTQ là một thuật toán sau training quantization với hiệu chuẩn. Nhân Marlin giúp nó nhanh trên GPU (tăng tốc gấp 2,6 lần so với GPTQ không phải Marlin). ~712 tok/s trên 7B.

Chiến thắng độc đáo: GPTQ-Int4 hỗ trợ bộ điều hợp LoRA trong vLLM. Nếu bạn đang phân phát một model cơ sở cộng với 10-50 biến thể fine-tuned (mỗi biến thể dưới dạng LoRA), GPTQ là đường dẫn của bạn. NVFP4 chưa hỗ trợ LoRA kể từ đầu năm 2026.

### AWQ — trung tâm dữ liệu GPU mặc định

Trọng lượng nhận biết kích hoạt Quantization. Bảo vệ trọng lượng nổi bật nhất ~1% trong quá trình quantization. Nhân Marlin-AWQ: Tăng tốc gấp 10,9 lần so với ngây thơ. ~741 tok/s trên 7B, Pass@1 tốt nhất trong số các định dạng INT4.

Chọn AWQ để phân phối GPU mới trừ khi bạn cần đa LoRA (GPTQ) hoặc Blackwell FP4 (NVFP4) mạnh mẽ.

### FP8 — trung gian đáng tin cậy

Dấu phẩy động 8 bit. Gần như không mất dữ liệu. Được hỗ trợ rộng rãi. Lõi Tensor phễu tăng tốc FP8 nguyên bản. Blackwell thừa kế. FP8 là mặc định an toàn năm 2026 khi chất lượng không thể thương lượng (lý luận, y tế, tạo mã). Tiết kiệm bộ nhớ bằng một nửa INT4 nhưng rủi ro chất lượng thấp hơn nhiều.

### MXFP4 / NVFP4 - Blackwell hung hăng

Microscaling FP4. Mỗi khối trọng lượng có hệ số tỷ lệ riêng. Mạnh mẽ nhưng tăng tốc phần cứng trên Blackwell Tensor Cores. Giảm một nửa số byte trên token so với FP8 - chiến thắng kinh tế trong Giai đoạn 17 · 07.

Lưu ý:
- Chưa có hỗ trợ LoRA (đầu năm 2026).
- Chất lượng giảm có thể nhìn thấy trên khối lượng công việc nặng về lý luận.
- Xác thực trên bộ đánh giá của bạn trên mỗi model.

### Bẫy hiệu chuẩn

AWQ và GPTQ yêu cầu dataset hiệu chuẩn — thường là C4 hoặc WikiText. Đối với models miền (mã, y tế, pháp lý), hiệu chỉnh trên văn bản web chung cho phép thuật toán đưa ra quyết định sai về trọng số cần bảo vệ. Pass@1 trên HumanEval có thể giảm vài điểm.

Cách khắc phục: hiệu chỉnh trên dữ liệu trong miền. Hàng trăm mẫu tên miền thường là đủ. Kiểm tra trên bộ đánh giá trước khi shipping.

### Bẫy KV cache

AWQ thu nhỏ trọng lượng xuống còn 4 bit. KV cache riêng biệt và ở FP16/FP8. Đối với model 70B với AWQ:

- Trọng lượng: ~35 GB (INT4 từ 140 GB).
- KV cache ở 128 đồng thời × ngữ cảnh 2k: ~20 GB.
- Kích hoạt: ~5 GB.
- Tổng cộng: ~60 GB - phù hợp với H100 80GB.

Ngây thơ "Tôi đã lượng tử model của mình thành 4 GB" quên 30-50 GB còn lại. Ngân sách HBM một cách toàn diện.

Riêng biệt, KV cache quantization (FP8 KV hoặc INT8 KV) là một lựa chọn khác với sự đánh đổi riêng - nó ảnh hưởng trực tiếp đến attention accuracy và không phải là chiến thắng miễn phí.

### AWQ INT4 nguy hiểm cho lý luận

Chain-of-thought, toán học, tạo mã với ngữ cảnh dài - những thứ này bị ảnh hưởng rõ ràng bởi quantization tích cực. AWQ INT4 mất ~3-5 điểm trên TOÁN. Đối với khối lượng công việc nặng về lý luận, ship FP8 hoặc BF16; chấp nhận chi phí bộ nhớ.

### Hướng dẫn chọn hàng năm 2026

- CPU/edge phục vụ: GGUF Q4_K_M. Xong.
- GPU phục vụ, trò chuyện thông thường, không có LoRA: AWQ.
- GPU phục vụ, đa LoRA: GPTQ với Marlin.
- Khối lượng công việc suy luận: FP8.
- Trung tâm dữ liệu Blackwell, chất lượng đã được xác nhận: NVFP4 + FP8 KV.
- Mơ hồ: chạy đánh giá 1.000 mẫu trên mỗi định dạng ứng viên.

```figure
gpu-memory-breakdown
```

## Ứng dụng

`code/main.py` tính toán dung lượng bộ nhớ (trọng lượng + KV + kích hoạt) và thông lượng tương đối trên sáu định dạng cho một loạt các kích thước model. Cho biết nơi KV cache chiếm ưu thế, nơi nén trọng lượng được đền đáp và FP8 là lựa chọn an toàn.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-quantization-picker.md`. Với phần cứng, kích thước model, loại khối lượng công việc và dung sai chất lượng, hãy chọn một định dạng và tạo ra một kế hoạch calibration/validation.

## Bài tập

1. Chạy `code/main.py`. Đối với model 70B ở 128 đồng thời với ngữ cảnh 2k, hãy tính tổng HBM cho mỗi định dạng. Định dạng nào cho phép bạn vừa với một H100 80GB?
2. Bạn có một model mã hóa 7B. Chọn một định dạng và biện minh. Nếu bạn sai về dung sai chất lượng, con đường phục hồi là gì?
3. Tính toán kích thước dataset hiệu chuẩn cần thiết để hiệu chỉnh AWQ cho model lĩnh vực y tế. Tại sao nhiều dữ liệu hơn không phải lúc nào cũng tốt hơn?
4. Đọc tài liệu hạt nhân Marlin-AWQ hoặc ghi chú phát hành. Giải thích trong ba câu lý do tại sao AWQ đạt 741 tok/s trên 7B trong khi GPTQ thô đạt ~712.
5. Khi nào thì hợp lý khi kết hợp trọng số AWQ với FP8 KV cache so với giữ KV ở BF16?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| GGUF | "Định dạng llama.cpp" | Định dạng tệp gói các biến thể K-quant; CPU/edge mặc định |
| Q4_K_M | "Q4 KM" | Trung bình K-quant 4 bit; production GGUF mặc định |
| GPTQ | "Gee Pee Tee Q" | INT4 sau tàu với hiệu chuẩn; hỗ trợ LoRA trong vLLM |
| AWQ | "a w q" | INT4 nhận biết kích hoạt; Hạt Marlin; Pass@1 tốt nhất tại INT4 |
| Hạt Marlin | "nhân INT4 nhanh" | Hạt nhân CUDA tùy chỉnh cho INT4 trên Phễu; Tăng tốc gấp 10 lần |
| FP8 | "float tám bit" | An toàn precision mặc định trên Hopper/Ada/Blackwell |
| MXFP4 / NVFP4 | "Bốn quy mô vi mô" | Blackwell 4-bit FP với các hệ số tỷ lệ trên mỗi khối |
| Hiệu chuẩn dataset | "Dữ liệu CAL" | Nhập văn bản được sử dụng để chọn quantization parameters; Phải khớp với miền |
| KV cache quantization | "KV INT8" | Lựa chọn riêng biệt với trọng lượng; Điều này sẽ ảnh hưởng đến attention accuracy |

## Đọc thêm

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — benchmarks so sánh.
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — số thông lượng theo định dạng.
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — chọn theo định dạng theo định dạng.
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — các định dạng và cờ được hỗ trợ.
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) - công thức AWQ ban đầu.
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) - công thức GPTQ ban đầu.
