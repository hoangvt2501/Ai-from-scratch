# Edge Inference - Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> Hạn chế cạnh cốt lõi là băng thông bộ nhớ, không phải điện toán. DRAM di động ở mức 50-90 GB/s; trung tâm dữ liệu HBM3 xóa 2-3 TB/s - khoảng cách 30-50x. Giải mã bị ràng buộc bởi bộ nhớ nên khoảng cách là quyết định. Vào năm 2026, cảnh quan chia cắt bốn cách. Apple M4/A18 Neural Engine đạt đỉnh 38 TOPS với bộ nhớ hợp nhất (không CPU↔bản NPU). Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon đạt 45 TOPS. WebGPU + WebLLM chạy Llama 3.1 8B (Q4) ở ~41 tok/s trên M3 Max (khoảng 70-80% gốc); 17,6K GitHub sao, API tương thích với OpenAI, ~70-75% vùng phủ sóng di động. NVIDIA Jetson Orin Nano Super (8GB) phù hợp với Llama 3.2 3B / Phi-3; AGX Orin chạy gpt-oss-20b qua vLLM ở ~40 tok/s; Jetson T4000 (JetPack 7.1) là 2x AGX Orin. TensorRT Edge-LLM hỗ trợ EAGLE-3, NVFP4, chiết rót trước theo khối — được giới thiệu tại CES 2026 bởi Bosch, ThunderSoft, MediaTek.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy bandwidth-bound decode simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 04 (vLLM Phục vụ Nội bộ), Giai đoạn 17 · 09 (Production Quantization)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Giải thích lý do tại sao LLM inference di động bị ràng buộc với băng thông bộ nhớ và điện toán là thứ yếu.
- Liệt kê bốn mục tiêu biên (Apple ANE, Qualcomm Hexagon, WebGPU/WebLLM NVIDIA Jetson) và khớp từng mục tiêu với một trường hợp sử dụng.
- Kể tên khoảng cách vùng phủ sóng WebGPU năm 2026 (Firefox Android bắt kịp) và Safari iOS 26 hạ cánh.
- Chọn định dạng quantization cho mỗi mục tiêu (Core ML INT4 + FP16 cho ANE, QNN INT8/INT4 cho Hexagon, WebGPU Q4 cho trình duyệt, NVFP4 cho Jetson Thor).

## Vấn đề

Khách hàng muốn có một chatbot trên thiết bị: ưu tiên giọng nói, riêng tư theo mặc định, hoạt động ngoại tuyến. Trên MacBook Pro M3 Max, Llama 3.1 8B Q4 chạy ở ~55 tok/s - tốt. Trên iPhone 16 Pro, model tương tự chạy ở 3 tok/s - không ổn. Trên Android tầm trung với Snapdragon 8 Gen 3, 7 tok/s. Trong trình duyệt qua WebGPU trên Chrome Android v121+, 4-8 tok/s tùy thiết bị.

Thông lượng variance không phải là vấn đề chuyển đổi. Đó là khoảng cách băng thông nhân với định dạng quantization nhân với liệu NPU có thể truy cập được từ không gian người dùng hay không. Edge inference vào năm 2026 là bốn vấn đề khác nhau với bốn giải pháp khác nhau.

## Khái niệm

### Băng thông là trần thực sự

Giải mã đọc toàn bộ trọng số cho mỗi token. Một model 7B trong quý 4 là 3,5 GB. Đọc 3,5 GB ở 50 GB/s mất 70 mili giây — trần lý thuyết là ~14 tok/s. Ở 90 GB/s (DRAM di động cao cấp), trần di chuyển đến ~25 tok/s. Không có số lượng điện toán nào giúp thấp hơn con số này.

Trung tâm dữ liệu HBM3 ở 3 TB/s xóa cùng 3.5 GB trong 1.2 ms - trần là 830 tok/s. Cùng model, cùng trọng lượng. Hệ thống con bộ nhớ khác nhau.

### Công cụ thần kinh của Apple (M4 / A18)

- Lên đến 38 TOPS. Bộ nhớ hợp nhất (CPU và ANE chia sẻ cùng một nhóm) — không có chi phí sao chép.
- Truy cập qua Core ML + `.mlmodel` biên dịch models hoặc qua Metal Performance Shaders (MPS) đến PyTorch.
- Llama.cpp Phần phụ trợ Metal sử dụng MPS, không phải ANE trực tiếp; ANE gốc yêu cầu chuyển đổi ML cốt lõi.
- Con đường thực tế tốt nhất cho ứng dụng iOS vào năm 2026: Core ML với trọng số INT4 + kích hoạt FP16.

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- Lên đến 45 TOPS. Tích hợp với CPU và GPU trong SoC nhưng miền bộ nhớ riêng biệt.
- QNN (Qualcomm Neural Network) SDK và AI Hub cung cấp chuyển đổi từ PyTorch/ONNX.
- Các mẫu trò chuyện, Llama 3.2, Phi-3 đều ship là class artifacts đầu tiên trên AI Hub.

### NPU Intel / AMD (Hồ Mặt trăng, Ryzen AI 300)

- 40-50 ĐẦU. Phần mềm tụt hậu so với Apple/Qualcomm; OpenVINO đang được cải thiện nhưng thích hợp.
- Tốt nhất cho các ứng dụng phi công phụ Windows ARM; gốc trên máy tính để bàn AMD/Intel dành cho địa phương ưu tiên.

### WebGPU + WebLLM

- Chạy models trong trình duyệt thông qua trình đổ bóng điện toán WebGPU; Không cài đặt.
- Llama 3.1 8B Q4 ở ~41 tok/s trên M3 Max - khoảng 70-80% gốc thông qua cùng một chương trình phụ trợ.
- 17.6 nghìn GitHub sao trên WebLLM; JS API tương thích với OpenAI; Apache 2.0.
- Độ phủ sóng năm 2026: Chrome Android v121+, Safari iOS 26 GA, Firefox Android vẫn bắt kịp. Vùng phủ sóng di động tổng thể ~70-75%.

### NVIDIA Gia đình Jetson

- Orin Nano Super (8GB): phù hợp với Llama 3.2 3B, Phi-3 ở tok/s. tốt
- AGX Orin: chạy gpt-oss-20b qua vLLM ở ~40 tok/s.
- Thor / T4000 (JetPack 7.1): 2x hiệu suất AGX Orin, hỗ trợ EAGLE-3 và NVFP4.
- TensorRT Edge-LLM (2026) hỗ trợ giải mã suy đoán EAGLE-3, trọng số NVFP4, điền trước theo từng đoạn — tối ưu hóa trung tâm dữ liệu được chuyển sang biên.

### Quantization lựa chọn cho mỗi mục tiêu

| Mục tiêu | Định dạng | Ghi chú |
|--------|--------|-------|
| Táo ANE | Trọng lượng INT4 + kích hoạt FP16 | Đường dẫn chuyển đổi ML cốt lõi |
| Qualcomm lục giác | QNN INT8 / INT4 | Bộ chuyển đổi AI Hub |
| WebGPU / WebLLM | MLC quý 4 (q4f16_1) | Sử dụng `mlc_llm convert_weight` + `.wasm` biên dịch; GGUF không được hỗ trợ |
| Máy bay phản lực Orin Nano | Q4 GGUF hoặc TRT-LLM INT4 | Giới hạn bộ nhớ |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Đường dẫn LLM cạnh |

### Bẫy bối cảnh dài trên rìa

Bối cảnh 128K của Llama 3.1 là một feature trung tâm dữ liệu. Trên điện thoại có RAM 8 GB, model 4 GB + KV cache 2 GB cho tokens 32K + chi phí hệ điều hành = OOM. Triển khai biên giữ ngữ cảnh ở 4K-8K trừ khi KV tích cực quantization (Q4 KV) được chấp nhận.

### Giọng nói là ứng dụng sát thủ

agents giọng nói nhạy cảm với độ trễ (đầu tiên token < 500 mili giây). Local inference loại bỏ hoàn toàn độ trễ mạng. Kết hợp với chuyển giọng nói thành văn bản (các biến thể Whisper Turbo chạy trên cạnh) và inference cạnh trở thành vòng lặp giọng nói chất lượng production.

### Những con số bạn nên nhớ

- Apple M4 / A18 ANE: 38 ĐẦU.
- Qualcomm Hexagon SD X Elite: 45 ĐỈNH
- WebLLM M3 Tối đa: ~41 tok/s trên Llama 3.1 8B Q4.
- AGX Orin: ~40 tok/s trên gpt-oss-20b qua vLLM.
- Khoảng cách băng thông biên trung tâm dữ liệu: 30-50x.
- Phạm vi phủ sóng di động WebGPU: ~70-75% (độ trễ của Firefox Android).

## Ứng dụng

`code/main.py` tính toán trần thông lượng giải mã lý thuyết từ toán học giới hạn băng thông trên các mục tiêu biên. So sánh với benchmarks quan sát được và điểm nổi bật trong đó băng thông, không phải tính toán, là nút thắt cổ chai.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-edge-target-picker.md`. Nền tảng (iOS/Android/browser/Jetson), model và ngân sách latency/memory nhất định, chọn định dạng quantization và pipeline chuyển đổi.

## Bài tập

1. Chạy `code/main.py`. Đối với 7B model trong quý 4 trên Snapdragon 8 Gen 3 (~77 GB/s băng thông), hãy tính trần giải mã. So với 6-8 tok/s quan sát được - runtime có hiệu quả không?
2. WebGPU trên Android yêu cầu Chrome v121+. Thiết kế dự phòng cho các trình duyệt cũ hơn — server phía thông qua cùng một API tương thích với OpenAI.
3. Ứng dụng iOS của bạn cần streaming ngữ cảnh 4K. Sự kết hợp model/format nào cho phép bạn duy trì bộ nhớ hoạt động dưới 4 GB trên iPhone 16?
4. Jetson AGX Orin chạy gpt-oss-20b ở 40 tok/s. Jetson Nano chỉ phù hợp với 3B. Nếu sản phẩm của bạn nhắm mục tiêu cả hai, làm thế nào để bạn thống nhất inference stack?
5. Tranh luận liệu "WebLLM có sẵn sàng production vào năm 2026 hay không. Trích dẫn phạm vi phủ sóng, hiệu suất và khoảng cách Firefox Android.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| ANE | "Động cơ thần kinh của Apple" | NPU trên thiết bị ở dòng M và dòng A; Bộ nhớ hợp nhất |
| Hình lục giác | "NPU Qualcomm" | NPU snapdragon; QNN SDK để truy cập |
| WebGPU | "Trình duyệt GPU" | GPU API trình duyệt được tiêu chuẩn hóa W3C; Chrome/Safari 2026 |
| WebLLM | "Trình duyệt LLM runtime" | Dự án MLC-LLM; Apache 2.0; JS tương thích với OpenAI |
| Máy bay Jetson | "NVIDIA cạnh" | Dòng Orin Nano / AGX / Thor / T4000 |
| TRT Edge-LLM | "cạnh TensorRT" | Cổng biên 2026 của TensorRT-LLM; ĐẠI BÀNG-3 + NVFP4 |
| Bộ nhớ hợp nhất | "Hồ bơi chung" | CPU và NPU thấy cùng một RAM; không có chi phí sao chép |
| Giới hạn băng thông | "Bộ nhớ bị giới hạn" | Giải mã được kiểm soát bởi trọng số đọc bytes/sec |
| ML lõi | "Chuyển đổi Apple" | Apple framework cho ANE-Native models |
| QNN | "Qualcomm stack" | Mạng thần kinh Qualcomm SDK |

## Đọc thêm

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) - phong cảnh và benchmarks.
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) - Orin / AGX / Thor.
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) - Thông báo cổng biên năm 2026.
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) - thiết kế và benchmarks.
- [Apple Core ML](https://developer.apple.com/documentation/coreml) - Chuyển đổi ANE-gốc.
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — models được chuyển đổi trước cho Hexagon.
