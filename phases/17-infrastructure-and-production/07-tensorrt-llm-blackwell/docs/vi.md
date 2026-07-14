# TensorRT-LLM trên Blackwell với FP8 và NVFP4

> TensorRT-LLM chỉ có NVIDIA nhưng nó thắng Blackwell. Trên GB200 NVL72 với Dynamo orchestration, SemiAnalysis InferenceX đã đo $0.012 per million tokens on a 120B model in Q1-Q2 2026, against $ 0.09/M trên H100 + vLLM - khoảng cách kinh tế gấp 7 lần. stack là ba chế độ dấu phẩy động được kết hợp với nhau: FP8 vẫn rất quan trọng đối với các hạt nhân KV cache và attention vì nó có dải động mà chúng cần; NVFP4 (vi mô 4 bit) xử lý trọng số và kích hoạt; Dự đoán đa token (MTP) và phân tách prefill/decode thêm 2-3x khác lên trên. Hỗ trợ model Day-0 tải trọng lượng FP4 trực tiếp mà không cần chuyển đổi sau training. Điểm mấu chốt cho các nhóm kỹ sư năm 2026: TRT-LLM là một NVIDIA stack khép kín, vì vậy việc áp dụng nó sẽ đánh đổi tính di động để lấy thông lượng. Chạy phép toán trên hỗn hợp models và phần cứng của bạn trước khi cam kết.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy FP8/NVFP4 memory and cost calculator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 04 (vLLM Phục vụ Nội bộ), Giai đoạn 10 · 13 (Quantization)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Giải thích lý do tại sao FP8 vẫn quan trọng đối với KV cache và attention ngay cả khi trọng số nằm trong NVFP4.
- Tính toán dấu chân HBM của một model biên giới theo BF16, FP8 và NVFP4 và lý do về nguồn tiết kiệm đến từ đâu.
- Kể tên các khai thác TRT-LLM features dành riêng cho Blackwell (ngày 0 FP4, MTP, phân phối phân tách, tất cả primitives).
- Quyết định khi nào khóa NVIDIA của TRT-LLM xứng đáng với khoảng cách chi phí gấp 7 lần so với vLLM trên Hopper.

## Vấn đề

Ranh giới của kinh tế inference vào năm 2026 là "bao nhiêu tokens mỗi đô la". Câu trả lời phụ thuộc vào bốn lựa chọn xếp chồng lên nhau: thế hệ phần cứng (Hopper H100/H200 so với Blackwell B200/GB200), precision (BF16 → FP8 → NVFP4), công cụ phục vụ (vLLM so với SGLang so với TRT-LLM) và orchestration (đơn giản so với phân tách so với Dynamo).

Trên Hopper với vLLM, một MoE 120B chạy ở mức ~ $0.09 per million tokens. On Blackwell with TRT-LLM + Dynamo, the same model runs at ~$0.012 - rẻ hơn 7 lần. Một số khoảng cách đó là phần cứng (Blackwell là thông lượng 11-15 lần mỗi GPU LLM so với Hopper). Một số là stack: trọng lượng FP4, bản nháp MTP, prefill/decode phân tách và NVLink 5 tất cả để giao tiếp chuyên gia MoE.

Bạn không thể tái tạo điều này bên ngoài stack của NVIDIA. Đó là sự đánh đổi - tính di động cho kinh tế. Hiểu được lựa chọn stack nào mang lại phần chênh lệch nào là điểm của bài học này.

## Khái niệm

### Tại sao FP8 vẫn là sàn cho KV cache

Một sai lầm phổ biến vào năm 2026: giả sử NVFP4 áp dụng ở mọi nơi. KV cache cần FP8 (dấu phẩy động 8 bit) vì nó lưu trữ các khóa và giá trị attention span dải động rộng. Lượng tử hóa KV thành FP4 gây ra accuracy loss thảm khốc - đuôi phân phối giảm xuống và điểm số attention sụp đổ. Các bit số mũ của FP8 cung cấp cho KV cache phạm vi cần thiết.

NVFP4 (2025-2026) áp dụng cho trọng số và kích hoạt. Vi tỷ lệ: mỗi khối trọng lượng có hệ số tỷ lệ riêng để các khối nhỏ có thể span các dải động khác nhau mà không cần loss tỷ lệ trên mỗi tensor. Đối với kích hoạt, FP4 giữ được vì kích hoạt có phạm vi nhỏ trong một lớp.

Các config điển hình của Blackwell:

- Trọng lượng: NVFP4 (vi mô 4 bit).
- Kích hoạt: NVFP4.
- KV cache: FP8.
- Attention tích lũy: FP32 (softmax ổn định).

### primitives TRT-LLM dành riêng cho Blackwell sử dụng

- **Quả cân FP0 ngày 4**: Nhà cung cấp model ship quả cân FP4 trực tiếp; TRT-LLM tải mà không cần chuyển đổi sau training. Không có bước AWQ / GPTQ cho FP4.
- **Dự đoán đa token (MTP)**: ý tưởng tương tự như EAGLE (Giai đoạn 17 · 05) nhưng được tích hợp vào bản dựng TRT-LLM.
- **Phân phối phân loại**: điền trước và giải mã trên các nhóm GPU riêng biệt, KV cache chuyển qua NVLink hoặc InfiniBand. Ý tưởng tương tự như Dynamo (Giai đoạn 17 · 20).
- **primitives giao tiếp toàn diện**: NVLink 5 giảm độ trễ giao tiếp chuyên nghiệp MoE gấp 3 lần so với Hopper. Các hạt nhân MoE của TRT-LLM được điều chỉnh cho điều này.
- **NVFP4 + MXFP8 vi tỷ lệ**: xử lý hệ số tỷ lệ được tăng tốc phần cứng trên Blackwell Tensor Cores.

### Những con số bạn nên ghi nhớ

- HGX B200 với giá $0.02/M tokens trên GPT-OSS-120B qua TRT-LLM.
- GB200 NVL72 với giá $0.012/M tokens thông qua Dynamo (điều phối TRT-LLM).
- H100 + vLLM ≈ $0.09/M tokens trên khối lượng công việc tương đương.
- Tăng thông lượng gấp 2,8 lần trong ba tháng cập nhật TRT-LLM (2026).
- Thông lượng 11-15 lần mỗi GPU LLM, Blackwell vs Hopper.
- MLPerf Inference v6.0 (Tháng 4 năm 2026): Blackwell thống trị mọi nhiệm vụ được gửi.

### FP4 thực sự có giá bao nhiêu về chất lượng

NVFP4 rất hung hãn. Trên khối lượng công việc nặng về lý luận (chain-of-thought, toán học, tạo mã với ngữ cảnh dài), trọng số FP4 giảm rõ rệt. Hiệu chuẩn mỗi khối giảm thiểu nhưng không loại bỏ. Các đội shipping lý luận models thường sử dụng trọng số FP8 + kích hoạt FP4 như một sự thỏa hiệp hoặc gắn bó với H200 với FP8 trong suốt.

Quy tắc: luôn xác nhận chất lượng tác vụ trên bộ đánh giá của bạn trước khi cam kết với trọng số NVFP4.

### Tại sao đây là quyết định NVIDIA khóa

TRT-LLM là C++ + CUDA + hạt nhân mã nguồn đóng. Models cần được biên dịch cho một SKU GPU cụ thể. Không AMD, không Intel, không ARM. Nếu chiến lược cơ sở hạ tầng của bạn là nhiều nhà cung cấp, TRT-LLM không phải là người bắt đầu cho cấp TRT-LLM phục vụ — bạn vẫn có thể phân phối từ vLLM trên phần cứng hỗn hợp. Nếu bạn chỉ NVIDIA, khoảng cách 7x sẽ trả cho khóa.

### Công thức thực tế năm 2026

Đối với hóa đơn inference hàng năm 100 triệu đô la +, chạy trên Hopper + vLLM để lại 7-10 lần trên bàn. Di chuyển khối lượng công việc chiếm ưu thế về chi phí sang Blackwell + TRT-LLM + Dynamo. Giữ bậc thử nghiệm trên H100 + vLLM để có tốc độ lặp lại model. Xác nhận chất lượng trên mỗi model chuyển đổi NVFP4 trước khi production.

### Phần thưởng phân tách

Phân tích phân tách của TRT-LLM (các nhóm điền trước và giải mã riêng biệt) được đề cập sâu trong Giai đoạn 17 · 20. Trên Blackwell, hệ số nhân stacks: Trọng số FP4 × tăng tốc MTP × vị trí phân tách × định tuyến nhận biết bộ nhớ cache. Số 7x giả định stack đầy đủ này.

```figure
pipeline-parallel
```

## Ứng dụng

`code/main.py` tính toán dấu chân HBM, giải mã thông lượng (chế độ giới hạn bộ nhớ) và $/M-tokens cho một model trên ba stacks: H100 + BF16 + vLLM, H100 + FP8 + vLLM, B200 + NVFP4/FP8 + TRT-LLM. Chạy nó để xem hiệu ứng kép và phần chênh lệch mà mỗi thay đổi đóng góp.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-trtllm-blackwell-advisor.md`. Với khối lượng công việc, kích thước model và token volume hàng năm, nó quyết định liệu Blackwell + TRT-LLM stack có đáng để khóa NVIDIA hay không.

## Bài tập

1. Chạy `code/main.py`. Trên MoE 120B với 30% parameters hoạt động, hãy tính toán thông lượng giải mã giới hạn băng thông bộ nhớ trên H100 BF16, H100 FP8 và B200 NVFP4/FP8. Bước nhảy vọt lớn nhất đến từ đâu?
2. Một khách hàng chi tiêu $2M/year trên H100 + vLLM. Số hòa vốn của Blackwell là bao nhiêu GPUs họ cần mua để khấu hao chuyển sang TRT-LLM trong 12 tháng, với khoảng cách kinh tế gấp 7 lần?
3. Bạn thấy accuracy giảm 3 điểm trên MATH sau khi chuyển đổi trọng số NVFP4. Đặt tên cho hai đường dẫn khôi phục: một đường dẫn ưu tiên chất lượng (giữ trọng số FP8), một đường dẫn ưu tiên chi phí (hiệu chỉnh với dữ liệu trong miền).
4. Đọc kết quả inference MLPerf v6.0. Nhiệm vụ nào có khoảng cách Blackwell-over-Hopper nhỏ nhất và tại sao?
5. Tính toán HBM cần thiết cho model 405B ở trọng số NVFP4 + KV cache FP8 ở ngữ cảnh 128k. Nó có phù hợp với một nút GB200 NVL72 duy nhất không?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| FP8 | "float tám bit" | dấu phẩy động 8 bit; được sử dụng cho KV cache và attention do dải động |
| NVFP4 | "micro bốn bit" | Định dạng FP vi mô 4 bit của NVIDIA; trọng lượng và kích hoạt trên Blackwell |
| MXFP8 | "MX tám" | Biến thể FP8 vi tỷ lệ; tăng tốc phần cứng trên Blackwell Tensor Cores |
| Ngày-0 FP4 | "ship trọng lượng FP4" | Các nhà cung cấp Model phát hành trọng số đã có trong FP4; Không có bước chuyển đổi sau tàu |
| MTP | "Dự đoán nhiều token" | Dự thảo giải mã suy đoán tích hợp của TRT-LLM (Giai đoạn 17 · 05) |
| Khẩu phần ăn tách rời | "tách prefill/decode" | Điền trước và giải mã trên các nhóm GPU riêng biệt; KV được chuyển qua NVLink/IB |
| Tất cả cho tất cả | "MoE chuyên gia truyền thông" | Định tuyến mẫu giao tiếp tokens đến GPUs chuyên gia; NVLink 5 cắt 3x |
| Suy luậnX | "Bán phân tích inference băng ghế" | Chi phí mỗi token benchmark được ngành công nghiệp chấp nhận năm 2026 |

## Đọc thêm

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) - Kết quả MLPerf tháng 4 năm 2026.
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 hạt nhân tất cả và MoE.
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) - tài liệu chính thức về động cơ.
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — phân tách orchestration trên TRT-LLM.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) - bộ công bố benchmark số Blackwell.
