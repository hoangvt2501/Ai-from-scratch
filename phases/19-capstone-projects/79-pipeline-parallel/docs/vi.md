# Pipeline Phân tích song song và bong bóng

> Tensor song song phân chia ma trận nhân lên giữa các cấp bậc. Pipeline song song chia model giữa các cấp bậc, một giai đoạn cho mỗi cấp bậc. Các vi mẻ chảy qua pipeline. Thời gian trống ở đầu và cuối là bong bóng; giảm thiểu nó là toàn bộ thủ công.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 19 Bài học theo dõi C 42-49
**Thời lượng:** ~90 phút

## Mục tiêu học tập

- Chia một model tuần tự thành N giai đoạn và mô phỏng một pipeline chuyển tiếp qua N cấp bậc.
- Lên lịch M vi mẻ thông qua pipeline bằng cách sử dụng lịch trình GPipe (chỉ điền tiến, sau đó lùi) và tính toán phần bong bóng.
- So sánh bong bóng với lịch trình 1F1B xen kẽ được sử dụng trong Megatron-LM và PipeDream.
- Phân công giai đoạn bảo vệ: tính toán bằng nhau trên mỗi giai đoạn quan trọng hơn số lượng parameter mỗi giai đoạn bằng nhau.

## Vấn đề

Một parameter model 70B trong fp16 chỉ cần 140 GB parameters. Không có GPU tiêu dùng nào nắm giữ nó. Các mảnh vỡ ZeRO-3 parameters qua các cấp bậc nhưng vẫn cần mọi cấp bậc để thu thập toàn bộ lớp cho mỗi bước tiến lên, trả các bước nhảy log(N) cho mỗi lớp. Pipeline song song đi theo một lộ trình khác: cắt model thành N giai đoạn và đặt một giai đoạn trên mỗi cấp bậc. Phía trước của lớp 1 kết thúc ở hạng 0 và trao tensor kích hoạt lên hạng 1; xếp hạng 1 chạy lớp 2 và tay lên hạng 2; và như vậy. Dòng chảy ngược lại. Bộ nhớ giảm tuyến tính vì mỗi cấp chỉ giữ một giai đoạn; tính toán là tuần tự, đó là bài toán bong bóng.

Bong bóng là thời gian nhàn rỗi khi bắt đầu pipeline (chờ vi mẻ đầu tiên đến giai đoạn cuối cùng) và cuối cùng (chờ vi mẻ cuối cùng chảy trở lại). Với M vi mẻ và N giai đoạn, phần bong bóng trên mỗi giai đoạn là (N-1) / (M + N-1). Tại M = 8, N = 4 là 27%. Tại M = 64, N = 4 là 4,5%. Bong bóng co lại khi bạn có nhiều microbatch mỗi bước, có nghĩa là kích thước batch mỗi microbatch nhỏ, đây là ràng buộc thúc đẩy thiết kế microbatch.

## Khái niệm

```mermaid
flowchart LR
  R0[rank 0: stage 0 / layer 0] --> R1[rank 1: stage 1 / layer 1]
  R1 --> R2[rank 2: stage 2 / layer 2]
  R2 --> R3[rank 3: stage 3 / loss]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### Lịch trình GPipe

Đổ đầy pipeline về phía trước với tất cả M vi mẻ trước khi bắt đầu lùi lại; sau đó xả ngược lại. Kích hoạt từ mỗi microbat phải được giữ cho đến khi nó lùi, vì vậy bộ nhớ phát triển tuyến tính với M. Chuyển tiếp lấy chu kỳ M + N-1, lùi mất một chu kỳ M + N-1 khác. Công việc hữu ích trên mỗi giai đoạn là 2 triệu chu kỳ; bong bóng mỗi giai đoạn là 2 (N-1) chu kỳ. Phần bong bóng là (N-1) / (M + N-1) khi mỗi tiến và lùi mất một đơn vị thời gian. Chọn M lớn hơn nhiều so với N sẽ ẩn bong bóng.

### Lịch trình 1F1B

Xen kẽ: ngay sau khi chuyển tiếp của microbatch đến giai đoạn cuối cùng, hãy bắt đầu lùi lại và để nó quay trở lại. Lịch trình luân phiên một tiến và một lùi mỗi giai đoạn. Bong bóng vẫn là N-1 nhưng bộ nhớ kích hoạt bị giới hạn bởi độ sâu pipeline, không phải số lượng vi mẻ. Production pipelines sử dụng 1F1B (Megatron, PipeDream). Bài học triển khai GPipe trước vì nó đơn giản hơn và 1F1B như một bài tập.

### Tại sao điện toán bình đẳng trên mỗi giai đoạn lại quan trọng

Nếu giai đoạn 0 mất 50 mili giây và giai đoạn 1 mất 100 mili giây, thì mọi chu kỳ đều được kiểm soát ở giai đoạn 1. Các giai đoạn khác không hoạt động 50 ms mỗi chu kỳ chờ giai đoạn 1 phát hành. Số lượng parameter bằng nhau là trục sai: tính toán của một transformer bị chi phối bởi attention cộng với MLP trên mỗi lớp và các lớp embedding có nhiều parameters nhưng tính toán ít. Phân công giai đoạn nên cân bằng FLOPs trên mỗi giai đoạn, không phải trọng số trên mỗi giai đoạn.

### Microbatch so với batch

A pipeline chạy M mẻ vi mô cỡ B mỗi mẻ. Kích thước batch hiệu quả là M*B. The gradient at the end of a pipeline step is the gradient on the combined M*B ví dụ. Phần bong bóng phụ thuộc vào M; trình tối ưu hóa nhìn thấy M * B. Điều chỉnh M có nghĩa là bong bóng giao dịch (thấp hơn với M cao) so với bộ nhớ mỗi microbatch (bộ nhớ kích hoạt cao hơn với M cao cho GPipe).

## Tự xây dựng

`code/main.py` thực hiện:

- `PipelineStage`: một `nn.Module` nhỏ giữ parameters của một giai đoạn và để lộ `forward(activation)`.
- `Pipeline(stages, num_microbatches)`: điều phối lịch trình GPipe trên các giai đoạn mô phỏng bằng cách sử dụng đồng hồ treo tường mô phỏng trên mỗi giai đoạn.
- `bubble_fraction(num_stages, num_microbatches)`: dạng đóng (N-1)/(M+N-1).
- Bản demo 4 giai đoạn in trace mỗi lô nhỏ và phần bong bóng đo được.

Chạy nó:

```bash
python3 code/main.py
```

Đầu ra: biểu đồ Gantt từng từng lô nhỏ và tỷ lệ phần trăm bong bóng so với dự đoán dạng đóng.

## Production mô hình trong tự nhiên

Ba mẫu cứng lại pipeline đủ song song để ship.

**Điểm kiểm tra kích hoạt ghép nối với pipeline.** Với M microbatch đang bay trên GPipe, bộ nhớ kích hoạt là M nhân với một microbatch. Điểm kiểm tra kích hoạt tính toán lại chuyển tiếp tại thời gian lùi, trao đổi tính toán cho bộ nhớ; Sự kết hợp là điều làm cho pipeline dễ xử lý cho các chuỗi dài.

**Cân bằng giai đoạn được đo lường chứ không phải giả định.** Các nhóm Production chạy một đường chuyền phân tích đo lường điện toán trên mỗi lớp thực tế (FLOPs và đồng hồ treo tường) trên phần cứng đích, sau đó phân vùng theo phép đo đó. Cờ `--num-layers-per-stage` Megatron-LM chấp nhận một danh sách cho phép số lượng lớp không đồng đều khi các giai đoạn có chi phí trên mỗi lớp khác nhau.

**Lịch trình gửi-thu hồi phải tránh bế tắc.** Một pipeline có mọi stage gửi trước khi nhận được bế tắc trên dây. Cách khắc phục tiêu chuẩn là xen kẽ: các giai đoạn xếp hạng chẵn gửi trước sau đó thu hồi, giai đoạn xếp hạng lẻ thu hồi trước sau đó gửi. Lịch học được xếp hạng rõ ràng để mô hình có thể nhìn thấy.

## Ứng dụng

Production mẫu:

- **Megatron-LM.** Tham chiếu cho pipeline song song trên quy mô lớn. Sử dụng 1F1B và hỗ trợ kết hợp song song tensor + pipeline + dữ liệu.
- **DeepSpeed Pipeline.** Tích hợp với ZeRO; ZeRO-1 + pipeline là một combo phổ biến cho các models mở lớn nhất.
- **PyTorch Pipe.** Trình bao bọc pipeline gốc PyTorch, được xây dựng trên `torch.distributed.pipeline.sync.Pipe`.

## Sản phẩm bàn giao

Bài 80 lưu trữ các phân đoạn parameter trên mỗi giai đoạn trong checkpoint phân đoạn. Bài 81 soạn DDP + ZeRO + pipeline trên bản demo end-to-end (về tinh thần; bản demo giữ pipeline mô phỏng cho runtime).

## Bài tập

1. Triển khai 1F1B và xác minh phân số bong bóng khớp với GPipe nhưng bộ nhớ kích hoạt bị giới hạn.
2. Lập hồ sơ thời gian thực trên mỗi giai đoạn trên model sâu hơn và cân bằng lại các giai đoạn bằng đồng hồ treo tường được đo lường.
3. Thêm gradient tích lũy trên pipeline microbatch và kiểm tra độ gradient bằng gradient của full-batch forward tương đương.
4. Kết hợp pipeline với điểm kiểm tra kích hoạt và đo lường mức giảm bộ nhớ so với chi phí điện toán.
5. Kết hợp pipeline với DDP (mỗi pipeline xếp hạng được sao chép trên một nhóm dữ liệu song song) và suy luận thông qua lịch trình 2D.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Pipeline | "Model song song dọc theo độ sâu" | Một giai đoạn cho mỗi cấp bậc, các kích hoạt chuyển từ giai đoạn này sang giai đoạn khác |
| Bong bóng | "Pipeline thời gian nhàn rỗi" | (N-1) các bước ở đầu + cuối khi một số giai đoạn không có công việc |
| Hạt vi mô | "Lát cắt của batch" | Một đơn vị forward/backward; bong bóng co lại khi M phát triển |
| GPipe | "Đổ đầy rồi để ráo" | Tất cả M tiến trước bất kỳ lùi nào; bộ nhớ kích hoạt cao |
| 1F1B | "Lịch trình xen kẽ" | Một tiến một lùi mỗi giai đoạn; Bộ nhớ kích hoạt có giới hạn |

## Đọc thêm

- [Huang et al, GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Narayanan et al, PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)
- [Megatron-LM pipeline parallel docs](https://github.com/NVIDIA/Megatron-LM)
- Giai đoạn 19 Bài 76 - send/recv primitives lịch trình sử dụng
- Giai đoạn 19 Bài 78 - ZeRO trực giao với pipeline và thường kết hợp
