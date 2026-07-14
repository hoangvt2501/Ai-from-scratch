# Giới thiệu về JAX

> PyTorch đột biến tensors. TensorFlow xây dựng đồ thị. JAX biên dịch các hàm thuần túy. Điều cuối cùng đó thay đổi cách bạn nghĩ về deep learning.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 03 Bài 01-10, NumPy cơ bản
**Thời lượng:** ~90 phút

## Mục tiêu học tập

- Viết mã mạng nơ-ron chức năng thuần túy bằng cách sử dụng API chức năng (jax) của JAX. numpy, jax.grad, jax.jit, jax.vmap)
- Giải thích sự khác biệt thiết kế chính giữa đột biến háo hức của PyTorch và model biên dịch chức năng của JAX
- Áp dụng biên dịch jit và vectơ hóa vmap để tăng tốc vòng lặp training so với Python ngây thơ
- Huấn luyện một mạng đơn giản trong JAX và đối chiếu quản lý trạng thái rõ ràng với cách tiếp cận hướng đối tượng của PyTorch

## Vấn đề

Bạn biết cách xây dựng mạng nơ-ron trong PyTorch. Bạn xác định một `nn.Module`, gọi `.backward()`, bước optimizer. Nó hoạt động. Hàng triệu người sử dụng nó.

Nhưng PyTorch có một ràng buộc ăn sâu vào DNA của mình: nó traces hoạt động một cách háo hức, từng người một, trong Python. Mỗi `tensor + tensor` là một lần khởi chạy hạt nhân riêng biệt. Mỗi bước training diễn giải lại cùng một mã Python. Điều này hoạt động tốt cho đến khi bạn cần huấn luyện 540 tỷ parameter model trên 2.048 TPUs. Sau đó, cái trên cao giết chết bạn.

Google DeepMind huấn luyện Gemini trên JAX. Anthropic huấn luyện Claude về JAX. Đây không phải là những hoạt động nhỏ - chúng là mạng nơ-ron lớn nhất training chạy trên Trái đất. Họ chọn JAX vì nó coi vòng lặp training của bạn như một chương trình có thể biên dịch, không phải là một chuỗi các cuộc gọi Python.

JAX NumPy có ba siêu năng lực: vi phân tự động, biên dịch JIT sang XLA và vector hóa tự động. Bạn viết một hàm processes một ví dụ. JAX cung cấp cho bạn một chức năng processes một batch, tính toán gradients, biên dịch thành mã máy và chạy trên nhiều thiết bị. Tất cả mà không thay đổi chức năng ban đầu.

## Khái niệm

### Triết lý JAX

JAX là một framework chức năng. Không có classes, không có trạng thái thay đổi, không có phương thức `.backward()`. Thay vào đó:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class với trạng thái | Chức năng thuần túy: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Háo hức thực hiện | Biên dịch JIT qua XLA |
| `for x in batch:` vòng lặp thủ công | `jax.vmap(f)` tự động vectơ hóa |
| `DataParallel` / `FSDP` | `jax.pmap(f)` tự động song song |
| `model.parameters()` có thể thay đổi | Pytree bất biến của mảng |

Đây không phải là một sở thích phong cách. Nó là một ràng buộc trình biên dịch. Biên dịch JIT yêu cầu các hàm thuần túy - cùng một đầu vào luôn tạo ra cùng một đầu ra, không có tác dụng phụ. Hạn chế đó là điều giúp tăng tốc gấp 100 lần.

### jax. numpy: Bề mặt quen thuộc

JAX triển khai lại NumPy API trên accelerator:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Cùng tên hàm. Quy tắc phát sóng tương tự. Ngữ nghĩa cắt lát giống nhau. Nhưng các mảng tồn tại trên GPU/TPU và mọi hoạt động đều có thể theo dõi được bởi trình biên dịch.

Một điểm khác biệt quan trọng: JAX mảng là bất biến. Không `a[0] = 5`. Thay vào đó: `a = a.at[0].set(5)`. Điều này khiến bạn cảm thấy khó xử trong một tuần, sau đó nhấp chuột -- tính bất biến là thứ làm cho các phép biến đổi như `grad`, `jit` và `vmap` có thể kết hợp.

### jax.grad: Autodiff chức năng

PyTorch gắn gradients vào tensors (`.grad`). JAX gắn gradients vào các chức năng.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` nhận một hàm và trả về một hàm mới tính toán gradient. Không có cuộc gọi `.backward()`. Không có biểu đồ tính toán được lưu trữ trên tensors. gradient chỉ là một hàm khác mà bạn có thể gọi, soạn hoặc biên dịch JIT.

Điều này soạn tùy ý:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Đạo hàm thứ hai. Dẫn xuất thứ ba. Người Jacobians. Người Hessian. Tất cả bằng cách sáng tác `grad`. PyTorch cũng có thể làm điều này (`torch.autograd.functional.hessian`), nhưng nó được bắt vít. Trong JAX, nó là nền tảng.

Hạn chế: `grad` chỉ hoạt động trên các chức năng thuần túy. Không có câu lệnh in bên trong (chúng chạy trong quá trình theo dõi, không phải thực thi). Không có đột biến của trạng thái bên ngoài. Không tạo số ngẫu nhiên mà không có quản lý khóa rõ ràng.

### jit: Biên dịch thành XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

Trong lần gọi đầu tiên, JAX traces hàm - nó ghi lại những hoạt động nào xảy ra, mà không thực hiện chúng. Sau đó, nó giao trace đó cho XLA (Accelerated Linear Algebra), trình biên dịch của Google cho TPUs và GPUs. XLA hợp nhất các hoạt động, loại bỏ các bản sao bộ nhớ dư thừa và tạo mã máy được tối ưu hóa.

Các cuộc gọi tiếp theo bỏ qua hoàn toàn Python. Mã đã biên dịch chạy trên bộ tăng tốc ở tốc độ C++.

Khi JIT giúp:
- Training bước (cùng một phép tính lặp lại hàng nghìn lần)
- Inference (cùng model, đầu vào khác nhau)
- Bất kỳ hàm nào được gọi nhiều lần với đầu vào có hình dạng tương tự

Khi JIT đau:
- Các hàm có luồng điều khiển Python phụ thuộc vào các giá trị (`if x > 0` trong đó x là mảng được theo dõi)
- One-shot tính toán (chi phí biên dịch vượt quá runtime)
- Gỡ lỗi (theo dõi ẩn quá trình thực thi thực tế)

Hạn chế luồng điều khiển là có thật. `jax.lax.cond` thay thế `if/else`. `jax.lax.scan` thay thế các vòng lặp `for`. Đây không phải là tùy chọn - chúng là cái giá của việc biên soạn.

### vmap: Vector hóa tự động

Bạn viết một hàm processes một ví dụ:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` nâng nó lên process một batch:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` có nghĩa là: không batch quá `params` (dùng chung), batch trên trục 0 của `x`. Không có vòng lặp `for` thủ công. Không định hình lại. Không có luồng kích thước batch. JAX tìm ra chiều batch và vectơ hóa toàn bộ tính toán.

Đây không phải là đường cú pháp. `vmap` tạo mã vector hợp nhất chạy nhanh hơn 10-100 lần so với vòng lặp Python. Và nó sáng tác với `jit` và `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Mỗi ví dụ gradients. Một dòng. Điều này gần như không thể xảy ra ở PyTorch nếu không có hack.

### pmap: Dữ liệu song song trên các thiết bị

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` sao chép chức năng trên tất cả các thiết bị có sẵn (GPUs/TPUs) và chia nhỏ batch. Bên trong chức năng, `jax.lax.pmean` và `jax.lax.psum` đồng bộ hóa gradients trên các thiết bị.

Google huấn luyện Gemini trên hàng nghìn chip TPU v5e bằng cách sử dụng `pmap` (và `shard_map` kế nhiệm của nó). Lập trình model: viết phiên bản một thiết bị, bọc bằng `pmap`, xong.

### Pytrees: Cấu trúc dữ liệu phổ quát

JAX hoạt động trên "pytrees" - sự kết hợp lồng nhau của danh sách, bộ dữ liệu, dict và mảng. model parameters của bạn là một pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Mọi biến đổi JAX - `grad`, `jit`, `vmap` - đều biết cách đi qua pytrees. `jax.tree.map(f, tree)` áp dụng `f` cho mọi lá. Đây là cách optimizers cập nhật tất cả parameters cùng một lúc:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Không có phương pháp `.parameters()`. Không parameter đăng ký. Cấu trúc cây là model.

### Chức năng so với hướng đối tượng

PyTorch lưu trữ trạng thái bên trong các đối tượng:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX sử dụng các hàm thuần túy với trạng thái rõ ràng:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Các tham số được truyền vào. Không có gì được lưu trữ. Không có gì bị đột biến. Điều này làm cho mọi hàm đều có thể kiểm thử, kết hợp và biên dịch. Nó cũng có nghĩa là bạn tự quản lý các tham số - hoặc sử dụng một thư viện như Flax hoặc Equinox.

### Hệ sinh thái JAX

JAX cung cấp cho bạn primitives. Thư viện cung cấp cho bạn công thái học:

| Thư viện | Vai trò | Phong cách |
|---------|------|-------|
| **Hạt lanh** (Google) | Các lớp mạng nơ-ron | `nn.Module` với trạng thái rõ ràng |
| **Xuân phân **(Patrick Kidger) | Các lớp mạng nơ-ron | Dựa trên Pytree, Pythonic |
| **Optax** (DeepMind) | Lịch trình Optimizers + LR | Biến đổi gradient có khả năng kết hợp |
| **Orbax** (Google) | Trạm kiểm tra | Save/restore pytrees |
| **CLU** (Google) | Số liệu + ghi nhật ký | Tiện ích vòng lặp Training |

Optax là thư viện optimizer tiêu chuẩn. Nó tách chuyển đổi gradient (Adam, SGD, cắt) khỏi bản cập nhật parameter, khiến việc soạn thảo trở nên tầm thường:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### Khi nào nên sử dụng JAX so với PyTorch

| Yếu tố | JAX | PyTorch |
|--------|-----|---------|
| Hỗ trợ TPU | First-class (Google xây dựng cả hai) | Cộng đồng duy trì (torch_xla) |
| Hỗ trợ GPU | Tốt (CUDA qua XLA) | Tốt nhất trong class (CUDA gốc) |
| Gỡ lỗi | Cứng (theo dõi + biên soạn) | Dễ dàng (háo hức, từng dòng) |
| Hệ sinh thái | Tập trung vào nghiên cứu (Lax, Equinox) | Đồ sộ (HuggingFace, torchvision, v.v.) |
| Tuyển dụng | Ngách (Google/DeepMind/Anthropic) | Mainstream (ở khắp mọi nơi) |
| training quy mô lớn | Cao cấp (XLA, pmap, lưới) | Tốt (FSDP, DeepSpeed) |
| Tốc độ tạo mẫu | Chậm hơn (chi phí chức năng) | Nhanh hơn (đột biến và đi) |
| Production inference | Phục vụ TensorFlow, Vertex AI | Ngọn đuốcPhục vụ, Triton, ONNX |
| Ai sử dụng nó | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Độ ổn định AI |

Câu trả lời trung thực: sử dụng PyTorch trừ khi bạn có lý do cụ thể để sử dụng JAX. Những lý do đó là -- quyền truy cập TPU, nhu cầu gradients cho mỗi ví dụ, training đa thiết bị ở quy mô lớn hoặc làm việc ở Google/DeepMind/Anthropic.

### Số ngẫu nhiên trong JAX

JAX không có trạng thái ngẫu nhiên toàn cục. Mọi thao tác ngẫu nhiên đều yêu cầu một khóa PRNG rõ ràng:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

Điều này thật khó chịu lúc đầu. Nhưng nó đảm bảo khả năng tái tạo trên các thiết bị và biên dịch - một thuộc tính mà `torch.manual_seed` của PyTorch không thể đảm bảo trong cài đặt nhiều GPU.

```figure
batchnorm-effect
```

## Tự xây dựng

### Bước 1: Thiết lập và dữ liệu

Chúng ta sẽ huấn luyện MLP 3 lớp trên MNIST bằng cách sử dụng JAX và Optax. 784 đầu vào, hai lớp ẩn gồm 256 và 128 tế bào thần kinh, 10 đầu ra classes.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Bước 2: Khởi tạo Parameters

Không class. Chỉ là một hàm trả về một pytree:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

Anh khởi tạo, được thực hiện thủ công. Ba phím PRNG được tách ra từ một hạt giống. Mỗi trọng lượng là một mảng bất biến trong một dict lồng nhau.

### Bước 3: Forward Pass

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Chức năng thuần túy. Tham số vào, dự đoán ra. Không có `self`, không có trạng thái được lưu trữ. `loss_fn` tính toán entropy chéo từ đầu - softmax, log, trung bình âm.

### Bước 4: Bước Training biên dịch JIT

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` trả về cả giá trị loss và gradients trong một lần chuyển tiếp. Trình trang trí `@jax.jit` biên dịch cả hai hàm thành XLA. Sau cuộc gọi đầu tiên, mỗi bước training chạy mà không cần chạm vào Python.

### Bước 5: Vòng lặp Training

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 epochs. ~97% kiểm tra accuracy. epoch đầu tiên chậm (biên dịch JIT). Epochs 2-10 là nhanh.

Chú ý những gì còn thiếu: không có `.zero_grad()`, không có `.backward()`, không có `.step()`. Toàn bộ bản cập nhật là một lệnh gọi hàm được soạn thảo. Gradients được tính toán, biến đổi bởi Adam và áp dụng cho parameters - tất cả đều bên trong `train_step`.

## Ứng dụng

### Lanh: Tiêu chuẩn của Google

Flax là thư viện mạng nơ-ron JAX phổ biến nhất. Nó thêm `nn.Module` trở lại, nhưng với quản lý trạng thái rõ ràng:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

Cấu trúc tương tự như PyTorch, nhưng `params` tách biệt với model. `model.init()` tạo các tham số. `model.apply(params, x)` điều hành forward pass. Đối tượng model không có trạng thái.

### Equinox: Giải pháp thay thế Pythonic

Equinox (của Patrick Kidger) đại diện cho models dưới dạng pytrees:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Bản thân model là một cây pytree. Không cần `.apply()`. Parameters chỉ là những chiếc lá của model. Điều này gần với cách JAX suy nghĩ.

### Optax: Optimizers có khả năng kết hợp

Optax tách chuyển đổi gradient khỏi bản cập nhật:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient cắt, khởi động learning rate, giảm cân - tất cả đều được tạo thành một chuỗi biến đổi. Mỗi biến đổi nhìn thấy gradients, sửa đổi chúng và chuyển chúng cho người tiếp theo. Không có optimizer class nguyên khối.

## Sản phẩm bàn giao

**Cài đặt:**

```bash
pip install jax jaxlib optax flax
```

Để được hỗ trợ GPU:

```bash
pip install jax[cuda12]
```

Đối với TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Hiệu suất:**

- Lệnh gọi JIT đầu tiên chậm (biên dịch). Khởi động trước khi đo điểm chuẩn.
- Tránh Python vòng lặp trên JAX mảng bên trong JIT. Sử dụng `jax.lax.scan` hoặc `jax.lax.fori_loop`.
- `jax.debug.print()` hoạt động bên trong JIT. `print()` thông thường thì không.
- Hồ sơ với `jax.profiler` hoặc TensorBoard. Biên dịch XLA có thể che giấu các nút thắt cổ chai.
- JAX phân bổ trước 75% bộ nhớ GPU theo mặc định. Đặt `XLA_PYTHON_CLIENT_PREALLOCATE=false` để tắt.

**Điểm kiểm tra:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**Bài học này tạo ra:**
- `outputs/prompt-jax-optimizer.md` -- một prompt để chọn đúng JAX optimizer configuration
- `outputs/skill-jax-patterns.md` -- một skill bao gồm các mẫu chức năng trong JAX

## Bài tập

1. Thêm dropout vào MLP. Trong JAX, dropout yêu cầu một khóa PRNG - thread một khóa thông qua forward pass và chia nó cho mỗi lớp dropout. So sánh accuracy thử nghiệm có và không.

2. Sử dụng `jax.vmap` để tính toán mỗi ví dụ gradients cho batch 32 hình ảnh MNIST. Tính định mức gradient cho từng ví dụ. Ví dụ nào có số gradients lớn nhất và tại sao?

3. Thay thế chức năng chuyển tiếp thủ công bằng một `mlp_forward(params, x)` chung hoạt động cho bất kỳ số lượng lớp nào. Sử dụng `jax.tree.leaves` để tự động xác định độ sâu.

4. Benchmark bước training có và không có `@jax.jit`. Thời gian 100 bước mỗi bước. Tăng tốc trên phần cứng của bạn lớn như thế nào? Chi phí biên dịch trong cuộc gọi đầu tiên là bao nhiêu?

5. Triển khai cắt gradient bằng cách soạn `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Huấn luyện có và không có cắt. Vẽ định mức gradient trên training để xem hiệu quả.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|----------------------|
| XLA | "Điều khiến JAX nhanh chóng" | Accelerated Linear Algebra -- một trình biên dịch hợp nhất các hoạt động và tạo ra các hạt nhân GPU/TPU được tối ưu hóa từ biểu đồ tính toán |
| JIT | "Biên soạn đúng lúc" | JAX traces hàm trong lần gọi đầu tiên, hãy biên dịch thành XLA, sau đó chạy phiên bản đã biên dịch trong các lệnh gọi tiếp theo |
| Chức năng thuần túy | "Không có tác dụng phụ" | Một hàm trong đó đầu ra chỉ phụ thuộc vào đầu vào - không có trạng thái toàn cục, không có đột biến, không ngẫu nhiên mà không có các khóa rõ ràng |
| vmap | "Tự động lô" | Chuyển đổi một hàm processes một ví dụ thành một hàm processes batch mà không cần viết lại |
| Bản đồ | "Tự động song song" | Sao chép một chức năng trên nhiều thiết bị và chia batch đầu vào |
| Pytree | "Dict lồng nhau của mảng" | Bất kỳ cấu trúc lồng nhau nào của danh sách, bộ dữ liệu, dict và mảng mà JAX có thể duyệt qua và chuyển đổi |
| Truy tìm | "Ghi lại tính toán" | JAX thực thi hàm với các giá trị trừu tượng để xây dựng biểu đồ tính toán mà không tính toán kết quả thực |
| Chức năng tự động | "grad của một hàm" | Tính toán đạo hàm bằng cách chuyển đổi các hàm, không phải bằng cách đính kèm gradient lưu trữ vào tensors |
| Thuế Optax | "Thư viện optimizer của JAX" | Một thư viện có khả năng kết hợp gồm các phép biến đổi gradient -- Adam, SGD, cắt, lập lịch -- xâu chuỗi với nhau |
| Lanh | "JAX là nn. Mô-đun" | Thư viện mạng nơ-ron của Google dành cho JAX, thêm các trừu tượng lớp trong khi vẫn giữ trạng thái rõ ràng |

## Đọc thêm

- Tài liệu JAX: https://jax.readthedocs.io/ -- Tài liệu chính thức, với các hướng dẫn tuyệt vời về Grad, JIT và VMAP
- "JAX: các phép biến đổi có thể kết hợp của Python+NumPy chương trình" (Bradbury et al., 2018) - bài báo gốc giải thích triết lý thiết kế
- Tài liệu về lanh: https://flax.readthedocs.io/ -- Thư viện mạng nơ-ron của Google dành cho JAX
- Patrick Kidger, "Equinox: mạng nơ-ron trong JAX thông qua PyTrees có thể gọi được và các phép biến đổi được lọc" (2021) - giải pháp thay thế Pythonic cho Flax
- DeepMind, "Optax: chuyển đổi và tối ưu hóa gradient có thể kết hợp" -- thư viện optimizer tiêu chuẩn
- "Bạn không biết JAX" (Colin Raffel, 2020) - một hướng dẫn thực tế về các JAX và mô hình, từ một trong những tác giả T5
