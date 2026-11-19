# Thí nghiệm: vẽ lại các hình của `evaluation.tex` bằng simulator

Ghi ngày 2026-11-19. Mục tiêu là chạy lại đúng **bộ hình** trong section "Multi-state Serverless Evaluation"
(`sweep/evaluation.tex`) bằng simulator với profile đo mới. Không nhằm khớp số với paper: paper dùng mô hình Markov
giải tích, profile cũ và 60 container.

**Trạng thái:** lần chạy 10 seed với cấu hình dưới đây đã bị dừng giữa chừng, chưa có kết quả.
`sweep/results/results.csv` hiện rỗng. Chạy lại theo mục "Cách chạy".

## 1. Cần mang sang máy khác

| Thứ | Vai trò |
|---|---|
| `core/` (`config.py`, `platform.py`, `theory.py`) | simulator; `tools/sweep.py` kế thừa `Theory` |
| `data/profiles/gemma.json`, `data/profiles/yolo.json` | profile đo: 100 mẫu mỗi pha + RAM theo pha |
| `tools/sweep.py` | chạy mô phỏng, ghi `results.csv` |
| `tools/plot.py` | vẽ hình từ `results.csv` |
| `requirements.txt` | `simpy==4.1.2`, `matplotlib==3.10.9` (gymnasium, numpy không cần cho thí nghiệm này) |

Python 3.10.12. Chạy mọi lệnh từ thư mục `simulator/` (các tool import `core.*` và `tools.*`).

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## 2. Mô hình mô phỏng

Lớp `Markov(Theory)` trong `tools/sweep.py`. Một ứng dụng trên một hệ thống, vòng đời 4 trạng thái của paper:

```
Null --pod start--> WC (warm container) --init--> WM (warm model) <--> Active
        WM --theta4--> WC --theta3--> Null        (chỉ khi container rỗi)
```

- **Request đến:** Poisson, tốc độ λ (req/s).
- **Chọn container:** WM rỗi trước, rồi WC rỗi, rồi tạo mới nếu còn chỗ; hết chỗ thì request bị **block** (mất, không xếp hàng).
- **Dung lượng:** 15 container (testbed). Paper dùng 60.
- **Thời gian chuyển trạng thái và thời gian chạy:** rút từ **log-normal** fit maximum likelihood trên 100 mẫu của profile,
  riêng cho từng pha: `pod_start`, `init_prewarm` (WC→WM khi lấy từ WC rỗi), `init_cold` (WC→WM ngay sau cold start), `run`.
- **Lùi trạng thái:** mỗi container rỗi có timer **phân phối mũ**: WM→WC với tốc độ θ₄, WC→Null với tốc độ θ₃.
  θ = ∞ nghĩa là lùi ngay lập tức. Lùi là tức thời (không tốn thời gian).
- **RAM theo pha** (MB, working set, lấy từ `ram_mb` của profile): `start` (Null→WC), `stem` (WC rỗi), `init` (WC→WM),
  `ready` (WM rỗi), `run` (Active). Container lùi về WC thì RAM về mức `stem`.

Tham số log-normal (μ, σ của log giây) để đối chiếu sau khi fit trên máy mới:

| Pha | gemma (LLM) | yolo (OD) |
|---|---|---|
| pod_start | 0.385, 0.340 | 0.388, 0.343 |
| init_prewarm | 0.420, 0.128 | 0.725, 0.245 |
| init_cold | 0.423, 0.138 | 0.930, 0.312 |
| run | 3.552, 0.331 | −0.807, 0.262 |

Trung bình mẫu (s): gemma pod start 1.56, init 1.53, init cold 1.54, run 36.53; yolo 1.56, 2.13, 2.66, 0.46.
RAM (MB) start/stem/init/run/ready: gemma 15.9/31.2/531.8/1222.7/1222.8; yolo 70.2/149.1/620.0/803.2/802.8.

## 3. Test case (Table "Evaluation test cases" của paper)

| Case | θ₃ | θ₄ |
|---|---|---|
| 2-state ON | 0.001 | 0.001 |
| 2-state OFF | ∞ | ∞ |
| 3-state WC | quét | ∞ |
| 3-state WM | ∞ | quét |
| 4-state | quét | quét |

- **Dải quét θ:** 0.001 → 100 /s, cách đều theo log. Paper dùng 100 giá trị đều trong [0.01, 1]; dải đó không phủ
  được vùng hoạt động (θ tốt nhất hay chạm mép 1.0).
- **Số điểm:** 41 giá trị cho mỗi case 3-state (`--n3 41`); lưới 11 × 11 cho 4-state (`--n4 11`).
- **Tên ứng dụng:** giữ tên của paper để thay thẳng vào tex: `llm_2b` = profile gemma, `yolo_x` = profile yolo.
- **Arrival rate:** rate của paper chia 4 (hệ thống nhỏ hơn 4 lần, giữ cùng tải trên mỗi container):
  `yolo_x` λ = 2⁻² … 2⁶, `llm_2b` λ = 2⁻⁸ … 2⁰ (9 giá trị mỗi app).

Tổng: 2 app × 9 rate × 205 cấu hình θ × 10 seed = 36 900 run.

## 4. Độ dài run và metric

- **Độ dài đo:** `max(requests / λ, turnovers × T)`, với `requests = 10000`, `turnovers = 10`,
  `T = max(1/θ₃, 1/θ₄, thời gian run trung bình)` (bỏ qua θ = ∞).
- **Khởi động bị loại:** `max(10 % độ dài đo, 3 × T)` trước khi bắt đầu đo; hệ thống bắt đầu rỗng.
- **Seed:** 1–10. Dãy request đến dùng `Random(seed × 1000 + 1)`, giống nhau ở mọi case cùng seed.
- **Metric mỗi run** (cột trong `results.csv`):
  - `block_prob` = request bị block / request đến.
  - `latency_s` = startup latency trung bình trên request được phục vụ (0 nếu lấy WM; init nếu lấy WC; pod start + init cold nếu cold).
  - `ram_gbs_per_req` = tổng RAM × thời gian (GB·s, MB/1024) của mọi pha, **kể cả lúc inference**, chia cho số request được phục vụ.
  - `cold`, `prewarm`, `warm` = số request theo loại khởi động.

## 5. Cách chạy

```
.venv/bin/python -m tools.sweep --seeds 1,2,3,4,5,6,7,8,9,10     # ghi sweep/results/{results.csv,config.json}
.venv/bin/python -m tools.plot                                    # ghi sweep/results/figures/ và best_ram_<app>.csv
```

Các giá trị ở mục 3 và 4 là mặc định của `tools/sweep.py`; chỉ `--seeds` phải truyền (mặc định là 1 seed).
`--jobs` mặc định bằng số nhân CPU; `--out` đổi thư mục kết quả. `sweep.py` ghi đè `results.csv` mỗi lần chạy.

Thời gian: ước tính 1,5–2 giờ trên 4 nhân (chưa đo; lần chạy bị dừng trước khi xong).

## 6. Hình tạo ra

`tools/plot.py` lấy trung bình qua seed cho từng (app, case, λ, θ₃, θ₄) rồi vẽ, mỗi hình một file `.pdf` và `.png`,
trong `sweep/results/figures/`, cùng tên thư mục như `evaluation.tex`:

| Hình paper | File | Cách vẽ |
|---|---|---|
| KPI theo θ₄ (Fig. 3d-app) | `3D-yolo_x-wm/{block,latency,ram}` | `yolo_x`, 3-state WM, λ = 0.5, 2, 8 (paper: 2, 8, 32); trục θ₄ thang log; chú thích ghi θ₄ cho giá trị thấp nhất của metric |
| KPI theo λ, OD (Fig. light-app) | `yolo_x-app/{block,latency,ram}` | 5 case; với 3-state và 4-state lấy cấu hình θ có RAM mỗi request thấp nhất tại từng λ |
| KPI theo λ, LLM (Fig. medium-app) | `llm_2b-app/{block,latency,ram}` | như trên |
| Pareto front (Fig. pareto-llm) | `pareto/llm_2b` (thêm `pareto/yolo_x`) | λ = 0.25 (paper: 1); các điểm không bị trội về (RAM, latency); trục latency thang log |

`best_ram_<app>.csv` chứa đúng các điểm vẽ trên hình KPI theo λ, kèm θ₃, θ₄ được chọn.
Hình `evaluated-lifecycle.pdf` là sơ đồ vẽ tay, không thuộc thí nghiệm này.

## 7. Số tham chiếu để kiểm tra nhanh

Từ lần chạy trước (1 seed, 50 000 request mỗi run, lưới θ cũ đều trong [0.01, 1]). Chỉ các điểm ON/OFF và mức bão hoà
dưới đây không phụ thuộc lưới θ, nên dùng được để kiểm tra máy mới chạy đúng mô hình:

| Điểm | block | latency (s) | RAM/request (GB·s) |
|---|---|---|---|
| `yolo_x`, 2-state OFF, λ = 1 | 0.0001 | 4.22 | 2.08 |
| `yolo_x`, 2-state OFF, λ = 64 | 0.950 | 4.23 | 2.09 |
| `yolo_x`, 2-state ON, λ = 0.25 | 0 | 0.025 | 7.61 |
| `yolo_x`, 2-state ON, λ = 64 | 0.521 | 0.000 | 0.38 |
| `llm_2b`, 2-state OFF, λ = 0.25 | 0.038 | 3.09 | 44.7 |
| `llm_2b`, 2-state ON, λ = 0.25 | 0.024 | 0.04 | 68.7 |
| `llm_2b`, 2-state ON, λ = 1 | 0.610 | 0.00 | 45.8 |

## 8. Lưu ý khi diễn giải

- Đường RAM theo θ rất phẳng quanh điểm thấp nhất, nên "θ tiết kiệm RAM nhất" nhạy với nhiễu; đó là lý do chạy 10 seed.
- Lời văn trong `evaluation.tex` (ví dụ "θ₄ khoảng 0.5", "3 s", "0.68 s", "giảm 40 % RAM") viết cho profile và mô hình cũ,
  cần sửa lại theo hình mới.
- `core/theory.py` có sẵn timeout cố định; thí nghiệm này không dùng mà thay bằng timer mũ trong `Markov.on_idle`.
