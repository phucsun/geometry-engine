# GeometryEngine — Tài liệu kỹ thuật

> **Vị trí trong hệ thống:** OCR → LLM → **GeometryEngine** → Unity AR  
> Nhận JSON ràng buộc từ LLM, giải toạ độ 3D, trả về điểm + cạnh + mặt cho Unity render.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Input / Output Schema](#3-input--output-schema)
4. [Các constraint được hỗ trợ](#4-các-constraint-được-hỗ-trợ)
5. [Thuật toán giải](#5-thuật-toán-giải)
6. [Các kỹ thuật toán học](#6-các-kỹ-thuật-toán-học)
7. [Pipeline xử lý sau khi giải](#7-pipeline-xử-lý-sau-khi-giải)
8. [API Server](#8-api-server)
9. [CLI](#9-cli)
10. [Test Suite](#10-test-suite)
11. [Ví dụ end-to-end](#11-ví-dụ-end-to-end)

---

## 1. Tổng quan kiến trúc

```
┌─────────────┐   ảnh đề toán   ┌─────────┐   văn bản   ┌─────────┐
│  Mobile App │ ─────────────▶  │   OCR   │ ──────────▶ │   LLM   │
└─────────────┘                 └─────────┘             └────┬────┘
                                                             │ GeometryInput JSON
                                                             ▼
                                                   ┌─────────────────┐
                                                   │  GeometryEngine │
                                                   │  (Python)       │
                                                   └────────┬────────┘
                                                            │ GeometryOutput JSON
                                                            │ (points + edges + faces)
                                                            ▼
                                                   ┌─────────────────┐
                                                   │   Unity AR App  │
                                                   │ LineRenderer    │
                                                   │ Mesh            │
                                                   └─────────────────┘
```

**GeometryEngine** là một HTTP microservice (FastAPI). Unity gọi `POST /solve`, nhận JSON với toạ độ 3D của mọi điểm, danh sách cạnh và danh sách mặt để render.

---

## 2. Cấu trúc thư mục

```
GeometryEngine/
├── geometry_engine/
│   ├── __init__.py          # Export GeometryEngine, GeometryInput
│   ├── __main__.py          # CLI entry point
│   ├── models.py            # Pydantic I/O schemas
│   ├── engine.py            # Solver chính — constraint propagation
│   ├── topology.py          # Sinh edges + faces từ constraint
│   ├── validator.py         # Kiểm tra toạ độ đã giải có thoả mãn constraint không
│   ├── normalizer.py        # Chuẩn hoá về origin, scale về unit sphere
│   └── utils.py             # Thư viện vector 3D (numpy)
├── server.py                # FastAPI app
├── tests/
│   ├── conftest.py
│   ├── test_engine.py       # 56 tests — core solving
│   ├── test_topology.py     # 18 tests — edges/faces
│   ├── test_server.py       # 11 tests — HTTP API
│   └── test_advanced.py     # 24 tests — constraint types mới
└── requirements.txt
```

### Vai trò từng module

| Module | Trách nhiệm |
|--------|------------|
| `models.py` | Định nghĩa `GeometryInput` / `GeometryOutput` bằng Pydantic v2 |
| `engine.py` | Vòng lặp constraint propagation, sinh toạ độ 3D |
| `topology.py` | Map constraint type → danh sách Edge + Face |
| `validator.py` | Kiểm tra hậu kỳ: mỗi constraint có thoả mãn không |
| `normalizer.py` | Center về gốc toạ độ, scale về unit sphere để Unity AR nhất quán |
| `utils.py` | Tất cả phép tính vector: chuẩn hoá, chiếu, phản chiếu, giao điểm |
| `server.py` | FastAPI wrapper, `POST /solve`, `GET /health` |
| `__main__.py` | CLI: `solve`, `validate`, `serve` |

---

## 3. Input / Output Schema

### GeometryInput

```json
{
  "points": ["A", "B", "C", "D", "S"],
  "constraints": [
    {
      "type": "square",
      "points": ["A", "B", "C", "D"]
    },
    {
      "type": "right_angle",
      "points": ["S", "A", "B"]
    },
    {
      "type": "right_angle",
      "points": ["S", "A", "D"]
    }
  ],
  "side_length": 2.0,
  "normalize": false,
  "validate_constraints": true
}
```

**Tham số của `Constraint`:**

| Trường | Kiểu | Dùng bởi |
|--------|------|----------|
| `type` | `str` | tất cả |
| `points` | `list[str]` | hình học đa giác, right_angle, angle, distance |
| `point` | `str` | điểm kết quả của midpoint, centroid, intersection… |
| `segment` | `list[str]` | cặp điểm [A, B] cho midpoint, foot_perpendicular, intersection |
| `from_point` | `str` | điểm xuất phát cho foot_on_plane, symmetric, perpendicular_to_plane |
| `length` | `float` | chiều dài / kích thước x |
| `width` | `float` | kích thước y (rectangle, prism) |
| `height` | `float` | kích thước z (prism, truncated_pyramid) |
| `ratio` | `float` | tỉ lệ [0,1] cho ratio_point hoặc truncated_pyramid |
| `degrees` | `float` | góc (độ) cho constraint `angle` |

### GeometryOutput

```json
{
  "points": {
    "A": {"x": 0.0, "y": 0.0, "z": 0.0},
    "B": {"x": 2.0, "y": 0.0, "z": 0.0},
    "S": {"x": 0.0, "y": 0.0, "z": 2.0}
  },
  "edges": [
    {"p1": "A", "p2": "B"},
    {"p1": "A", "p2": "S"}
  ],
  "faces": [
    {"vertices": ["A", "B", "C", "D"]}
  ],
  "unresolved_points": [],
  "violations": []
}
```

---

## 4. Các constraint được hỗ trợ

### 4.1 Hình phẳng (Shape anchors — tự đặt toạ độ)

| Constraint | Số điểm | Mô tả | Tham số thêm |
|-----------|---------|-------|-------------|
| `square` | 4 | Hình vuông trong mặt XY | `side_length` |
| `rectangle` | 4 | Hình chữ nhật | `length`, `width` |
| `rhombus` | 4 | Hình thoi (góc 60°) | `side_length` |
| `trapezoid` | 4 | Hình thang (AB ∥ DC) | `length`, `width`, `height` |
| `equilateral_triangle` | 3 | Tam giác đều | `side_length` |
| `isosceles_triangle` | 3 | Tam giác cân | `length` (chiều cao) |
| `right_triangle` | 3 | Tam giác vuông | — |
| `regular_hexagon` | 6 | Lục giác đều trong XY | `side_length` |

### 4.2 Khối 3D (3D solids)

| Constraint | Số điểm | Mô tả | Tham số thêm |
|-----------|---------|-------|-------------|
| `regular_tetrahedron` | 4 | Tứ diện đều | `side_length` |
| `cube` | 8 | Hình lập phương | `side_length` |
| `rectangular_prism` | 8 | Hình hộp chữ nhật | `length`, `width`, `height` |
| `prism` | 6 | Lăng trụ tam giác đều | `height` |
| `regular_octahedron` | 6 | Bát diện đều | `side_length` |

### 4.3 Điểm phái sinh (Derived points — cần điểm trước)

| Constraint | Đầu vào | Kết quả |
|-----------|---------|---------|
| `midpoint` | `segment=[A,B]`, `point=M` | M = (A+B)/2 |
| `ratio_point` | `segment=[A,B]`, `point=G`, `ratio=t` | G = A + t·(B−A) |
| `centroid` | `points=[A,B,C,…]`, `point=G` | G = trung bình các điểm |
| `foot_perpendicular` | `from_point=S`, `segment=[A,B]`, `point=H` | H = hình chiếu S lên AB |
| `foot_on_plane` | `from_point=S`, `points=[A,B,C,…]`, `point=H` | H = hình chiếu S lên mặt phẳng |
| `perpendicular_to_plane` | `from_point=A`, `points=[A,B,C,…]`, `point=S`, `length=h` | S = A + h·n̂ (n̂ là pháp tuyến mặt phẳng) |
| `symmetric` | `from_point=P`, `points=[M]`, `point=P'` | Đối xứng qua điểm M |
| `symmetric` | `from_point=P`, `points=[A,B]`, `point=P'` | Đối xứng qua đường AB |
| `symmetric` | `from_point=P`, `points=[A,B,C,…]`, `point=P'` | Đối xứng qua mặt phẳng ABC |
| `intersection` | `segment=[A,B]`, `points=[C,D]`, `point=I` | Giao điểm hai đường thẳng |
| `intersection` | `segment=[A,B]`, `points=[C,D,E,…]`, `point=I` | Giao điểm đường AB với mặt phẳng CDE |
| `apex` / `regular_pyramid` / `pyramid` | `points=[S,A,B,C,D]` | S = tâm đáy + h·pháp tuyến |
| `truncated_pyramid` | `points=[A,B,…,A',B',…]`, `ratio`, `height` | Hình chóp cụt (N đáy + N đỉnh) |

### 4.4 Ràng buộc lọc / định vị (Filters)

| Constraint | Vai trò |
|-----------|---------|
| `right_angle` | `points=[arm1, vertex, arm2]`: lọc candidates theo vuông góc |
| `angle` | `points=[arm1, vertex, arm2]`, `degrees=θ`: lọc candidates theo góc cụ thể |
| `distance` | `points=[P,Q]`, `length=L`: lọc candidates theo khoảng cách, hoặc cập nhật `side_length` |
| `edge_length` | Cập nhật `side_length` mặc định |
| `on_line` | Đánh dấu điểm nằm trên đường (cần constraint khác để định vị chính xác) |
| `parallel` | Ghi nhận (hiện là passthrough) |
| `perpendicular` | Ghi nhận (hiện là passthrough) |

---

## 5. Thuật toán giải

Engine sử dụng **constraint propagation** (lan truyền ràng buộc) theo mô hình fixed-point:

```
┌─────────────────────────────────────────────────────────────┐
│  Input: danh sách constraints, coords = {}, candidates = {} │
└─────────────────────────────┬───────────────────────────────┘
                              │
              ┌───────────────▼───────────────┐
              │  _one_pass(pending)            │  ◀─────────────┐
              │  Thử từng handler:             │                │
              │  • True  → đã giải, bỏ khỏi   │                │
              │    pending                     │                │
              │  • False → thiếu tiền đề, giữ │                │
              │    lại                         │                │
              └───────────────┬───────────────┘                │
                              │                                 │
                    progress? ├── Có ──────────────────────────┘
                              │
                              │ Không
                              ▼
              ┌───────────────────────────────┐
              │  _try_perpendicular_system()  │
              │  Phát hiện mẫu SA⊥(ABCD) qua │
              │  nhiều right_angle → sinh     │
              │  candidates cho đỉnh S        │
              └───────────────┬───────────────┘
                              │
                    progress? ├── Có ──────────────────────────┘ (quay lại one_pass)
                              │
                              │ Không
                              ▼
              ┌───────────────────────────────┐
              │  _commit_one_candidate()      │
              │  Chọn candidate tốt nhất      │
              │  (heuristic: z cao nhất)      │
              └───────────────┬───────────────┘
                              │
                    Không còn candidates → kết thúc
```

### 5.1 Candidate list (danh sách ứng viên)

Nhiều điểm không có vị trí duy nhất từ một constraint đơn lẻ. Engine lưu **danh sách ứng viên** (`_candidates`), sau đó dùng các constraint lọc để thu hẹp:

```
equilateral_triangle [S, A, B]  →  4 ứng viên (4 hướng ⊥ AB)
right_angle [S, A, D]           →  lọc còn 2 ứng viên
right_angle [S, A, B] (lần 2)  →  lọc còn 1 → commit
```

Nếu vẫn còn nhiều candidates sau tất cả filters → **z-priority heuristic**: chọn điểm có z lớn nhất (apex ở trên đáy).

### 5.2 Multi-right-angle perpendicular solver

Đây là kỹ thuật then chốt xử lý bài toán phổ biến nhất trong hình học không gian THPT Việt Nam: **SA ⊥ (ABCD)**.

Bài toán thường được LLM phân tích thành:
```
right_angle [S, A, B]   →  SA ⊥ AB
right_angle [S, A, D]   →  SA ⊥ AD
```

Engine phát hiện mẫu này qua `_try_perpendicular_system()`:

```python
# Nhóm các right_angle có cùng (điểm_chưa_biết, đỉnh)
groups[(S, A)] = [B, D]   # S chưa biết, A đã biết, B và D đã biết

# Tính pháp tuyến của mặt phẳng qua AB × AD
normal = normalize(cross(AB, AD))

# Sinh 2 candidates: S = A ± h * normal
candidates[S] = [A + h*normal, A - h*normal]
```

Sau đó z-priority chọn điểm phía trên (z dương) → S đúng vị trí.

---

## 6. Các kỹ thuật toán học

Tất cả tính toán sử dụng **numpy float64**, không dùng sympy (ưu tiên tốc độ).

### 6.1 Cơ sở trực giao (Orthonormal basis)

```python
def perpendicular_pair(u):
    # Cho vector u, sinh (v, w) sao cho {u, v, w} là cơ sở trực giao phải
    ref = [1,0,0] nếu |u[0]| < 0.9 else [0,1,0]
    v = normalize(cross(u, ref))
    w = cross(u, v)   # đã là unit vì u ⊥ v
    return v, w
```

Dùng để sinh candidates vuông góc với một vector đã biết (equilateral_triangle, isosceles_triangle, perpendicular_to_plane).

### 6.2 Hình chiếu điểm lên đường thẳng

```python
def project_point_onto_line(P, A, B):
    t = dot(P-A, B-A) / dot(B-A, B-A)
    return A + t * (B-A)
```

Dùng bởi: `foot_perpendicular`, `reflect_over_line`.

### 6.3 Hình chiếu điểm lên mặt phẳng

```python
def project_point_onto_plane(P, plane_pt, normal):
    n = normalize(normal)
    return P - dot(P - plane_pt, n) * n
```

Dùng bởi: `foot_on_plane`, `reflect_over_plane`.

### 6.4 Giao điểm hai đường thẳng trong 3D (Least Squares)

Hai đường thẳng trong 3D thường **chéo nhau** (không cắt nhau). Engine dùng least-squares để tìm điểm gần nhất:

```
Giải: [d1 | -d2] [t; s] = C - A  (ma trận 3×2)
→ dùng np.linalg.lstsq
Nếu |P1 - P2| < tol → trả về (P1+P2)/2
Nếu không → trả về None (đường chéo)
```

### 6.5 Giao điểm đường thẳng và mặt phẳng

```python
def intersect_line_plane(P, direction, plane_pt, plane_normal):
    denom = dot(direction, plane_normal)
    if |denom| < 1e-12: return None   # song song
    t = dot(plane_pt - P, plane_normal) / denom
    return P + t * direction
```

### 6.6 Pháp tuyến đa giác — Phương pháp Newell

```python
for i in range(n):
    cur, nxt = positions[i], positions[(i+1) % n]
    normal[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2])
    normal[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0])
    normal[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1])
```

Ưu điểm: ổn định số học với đa giác lồi bất kỳ, không cần chọn 3 điểm cụ thể.  
Bias về +z: nếu normal[2] < 0 thì đảo dấu → pháp tuyến luôn hướng lên.

### 6.7 Phản chiếu (Reflection)

| Phản chiếu qua | Công thức |
|---------------|-----------|
| Điểm M | P' = 2M − P |
| Đường AB | P' = 2·foot(P→AB) − P |
| Mặt phẳng | P' = 2·foot(P→plane) − P |

### 6.8 Kiểm tra song song / vuông góc

```python
def are_perpendicular(v1, v2, tol=1e-6):
    return |dot(v1,v2)| / (|v1| * |v2|) < tol

def are_parallel(v1, v2, tol=1e-6):
    return |cross(v1/|v1|, v2/|v2|)| < tol
```

---

## 7. Pipeline xử lý sau khi giải

```
Toạ độ đã giải
      │
      ▼
┌─────────────────┐
│  TopologyBuilder │  → edges (LineRenderer) + faces (Mesh)
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  ConstraintValidator │  → violations: list[str]
└────────┬─────────────┘
         │
         ▼
┌────────────────┐
│   Normalizer   │  (nếu normalize=True)
│  center→origin │
│  scale→unit    │
└────────────────┘
         │
         ▼
   GeometryOutput
```

### TopologyBuilder

Mỗi constraint type biết nó đóng góp bao nhiêu cạnh và mặt:

| Shape | Cạnh | Mặt |
|-------|------|-----|
| triangle | 3 | 1 tam giác |
| square/rectangle/rhombus/trapezoid | 4 | 1 tứ giác |
| regular_hexagon | 6 | 1 lục giác |
| cube/rectangular_prism | 12 | 6 tứ giác |
| prism | 9 | 5 (2 tam giác + 3 tứ giác) |
| regular_tetrahedron | 6 | 4 tam giác |
| regular_octahedron | 12 | 8 tam giác |
| pyramid (N đáy) | N+N | N+1 (đáy + N tam giác) |
| truncated_pyramid (N đáy) | N+N+N | N+2 (đáy + đỉnh + N tứ giác) |

**Deduplication:** Cạnh lưu theo `(min(p1,p2), max(p1,p2))` dạng set. Mặt lưu theo `frozenset(vertices)` → tránh trùng khi nhiều constraint cùng tạo ra một mặt (ví dụ: square + pyramid cùng tạo mặt đáy ABCD).

### ConstraintValidator

Sau khi giải xong, validator kiểm tra từng constraint bằng toán học:

- **square**: 4 cạnh bằng nhau + 4 góc = 90°
- **equilateral_triangle**: 3 cạnh bằng nhau
- **right_angle**: dot product của hai vector = 0
- **midpoint**: |M - (A+B)/2| < tol
- **centroid**: |G - mean(A,B,C,…)| < tol
- **perpendicular_to_plane**: SA × normal ≈ 0
- **symmetric**: |P' - expected_reflection| < tol
- **regular_octahedron**: 12 cạnh bằng nhau
- v.v.

Kết quả violations trả về trong `GeometryOutput.violations: list[str]`.

---

## 8. API Server

### Khởi động

```bash
# Trực tiếp
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Qua CLI
python -m geometry_engine serve --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

#### `GET /health`
```json
{"status": "ok", "version": "1.0.0"}
```

#### `POST /solve`

**Request:** `Content-Type: application/json`, body là `GeometryInput`

**Response:** `GeometryOutput`

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "points": ["A","B","C","D","S"],
    "constraints": [
      {"type": "square", "points": ["A","B","C","D"]},
      {"type": "right_angle", "points": ["S","A","B"]},
      {"type": "right_angle", "points": ["S","A","D"]}
    ],
    "side_length": 2.0
  }'
```

**Swagger UI:** `http://localhost:8000/docs`  
**ReDoc:** `http://localhost:8000/redoc`

---

## 9. CLI

```bash
# Giải từ file
python -m geometry_engine solve problem.json --pretty

# Giải từ stdin
echo '{"points":["A","B","C"],"constraints":[...],"side_length":1}' | \
  python -m geometry_engine solve

# Chỉ validate (kiểm tra vi phạm, không in toạ độ)
python -m geometry_engine validate problem.json

# Khởi động server
python -m geometry_engine serve --port 8000 --reload
```

Exit code `1` nếu có violations.

---

## 10. Test Suite

```bash
# Chạy tất cả (99 tests)
pytest tests/ -v

# Chỉ một file
pytest tests/test_engine.py -v

# Với coverage
pytest tests/ --cov=geometry_engine --cov-report=term-missing
```

### Phân bổ tests

| File | Số tests | Nội dung |
|------|---------|----------|
| `test_engine.py` | 46 | Toạ độ, constraint solving, full pyramid example |
| `test_topology.py` | 18 | Đếm cạnh + mặt cho từng shape |
| `test_server.py` | 11 | HTTP API, status codes, response schema |
| `test_advanced.py` | 24 | Constraint types mới, SA⊥(ABCD) pattern |
| **Tổng** | **99** | **99/99 passed** |

---

## 11. Ví dụ end-to-end

### Bài toán điển hình: Chóp S.ABCD với SA ⊥ (ABCD)

**Đề bài:** Hình vuông ABCD cạnh 2. SA ⊥ (ABCD), SA = 2. Tìm toạ độ các đỉnh.

**LLM sinh ra JSON:**
```json
{
  "points": ["A", "B", "C", "D", "S"],
  "constraints": [
    {"type": "square",      "points": ["A","B","C","D"], "length": 2.0},
    {"type": "right_angle", "points": ["S","A","B"]},
    {"type": "right_angle", "points": ["S","A","D"]},
    {"type": "distance",    "points": ["S","A"], "length": 2.0}
  ],
  "side_length": 2.0
}
```

**Quá trình giải:**
1. `square` → đặt A=(0,0,0), B=(2,0,0), C=(2,2,0), D=(0,2,0)
2. `right_angle [S,A,B]` + `right_angle [S,A,D]` → `_try_perpendicular_system()`:
   - AB = (2,0,0), AD = (0,2,0)
   - normal = normalize(AB × AD) = (0,0,1)
   - candidates[S] = [A+(0,0,1), A-(0,0,1)] = [(0,0,1), (0,0,-1)]
3. `distance [S,A]=2` → scale candidates → S=(0,0,2)
4. z-priority → chọn S=(0,0,2)

**Kết quả:**
```json
{
  "points": {
    "A": {"x": 0.0, "y": 0.0, "z": 0.0},
    "B": {"x": 2.0, "y": 0.0, "z": 0.0},
    "C": {"x": 2.0, "y": 2.0, "z": 0.0},
    "D": {"x": 0.0, "y": 2.0, "z": 0.0},
    "S": {"x": 0.0, "y": 0.0, "z": 2.0}
  },
  "edges": [
    {"p1": "A", "p2": "B"}, {"p1": "B", "p2": "C"},
    {"p1": "C", "p2": "D"}, {"p1": "D", "p2": "A"},
    {"p1": "S", "p2": "A"}, {"p1": "S", "p2": "B"},
    {"p1": "S", "p2": "C"}, {"p1": "S", "p2": "D"}
  ],
  "faces": [
    {"vertices": ["A","B","C","D"]},
    {"vertices": ["S","A","B"]},
    {"vertices": ["S","B","C"]},
    {"vertices": ["S","C","D"]},
    {"vertices": ["S","D","A"]}
  ],
  "unresolved_points": [],
  "violations": []
}
```

### Bài toán tam giác đều trong không gian

```json
{
  "points": ["A", "B", "C", "G"],
  "constraints": [
    {"type": "equilateral_triangle", "points": ["A","B","C"]},
    {"type": "centroid", "point": "G", "points": ["A","B","C"]}
  ],
  "side_length": 1.0
}
```

### Hình hộp chữ nhật với điểm trung điểm

```json
{
  "points": ["A","B","C","D","A1","B1","C1","D1","M"],
  "constraints": [
    {
      "type": "rectangular_prism",
      "points": ["A","B","C","D","A1","B1","C1","D1"],
      "length": 3.0, "width": 2.0, "height": 4.0
    },
    {"type": "midpoint", "point": "M", "segment": ["A","C1"]}
  ]
}
```

---

## Phụ lục: Dependencies

```
numpy>=1.26      # vector math (float64)
pydantic>=2.0    # I/O validation
fastapi>=0.110   # HTTP server
uvicorn>=0.29    # ASGI runner
pytest>=7.4      # test framework
pytest-cov>=4.1  # coverage
httpx>=0.27      # test HTTP client
sympy>=1.12      # (reserved, chưa dùng)
```
