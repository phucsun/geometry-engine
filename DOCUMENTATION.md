# GeometryEngine — Tài liệu kỹ thuật

## 1. Tổng quan

Project hiện tại gồm ba lớp rõ ràng:

- `ocr_llm`: đọc ảnh, OCR, prompt LLM, repair payload thành `GeometryInput`
- `backend_pipeline`: orchestration cho luồng ảnh
- `geometry_engine`: deterministic solver sinh `GeometryOutput`

Hai luồng vào chính:

```text
Structured input
Client -> POST /solve -> GeometryEngine -> GeometryOutput

Image input
Client -> POST /solve-image -> backend_pipeline -> ocr_llm -> GeometryEngine -> GeometryOutput
```

Mục tiêu của việc tách lớp:

- OCR/LLM xử lý phần ngôn ngữ mơ hồ
- solver xử lý phần hình học xác định và kiểm chứng được
- API chỉ là HTTP wrapper, không chứa logic giải

## 2. Kiến trúc module

```text
geometry_engine/
├── geometry_engine/
│   ├── handlers/              # grouped handlers theo family
│   ├── __init__.py            # public Python API exports
│   ├── __main__.py            # CLI: solve / validate / serve
│   ├── engine.py              # fixed-point solver
│   ├── errors.py              # SolverError
│   ├── models.py              # GeometryInput / GeometryOutput
│   ├── normalizer.py          # normalize output
│   ├── registry.py            # constraint type -> handler
│   ├── topology.py            # edges / faces
│   ├── utils.py               # vector math
│   └── validator.py           # hậu kiểm constraint
├── ocr_llm/
│   ├── analyzer.py            # OCR + LLM + validation entrypoints
│   ├── problem_types.py       # detect family từ đề bài
│   ├── prompts/               # prompt template + rule snippets
│   └── repairs/               # sửa payload LLM trước khi validate
├── backend_pipeline.py        # image -> GeometryOutput
├── server.py                  # FastAPI routes
├── math_problem_image/        # ảnh mẫu để test
└── tests/
```

Vai trò từng khối:

- `geometry_engine` không hiểu tiếng Việt hay OCR; nó chỉ nhận `GeometryInput`
- `ocr_llm` không gọi solver; nó chỉ xuất `GeometryInput`
- `backend_pipeline` là chỗ nối hai khối đó
- `server.py` đưa public API ra HTTP

## 3. Public interfaces

### 3.1 `GeometryInput`

Schema được định nghĩa trong `geometry_engine/models.py`.

```json
{
  "points": ["A", "B", "C", "D", "S"],
  "constraints": [
    { "type": "square", "points": ["A", "B", "C", "D"] },
    { "type": "right_angle", "points": ["S", "A", "B"] },
    { "type": "right_angle", "points": ["S", "A", "D"] }
  ],
  "side_length": 2.0,
  "normalize": false,
  "validate_constraints": true
}
```

Các field của `Constraint` hiện có:

| Field | Meaning |
|------|---------|
| `type` | loại constraint |
| `points` | danh sách điểm theo thứ tự có nghĩa |
| `point` | điểm kết quả |
| `segment` | đoạn thẳng hai điểm |
| `from_point` | điểm xuất phát |
| `length` | độ dài / kích thước chính |
| `width` | kích thước phụ |
| `height` | chiều cao |
| `ratio` | tỉ lệ |
| `degrees` | góc theo độ |

### 3.2 `GeometryOutput`

```json
{
  "points": {
    "A": {"x": 0.0, "y": 0.0, "z": 0.0}
  },
  "edges": [],
  "faces": [],
  "unresolved_points": [],
  "violations": []
}
```

Ý nghĩa:

- `points`: toạ độ 3D đã giải được
- `edges`: cạnh cấu trúc cho render
- `faces`: mặt polygon cho render mesh
- `unresolved_points`: điểm không giải ra được
- `violations`: constraint không thoả sau hậu kiểm

## 4. API hiện tại

### `GET /health`

Trả:

```json
{"status": "ok", "version": "1.0.0"}
```

### `POST /solve`

- content type: `application/json`
- body: `GeometryInput`
- response: `GeometryOutput`

Use case:

- client đã có JSON có cấu trúc
- test solver trực tiếp
- bypass toàn bộ OCR/LLM

### `POST /solve-image`

- content type: `multipart/form-data`
- field bắt buộc: `image`
- response: `GeometryOutput`

Flow nội bộ:

1. FastAPI nhận `UploadFile`
2. validate `content_type` phải là `image/*`
3. từ chối file rỗng
4. ghi file tạm
5. gọi `backend_pipeline.solve_image(path)`
6. xoá file tạm trong `finally`

Error contract hiện tại:

- `400`: file không phải ảnh hoặc file rỗng
- `422`: pipeline OCR/LLM/solver ném lỗi nghiệp vụ
- `500`: lỗi không mong đợi

Swagger:

- [http://localhost:8000/docs](http://localhost:8000/docs)
- [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 5. CLI hiện tại

`geometry_engine/__main__.py` export 3 lệnh:

```bash
python -m geometry_engine solve problem.json --pretty
python -m geometry_engine validate problem.json
python -m geometry_engine serve --host 0.0.0.0 --port 8000 --reload
```

Luồng ảnh hiện có CLI riêng qua `backend_pipeline.py`:

```bash
python backend_pipeline.py math_problem_image/math_prob.jpg --pretty
```

## 6. Solver architecture

`GeometryEngine.solve(...)` trong `engine.py` chạy theo fixed-point loop:

1. reset state `coords`, `candidates`, `side_length`
2. lặp qua `pending constraints`
3. mỗi constraint được dispatch qua `registry.get_handler(...)`
4. handler:
   - trả `True` nếu đã tạo progress
   - trả `False` nếu chưa đủ tiền đề
5. khi loop bị stall:
   - thử `_try_perpendicular_system(...)`
   - thử `_try_equilateral_right_angle_system(...)`
   - nếu vẫn chưa tiến triển, commit một candidate
6. sau cùng:
   - commit toàn bộ candidate còn lại
   - build topology
   - validate constraints nếu bật cờ
   - normalize output nếu bật cờ

Các nhóm handler hiện có:

- `BaseShapeHandlers`
- `SolidShapeHandlers`
- `DerivedPointHandlers`
- `SpecialRuleHandlers`
- `ConstraintHandlers`

Registry hiện map các nhóm constraint sau:

- 2D anchors:
  - `square`, `rectangle`, `parallelogram`, `rhombus`, `trapezoid`
  - `equilateral_triangle`, `isosceles_triangle`, `right_triangle`
  - `regular_hexagon`, `regular_polygon`
- 3D solids / prism / pyramid:
  - `regular_tetrahedron`, `cube`, `rectangular_prism`, `prism`
  - `oblique_prism`, `right_prism`, `regular_octahedron`
  - `apex`, `regular_pyramid`, `pyramid`, `truncated_pyramid`
- derived points:
  - `midpoint`, `ratio_point`, `centroid`, `circumcenter`, `orthocenter`
  - `incenter`, `equidistant`, `angle_bisector`, `median`
  - `foot_perpendicular`, `foot_on_plane`, `perpendicular_to_plane`
  - `symmetric`, `intersection`
- filter / disambiguation:
  - `right_angle`, `angle`, `distance`, `edge_length`, `on_line`, `collinear`
- passthrough:
  - `parallel`, `perpendicular`, `coplanar`

Backend-only constraints do OCR repair sinh ra:

- `dihedral_angle`
- `equal_side_face_angle`

## 7. OCR/LLM preprocessing

`ocr_llm/analyzer.py` là entrypoint chính:

- `run_ocr(image_path) -> str`
- `analyze_problem_text(problem_text) -> GeometryInput`
- `analyze_image(image_path) -> tuple[str, GeometryInput]`

### 7.1 Problem type detection

`ocr_llm/problem_types.py` hiện có các family:

- `square_pyramid`
- `dihedral_pyramid`
- `equal_side_face_angle_pyramid`
- `right_triangular_prism`
- `oblique_triangular_prism`
- `generic_shapes`

Việc detect là local, regex-based, không gọi model.

### 7.2 Prompt scoping

Prompt builder chỉ đưa vào:

- family hiện tại
- subset constraint nên ưu tiên
- rule snippets cho family đó
- một ví dụ gần nhất

Mục tiêu là thu hẹp không gian trả lời của LLM, không để model tự phát minh format.

### 7.3 Payload repair

Sau khi LLM trả payload, project luôn chạy repair trước khi `GeometryInput.model_validate(...)`.

Các repair hiện có:

- chuẩn hoá giá trị symbolic: `a`, `2a`, `a/2`, `a√3`
- sửa field placement cho `centroid`, `intersection`, `perpendicular_to_plane`
- thêm `apex` hoặc `parallelogram` nếu đề chóp có nhưng payload thiếu
- đổi góc giữa hai mặt phẳng thành `dihedral_angle`
- đổi cụm “các mặt bên cùng tạo với mặt đáy góc ...” thành `equal_side_face_angle`
- gom dữ kiện lăng trụ đứng thành `right_prism`
- gom dữ kiện lăng trụ xiên thành `oblique_prism`
- reorder constraints để solver xử lý thuận hơn

Điểm quan trọng:

- `ocr_llm` chịu trách nhiệm semantic cleanup
- `geometry_engine` giả định input đã là schema sạch

## 8. Backend image pipeline

`backend_pipeline.py` là orchestration layer:

```text
image -> analyze_image -> GeometryInput -> GeometryEngine.solve -> GeometryOutput
```

Các entrypoint:

- `solve_image(path) -> GeometryOutput`
- `solve_image_json(path, pretty=False) -> str`
- CLI `python backend_pipeline.py image.png --pretty`

Pipeline này là nơi thích hợp để test end-to-end ngoài HTTP server.

## 9. Examples

### 9.1 Structured solve

```json
{
  "points": ["A", "B", "C", "D"],
  "constraints": [
    {"type": "square", "points": ["A", "B", "C", "D"]}
  ],
  "side_length": 2.0
}
```

Kỳ vọng:

- solver đặt 4 điểm trên mặt phẳng `z = 0`
- sinh 4 cạnh, 1 mặt
- không có `violations`

### 9.2 Image solve

```bash
curl -X POST "http://localhost:8000/solve-image" \
  -H "accept: application/json" \
  -F "image=@math_problem_image/math_prob.jpg;type=image/jpeg"
```

Kết quả trả về vẫn là `GeometryOutput`, không lộ `ocr_text` hay `GeometryInput`.

## 10. Dependencies

Runtime dependencies hiện tại:

- `numpy>=1.26`
- `pydantic>=2.0`
- `fastapi>=0.110`
- `uvicorn[standard]>=0.29`
- `httpx>=0.27`
- `python-multipart>=0.0.9`
- `langchain-core>=0.2`
- `langchain-groq>=0.1`
- `python-dotenv>=1.0`

Test / tooling:

- `pytest>=7.4`
- `pytest-cov>=4.1`

Lưu ý vận hành:

- `POST /solve-image` và `backend_pipeline.py` cần `GROQ_API_KEY`
- test server có thể bị skip nếu môi trường chưa cài `fastapi`

## 11. Trạng thái test hiện tại

Trong môi trường kiểm tra gần nhất của repo:

```bash
pytest tests -q
```

Kết quả:

```text
151 passed, 17 skipped
```

Các nhóm test đang có:

- core engine
- topology
- advanced geometry cases
- OCR/LLM extraction và repair
- backend image pipeline
- problem type detection
- FastAPI server

## 12. Ranh giới trách nhiệm

Nếu bạn cần sửa project này, nên giữ ranh giới sau:

- thay đổi OCR/prompt/repair: sửa trong `ocr_llm`
- thay đổi logic nối ảnh -> solver: sửa trong `backend_pipeline.py`
- thay đổi public API HTTP: sửa trong `server.py`
- thay đổi suy luận toạ độ hoặc constraint semantics: sửa trong `geometry_engine`

Không đẩy logic solver vào prompt LLM. Project hiện được tách như vậy để:

- dễ test
- deterministic hơn
- validate được
- không phụ thuộc model cho phần suy luận hình học thuần
