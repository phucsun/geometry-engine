# GeometryEngine

`GeometryEngine` là backend dựng hình học không gian 3D từ hai loại đầu vào:

- `GeometryInput` JSON đã có cấu trúc
- ảnh đề toán, qua pipeline `ocr_llm -> backend_pipeline -> geometry_engine`

Kết quả đầu ra là `GeometryOutput` gồm toạ độ điểm, cạnh, mặt, các điểm chưa giải được, và danh sách vi phạm constraint.

```text
Client
  ├─ POST /solve       -> GeometryInput JSON -> GeometryEngine -> GeometryOutput
  └─ POST /solve-image -> image upload       -> OCR/LLM        -> GeometryEngine -> GeometryOutput
```

## Thành phần chính

- `geometry_engine/`: deterministic solver, topology builder, validator, normalizer
- `ocr_llm/`: OCR + LLM extraction + payload repair về `GeometryInput`
- `backend_pipeline.py`: orchestration `image -> analyze_image -> GeometryEngine`
- `server.py`: FastAPI app với `GET /health`, `POST /solve`, `POST /solve-image`

## Cài đặt

```bash
pip install -r requirements.txt
```

Project dùng Groq cho OCR/LLM image pipeline. Cần có `GROQ_API_KEY` nếu muốn gọi `POST /solve-image` hoặc chạy `backend_pipeline.py`.

PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

## Chạy nhanh

### 1. API server

```bash
python -m geometry_engine serve --port 8000 --reload
```

hoặc:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)  
ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 2. CLI solver cho `GeometryInput`

```bash
python -m geometry_engine solve problem.json --pretty
python -m geometry_engine validate problem.json
```

### 3. CLI ảnh đề toán

```bash
python backend_pipeline.py math_problem_image/math_prob.jpg --pretty
```

## API

### `GET /health`

```json
{"status": "ok", "version": "1.0.0"}
```

### `POST /solve`

Nhận `application/json` theo schema `GeometryInput`.

Ví dụ:

```bash
curl -X POST "http://localhost:8000/solve" \
  -H "Content-Type: application/json" \
  -d '{
    "points": ["A", "B", "C", "D", "S"],
    "constraints": [
      { "type": "square", "points": ["A", "B", "C", "D"] },
      { "type": "right_angle", "points": ["S", "A", "B"] },
      { "type": "right_angle", "points": ["S", "A", "D"] }
    ],
    "side_length": 2.0,
    "normalize": false,
    "validate_constraints": true
  }'
```

### `POST /solve-image`

Nhận `multipart/form-data` với field bắt buộc `image`.

Ví dụ:

```bash
curl -X POST "http://localhost:8000/solve-image" \
  -H "accept: application/json" \
  -F "image=@math_problem_image/math_prob.jpg;type=image/jpeg"
```

Response của cả hai route đều là `GeometryOutput`:

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

`POST /solve-image` có error contract hiện tại:

- `400`: file không phải ảnh hoặc file rỗng
- `422`: OCR/LLM/solver xử lý không được ảnh hợp lệ
- `500`: lỗi không mong đợi

## Python API

### Gọi solver trực tiếp

```python
from geometry_engine import GeometryEngine, GeometryInput
from geometry_engine.models import Constraint

engine = GeometryEngine()
result = engine.solve(
    GeometryInput(
        points=["A", "B", "C", "D"],
        constraints=[
            Constraint(type="square", points=["A", "B", "C", "D"]),
        ],
        side_length=2.0,
    )
)

print(result.points)
print(result.edges)
print(result.faces)
```

### Gọi pipeline ảnh

```python
from backend_pipeline import solve_image

output = solve_image("math_problem_image/math_prob.jpg")
print(output.model_dump())
```

## OCR/LLM pipeline hiện tại

`ocr_llm` không giải hình học. Nó làm 3 việc:

1. OCR ảnh thành text đề bài
2. nhận diện dạng bài để chọn prompt hẹp hơn
3. sửa payload LLM trước khi validate thành `GeometryInput`

Một số family đã có rule repair chuyên biệt:

- `square_pyramid`
- `dihedral_pyramid`
- `equal_side_face_angle_pyramid`
- `right_triangular_prism`
- `oblique_triangular_prism`
- `generic_shapes`

Các repair hiện có gồm:

- chuẩn hoá `a`, `2a`, `a/2`, `a√3` thành số
- sửa `centroid`, `intersection`, `perpendicular_to_plane`
- đổi angle mặt phẳng thành `dihedral_angle`
- tạo `equal_side_face_angle`
- gom dữ kiện lăng trụ thành `right_prism` hoặc `oblique_prism`

## Constraint groups

Các loại constraint hiện được dispatch qua `geometry_engine.registry`:

- base anchors: `square`, `rectangle`, `parallelogram`, `rhombus`, `trapezoid`, `equilateral_triangle`, `isosceles_triangle`, `right_triangle`, `regular_hexagon`, `regular_polygon`
- solids and pyramids/prisms: `regular_tetrahedron`, `cube`, `rectangular_prism`, `prism`, `oblique_prism`, `regular_octahedron`, `right_prism`, `apex`, `regular_pyramid`, `pyramid`, `truncated_pyramid`
- derived points: `midpoint`, `ratio_point`, `centroid`, `circumcenter`, `orthocenter`, `incenter`, `equidistant`, `angle_bisector`, `median`, `foot_perpendicular`, `foot_on_plane`, `perpendicular_to_plane`, `symmetric`, `intersection`
- filtering/disambiguation: `right_angle`, `angle`, `distance`, `edge_length`, `on_line`, `collinear`
- passthrough: `parallel`, `perpendicular`, `coplanar`

Ngoài ra backend có các internal rule dùng sau bước repair:

- `dihedral_angle`
- `equal_side_face_angle`

## Test suite

Kết quả hiện tại trong repo:

```bash
pytest tests -q
```

```text
151 passed, 17 skipped
```

Các nhóm test chính:

- solver core và topology
- advanced geometry cases
- OCR/LLM extraction + repair
- backend image pipeline
- FastAPI server

Lưu ý: các test API có thể bị skip nếu môi trường chưa cài `fastapi`.

## Cấu trúc repo

```text
geometry_engine/
├── geometry_engine/
│   ├── handlers/
│   ├── __main__.py
│   ├── engine.py
│   ├── models.py
│   ├── registry.py
│   ├── topology.py
│   ├── validator.py
│   └── normalizer.py
├── ocr_llm/
│   ├── prompts/
│   └── repairs/
├── backend_pipeline.py
├── server.py
├── math_problem_image/
├── tests/
├── README.md
└── DOCUMENTATION.md
```

## Dependencies

- `numpy`
- `pydantic`
- `fastapi`
- `uvicorn[standard]`
- `httpx`
- `python-multipart`
- `langchain-core`
- `langchain-groq`
- `python-dotenv`
- `pytest`
- `pytest-cov`

Chi tiết kỹ thuật xem [DOCUMENTATION.md](DOCUMENTATION.md).
