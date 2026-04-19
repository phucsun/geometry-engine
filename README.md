# GeometryEngine

Nhận JSON ràng buộc hình học từ LLM, giải toạ độ 3D, trả về điểm + cạnh + mặt cho Unity AR render.

```
Mobile App → OCR → LLM → GeometryEngine → Unity AR
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng nhanh

### API Server

```bash
python -m geometry_engine serve --port 8000
# hoặc
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI tại `http://localhost:8000/docs`

### CLI

```bash
# Giải từ file
python -m geometry_engine solve problem.json --pretty

# Giải từ stdin
echo '{...}' | python -m geometry_engine solve

# Kiểm tra vi phạm constraint
python -m geometry_engine validate problem.json
```

### Python API

```python
from geometry_engine import GeometryEngine, GeometryInput
from geometry_engine.models import Constraint

engine = GeometryEngine()
result = engine.solve(GeometryInput(
    points=["A", "B", "C", "D", "S"],
    constraints=[
        Constraint(type="square", points=["A", "B", "C", "D"]),
        Constraint(type="right_angle", points=["S", "A", "B"]),
        Constraint(type="right_angle", points=["S", "A", "D"]),
    ],
    side_length=2.0,
))
print(result.points)   # {"A": Point3D(x=0,y=0,z=0), "S": Point3D(x=0,y=0,z=2), ...}
print(result.edges)    # [Edge(p1="A", p2="B"), ...]
print(result.faces)    # [Face(vertices=["A","B","C","D"]), ...]
```

## Input / Output

**Request — `POST /solve`:**

```json
{
  "points": ["A", "B", "C", "D", "S"],
  "constraints": [
    { "type": "square",      "points": ["A","B","C","D"] },
    { "type": "right_angle", "points": ["S","A","B"] },
    { "type": "right_angle", "points": ["S","A","D"] }
  ],
  "side_length": 2.0,
  "normalize": false,
  "validate_constraints": true
}
```

**Response:**

```json
{
  "points": {
    "A": {"x": 0.0, "y": 0.0, "z": 0.0},
    "S": {"x": 0.0, "y": 0.0, "z": 2.0}
  },
  "edges":  [{"p1": "A", "p2": "B"}, "..."],
  "faces":  [{"vertices": ["A","B","C","D"]}, "..."],
  "unresolved_points": [],
  "violations": []
}
```

## Các constraint được hỗ trợ

| Nhóm | Loại |
|------|------|
| Hình phẳng | `square`, `rectangle`, `rhombus`, `trapezoid`, `equilateral_triangle`, `isosceles_triangle`, `right_triangle`, `regular_hexagon` |
| Khối 3D | `cube`, `rectangular_prism`, `prism`, `regular_tetrahedron`, `regular_octahedron` |
| Điểm phái sinh | `midpoint`, `ratio_point`, `centroid`, `foot_perpendicular`, `foot_on_plane`, `perpendicular_to_plane`, `symmetric`, `intersection`, `apex`, `regular_pyramid`, `pyramid`, `truncated_pyramid` |
| Bộ lọc | `right_angle`, `angle`, `distance`, `edge_length`, `on_line` |

Chi tiết đầy đủ xem [DOCUMENTATION.md](DOCUMENTATION.md).

## Tests

```bash
pytest tests/ -v                                        # 99 tests
pytest tests/ --cov=geometry_engine --cov-report=term  # với coverage
```

## Cấu trúc

```
geometry_engine/
├── engine.py       # Constraint propagation solver
├── models.py       # Pydantic I/O schemas
├── topology.py     # Sinh edges + faces cho Unity
├── validator.py    # Kiểm tra hậu kỳ
├── normalizer.py   # Scale về unit sphere
└── utils.py        # Vector math (numpy)
server.py           # FastAPI app
```

## Stack

- **Python 3.10+**, numpy, pydantic v2, FastAPI, uvicorn
