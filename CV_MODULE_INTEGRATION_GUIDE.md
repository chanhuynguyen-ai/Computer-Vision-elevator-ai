# Tài liệu mô tả module CV hiện tại và cách ghép vào đề tài Elevator AI

## 1. Mục đích của tài liệu

Tài liệu này mô tả đầy đủ module Computer Vision (CV) hiện tại đang chạy độc lập trên Jetson Nano theo hướng service hóa. Mục tiêu là để:

- dùng tài liệu này làm đầu vào cho chatbot hoặc một agent khác hiểu rõ module CV hiện tại đang làm gì;
- ghép module CV hiện tại vào đề tài thang máy AI tổng thể của bạn;
- giúp chatbot hiểu cách nối dữ liệu CV với chatbot/LLM, frontend, backend và database của đề tài;
- làm nền để sau này tạo một tài liệu `.md` tổng hợp cuối cùng cho toàn bộ project.

Tài liệu này tập trung vào **module CV đang chạy ổn**. Tài liệu project tổng thể của Elevator AI sẽ đóng vai trò mô tả phần chatbot, web app, voice, DB ứng dụng và logic điều phối chính. Khi đưa hai tài liệu này cho chatbot, chatbot phải hiểu:

1. module CV hiện tại là một service độc lập;
2. service này cung cấp camera stream + dữ liệu CV qua API + PostgreSQL;
3. project Elevator AI tổng thể phải gọi/nhúng/đọc service CV này thay vì viết lại từ đầu;
4. chatbot của project không được đoán dữ liệu camera bằng LLM mà phải đọc từ database CV bằng SQL.

---

## 2. Trạng thái hiện tại của module CV

Module CV hiện tại đã được service hóa thành một backend chạy trên Jetson Nano theo hướng:

- camera CSI/USB -> OpenCV/GStreamer;
- inference bằng TensorRT trên Jetson Nano;
- nhận diện người bằng YOLO detect;
- tùy chọn pose để suy posture;
- tracking theo track id;
- log dữ liệu vào PostgreSQL;
- expose API để frontend và chatbot có thể sử dụng;
- có dashboard test trực tiếp tại route `/`.

### Những gì đã chạy được

- backend FastAPI/Uvicorn chạy được trên Jetson;
- API status hoạt động;
- API stream hoạt động;
- camera CSI Jetson đã mở được bằng `nvarguscamerasrc`;
- TensorRT + pycuda chạy được trong `python3` hệ thống trên Jetson;
- dữ liệu density (`camera_occupancy_samples`) đã sinh ra;
- route `/` đã được thêm để test giao diện mà không cần chờ frontend React chính.

### Vai trò hiện tại của module CV

Module này là **CV Realtime Service**, chưa phải toàn bộ project Elevator AI.

Nó chịu trách nhiệm:

- đọc camera;
- detect người và một số object;
- tracking;
- pose và một số event cơ bản;
- ghi dữ liệu vào database CV;
- cung cấp API để frontend và chatbot đọc dữ liệu.

Nó **không** chịu trách nhiệm:

- quản lý prompt/FAQ/knowledge base của chatbot;
- semantic matching của chatbot;
- UI web tổng thể của Elevator AI;
- orchestration đầy đủ của project thang máy AI.

---

## 3. Vì sao phải tách 2 database: `elevator_cv` và `elevator_llm`

Đây là nguyên tắc kiến trúc rất quan trọng.

### 3.1 Database `elevator_cv`

Dùng cho dữ liệu Computer Vision thời gian thực.

Ví dụ các bảng:

- `camera_events`
- `camera_occupancy_samples`
- `person_registry`
- `face_embeddings`

Loại dữ liệu trong `elevator_cv`:

- sự kiện té ngã;
- lying;
- crowd;
- bottle hoặc vật thể lạ;
- số người theo thời gian;
- thông tin tracking/person/embedding;
- snapshot/event metadata.

### 3.2 Database `elevator_llm`

Dùng cho chatbot/LLM.

Ví dụ dữ liệu:

- prompt templates;
- câu hỏi mẫu;
- câu trả lời mẫu;
- embeddings cho FAQ hoặc tài liệu;
- semantic matcher data;
- tri thức nghiệp vụ thang máy.

### 3.3 Vì sao không gộp chung

Không nên để dữ liệu CV và dữ liệu chatbot chung một database logic, vì:

1. **Bản chất dữ liệu khác nhau**
   - CV là dữ liệu sự kiện realtime, time series, analytics.
   - LLM là dữ liệu tri thức, text, embeddings, prompts.

2. **Cách truy vấn khác nhau**
   - CV chủ yếu query bằng SQL analytics.
   - LLM/query knowledge chủ yếu là semantic matching, retrieval, hoặc metadata lookup.

3. **Mục tiêu khác nhau**
   - `elevator_cv` trả lời: camera thấy gì, lúc nào, ở đâu, bao nhiêu người, có sự kiện gì.
   - `elevator_llm` trả lời: hệ thống là gì, hướng dẫn sử dụng, FAQ, nghiệp vụ thang máy, giải thích cảnh báo.

4. **Tránh chatbot đoán sai dữ liệu camera**
   - Mọi câu hỏi về dữ liệu camera phải đọc SQL từ `elevator_cv`.
   - LLM chỉ format câu trả lời, không tự bịa dữ liệu thị giác.

### 3.4 Quy tắc tích hợp cho chatbot

- Câu hỏi dạng **dữ liệu camera/realtime** -> query `elevator_cv`.
- Câu hỏi dạng **FAQ/nghiệp vụ/giải thích** -> query hoặc retrieve từ `elevator_llm`.

Ví dụ:

- "Hôm nay có bao nhiêu lần té ngã?" -> `elevator_cv`
- "Camera 1 đông nhất lúc nào?" -> `elevator_cv`
- "Hệ thống cảnh báo lying nghĩa là gì?" -> `elevator_llm`
- "Cách sử dụng giao diện chatbot thang máy" -> `elevator_llm`

---

## 4. Cây thư mục của module CV hiện tại

Đây là cây thư mục của bundle CV hiện tại (bản đã vá để chạy trên Jetson):

```text
/elevator_cv_jetson_bundle
├── .env.cv.example
├── .env.llm.example
├── main.py
├── README_RUN.md
├── requirements.txt
├── schema_cv.sql
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── camera_service.py
│   ├── config.py
│   ├── db.py
│   ├── event_logger.py
│   ├── face_recog.py
│   ├── posture.py
│   ├── runtime_trt.py
│   ├── runtime_ultra.py
│   ├── tracker.py
│   └── yolo_utils.py
├── scripts/
│   ├── build_engines.sh
│   └── export_models.py
└── frontend/
    ├── index.html
    ├── package.json
    ├── package-lock.json
    ├── vite.config.js
    └── README_FRONTEND.txt
```

### Ý nghĩa nhanh của cây thư mục

- `main.py`: entrypoint chạy service CV.
- `app/`: toàn bộ backend logic của CV service.
- `scripts/`: script export ONNX và build TensorRT engine.
- `frontend/`: frontend mẫu/Vite cũ giữ lại để tham khảo, không phải UI tích hợp cuối cùng.
- `.env.cv.example`: biến môi trường cho phần CV.
- `.env.llm.example`: biến môi trường cho phần LLM.
- `schema_cv.sql`: schema database cho `elevator_cv`.

---

## 5. Vai trò chi tiết của từng file

## 5.1 File gốc ở thư mục root

### `main.py`

Entrypoint tối giản để chạy Uvicorn và load `app.api`.

Nhiệm vụ:

- khởi chạy web server CV;
- expose FastAPI app ra cổng mặc định 8000.

### `README_RUN.md`

Tài liệu chạy nhanh module CV trên Jetson.

Nội dung chính:

- tạo PostgreSQL;
- chuẩn bị model;
- build TensorRT engine;
- cài dependencies;
- source env;
- chạy service;
- test status/stream/events/density.

### `requirements.txt`

Danh sách package tối thiểu để backend CV chạy trên Jetson Python hệ thống.

### `schema_cv.sql`

Schema dành cho database `elevator_cv`.

Bao gồm các bảng chính:

- `person_registry`
- `face_embeddings`
- `camera_events`
- `camera_occupancy_samples`

### `.env.cv.example`

Cấu hình cho module CV.

Bao gồm:

- backend `trt|ultralytics`
- camera source
- database CV
- database LLM (chỉ để biết tách riêng, không dùng để ghi dữ liệu camera)
- bật/tắt face/pose
- đường dẫn TensorRT engine
- threshold và sampling

### `.env.llm.example`

Cấu hình mẫu cho phần LLM database.

File này không dùng để chạy CV realtime. Nó chỉ nhắc rằng chatbot cần một DB riêng.

---

## 5.2 Thư mục `app/`

### `app/config.py`

Nơi đọc toàn bộ config từ environment variables.

Nhiệm vụ:

- chọn backend `trt` hoặc `ultralytics`;
- chọn camera source;
- định nghĩa DB CV và DB LLM;
- bật/tắt face và pose;
- thiết lập threshold, interval, image size;
- cấu hình API host/port.

Đây là file cấu hình trung tâm của service CV.

### `app/db.py`

Lớp giao tiếp PostgreSQL cho phần CV.

Nhiệm vụ:

- kết nối database `elevator_cv`;
- khởi tạo schema;
- insert event;
- insert occupancy sample;
- query events;
- query density;
- load face embeddings.

### `app/event_logger.py`

Logger chuẩn hóa cho module CV.

Nhiệm vụ:

- ghi event vào `camera_events`;
- ghi occupancy sample vào `camera_occupancy_samples`.

Nó là lớp trung gian giữa pipeline CV và database.

### `app/tracker.py`

Tracker đơn giản để gán `track_id` cho các bbox người.

Nhiệm vụ:

- tính IoU;
- gắn detection mới vào track cũ;
- tạo track mới khi cần;
- xóa track cũ khi quá tuổi.

### `app/posture.py`

Suy luận posture cơ bản từ keypoints pose.

Nhiệm vụ:

- classify posture (`standing`, `lying`, `unknown`);
- phát hiện chuyển tiếp có thể coi là ngã.

### `app/face_recog.py`

Khung face recognition, hiện mặc định thường để tắt khi test Jetson cho nhẹ máy.

Nhiệm vụ:

- tạo `FaceAnalysis` nếu bật face;
- so khớp embedding với database;
- map face sang `person_id` / `person_name`.

### `app/yolo_utils.py`

Các utility chung cho inference.

Nhiệm vụ:

- preprocess frame;
- letterbox;
- parse output detect;
- parse output pose;
- scale bbox;
- NMS bằng NumPy.

### `app/runtime_trt.py`

Runtime chính cho Jetson Nano.

Nhiệm vụ:

- load `.engine` TensorRT;
- tạo execution context;
- infer detect;
- infer pose;
- xử lý memory copy H2D/D2H;
- cleanup CUDA resources.

Đây là file quan trọng nhất để chạy production trên Jetson.

### `app/runtime_ultra.py`

Runtime dành cho máy dev.

Nhiệm vụ:

- load model bằng `ultralytics.YOLO`;
- chạy `.predict(...)` cho detect/pose.

Mục đích:

- dev/test/export trên máy mạnh;
- không dùng làm runtime production chính trên Jetson.

### `app/camera_service.py`

Lõi pipeline realtime.

Nhiệm vụ:

- mở camera;
- khởi tạo runtime bên trong worker thread;
- chạy detect;
- chạy pose;
- tracking;
- tạo trạng thái realtime;
- ghi occupancy sample định kỳ;
- sinh event;
- encode MJPEG stream;
- expose status cho API.

Đây là file quan trọng nhất về logic CV thời gian thực.

### `app/api.py`

Backend web layer cho module CV.

Nhiệm vụ:

- khởi tạo FastAPI app;
- startup/shutdown service;
- dashboard test ở route `/`;
- status API;
- stream API;
- events API;
- density API.

---

## 5.3 Thư mục `scripts/`

### `scripts/export_models.py`

Chạy ở máy dev.

Nhiệm vụ:

- export `yolov8n.pt` -> `yolov8n.onnx`
- export `yolov8n-pose.pt` -> `yolov8n-pose.onnx`

### `scripts/build_engines.sh`

Chạy trên Jetson Nano.

Nhiệm vụ:

- build TensorRT engine từ ONNX;
- sinh:
  - `yolov8n_fp16.engine`
  - `yolov8n_pose_fp16.engine`

---

## 5.4 Thư mục `frontend/`

Đây là frontend mẫu/giữ lại từ bundle.

Không phải frontend tích hợp cuối cùng của project Elevator AI, nhưng có thể dùng để:

- tham khảo cấu hình Vite;
- test nhanh nếu muốn dựng UI tách riêng;
- giữ metadata Node của bundle CV.

Trong giai đoạn hiện tại, route `/` ngay trong FastAPI mới là giao diện test nhanh quan trọng hơn.

---

## 6. Luồng chạy của module CV

## 6.1 Luồng môi trường/model

1. Trên máy dev:
   - đặt `yolov8n.pt`, `yolov8n-pose.pt` vào `models/`
   - chạy `python scripts/export_models.py`
   - tạo file `.onnx`

2. Copy `.onnx` sang Jetson.

3. Trên Jetson:
   - chạy `bash scripts/build_engines.sh`
   - tạo TensorRT engine `.engine`

## 6.2 Luồng service runtime

1. `python3 main.py`
2. Uvicorn chạy `app.api`
3. `startup()` gọi:
   - `db.init_schema()`
   - `camera_service.start()`
4. `camera_service` tạo worker thread
5. Trong worker thread:
   - mở camera
   - build runtime TRT
   - đọc frame
   - infer detect/pose
   - tracking
   - update status
   - ghi occupancy
   - ghi event nếu có
   - mã hóa JPEG cho stream

## 6.3 Luồng API

- `/` -> dashboard test nhanh
- `/api/cv/status` -> trạng thái realtime
- `/api/cv/stream` -> MJPEG stream
- `/api/cv/events` -> danh sách sự kiện
- `/api/cv/density?days=N` -> dữ liệu density theo ngày

---

## 7. Chức năng hiện tại của module CV

### 7.1 Chức năng đang có

- đọc camera Jetson CSI hoặc USB;
- detect người;
- detect bottle;
- tracking theo `track_id`;
- pose detection (nếu bật);
- suy posture cơ bản;
- event cơ bản:
  - `CROWD`
  - `BOTTLE`
  - `FALL`
  - `LYING`
- stream MJPEG qua HTTP;
- dashboard test trực tiếp ở `/`;
- lưu occupancy samples;
- query density theo ngày;
- lưu face registry schema sẵn trong DB.

### 7.2 Chức năng đang ở mức khung/chưa hoàn chỉnh hoàn toàn

- face recognition production-ready;
- snapshot event tự động;
- timeline ảnh minh họa theo event;
- nhiều loại event nâng cao như `UNKNOWN_PERSON`, `OVERLOAD`, `UNATTENDED_OBJECT`;
- ghép frontend chính của Elevator AI vào live stream thật;
- chatbot SQL tool hoàn chỉnh để hỏi `elevator_cv`.

---

## 8. Database schema và ý nghĩa dữ liệu

## 8.1 `person_registry`

Mục đích:

- danh sách nhân sự/đối tượng đã biết;
- nối với face embedding;
- dùng cho face recognition sau này.

## 8.2 `face_embeddings`

Mục đích:

- lưu vector embedding của khuôn mặt;
- nối với `person_registry`.

## 8.3 `camera_events`

Mục đích:

- lưu sự kiện CV.

Các cột chính:

- `event_ts`
- `cam_id`
- `event_type`
- `track_id`
- `person_id`
- `person_name`
- `bbox`
- `posture`
- `people_count`
- `confidence`
- `snapshot_path`
- `extra`

Đây là bảng dùng cho timeline sự kiện và chatbot analytics.

## 8.4 `camera_occupancy_samples`

Mục đích:

- lưu mẫu mật độ người theo thời gian;
- làm nguồn cho density chart và câu hỏi dạng analytics.

Các cột chính:

- `sample_ts`
- `cam_id`
- `people_count`
- `unknown_count`
- `lying_count`
- `fall_count`
- `extra`

Đây là bảng cực kỳ quan trọng cho biểu đồ và câu hỏi như:

- "giờ nào đông nhất hôm nay?"
- "trung bình mỗi ngày có bao nhiêu người trước thang máy?"

---

## 9. Các bước chạy module CV

## 9.1 Tạo PostgreSQL riêng

```sql
CREATE USER elevator_ai WITH PASSWORD 'elevator123';
CREATE DATABASE elevator_cv OWNER elevator_ai;
CREATE DATABASE elevator_llm OWNER elevator_ai;
GRANT ALL PRIVILEGES ON DATABASE elevator_cv TO elevator_ai;
GRANT ALL PRIVILEGES ON DATABASE elevator_llm TO elevator_ai;
```

## 9.2 Chuẩn bị model

### Trên máy dev

1. đặt model:
   - `models/yolov8n.pt`
   - `models/yolov8n-pose.pt`
2. chạy:

```bash
python scripts/export_models.py
```

3. copy `.onnx` sang Jetson.

### Trên Jetson

```bash
bash scripts/build_engines.sh
```

## 9.3 Cài môi trường Jetson

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-psycopg2 build-essential python3-dev
python3 -m pip install --user -r requirements.txt
```

## 9.4 Nạp env CV

```bash
source .env.cv.example
```

Nếu dùng CSI camera Jetson, có thể dùng GStreamer pipeline.

## 9.5 Chạy service

```bash
python3 main.py
```

## 9.6 Test nhanh

```bash
curl http://127.0.0.1:8000/api/cv/status
curl http://127.0.0.1:8000/api/cv/events
curl "http://127.0.0.1:8000/api/cv/density?days=7"
```

Mở trình duyệt:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/api/cv/stream`

---

## 10. Cách ghép module CV này vào project Elevator AI

## 10.1 Tư duy tích hợp đúng

Không copy toàn bộ code CV rồi nhét vào backend chatbot hiện có.

Thay vào đó, tích hợp theo mô hình service:

- **CV service** chạy riêng
- **Project Elevator AI chính** gọi CV service qua HTTP + PostgreSQL

Kiến trúc ghép đúng:

1. **CV Realtime Engine**
   - module CV hiện tại
2. **PostgreSQL `elevator_cv`**
   - data nguồn thật của camera
3. **Backend API / Chatbot Agent của Elevator AI**
   - query `elevator_cv`
   - query `elevator_llm`
4. **Frontend Elevator AI chính**
   - nhúng stream, timeline, density, chatbot

## 10.2 Cách match module CV với project Elevator AI hiện có

### A. Frontend

Frontend project Elevator AI phải:

- thêm khu vực hiển thị camera live từ `http://<cv-host>:8000/api/cv/stream`
- thêm màn hình timeline bằng cách gọi `/api/cv/events`
- thêm biểu đồ density bằng cách gọi `/api/cv/density`

### B. Backend chatbot

Backend chatbot hiện có phải thêm một lớp SQL tool/query layer cho `elevator_cv`.

Ví dụ:

- `get_today_fall_count()`
- `get_peak_hour(cam_id)`
- `get_recent_cv_events(limit)`
- `get_daily_density(cam_id, days)`

### C. Database

- `elevator_cv` chỉ dành cho camera analytics.
- `elevator_llm` chỉ dành cho prompt, knowledge, FAQ, embeddings chatbot.

### D. Business logic

Luồng đúng cho chatbot:

1. nhận câu hỏi người dùng;
2. phân loại câu hỏi là **CV analytics** hay **LLM knowledge**;
3. nếu là CV analytics -> query SQL vào `elevator_cv`;
4. nếu là LLM knowledge -> query/retrieve từ `elevator_llm`;
5. format lại câu trả lời.

## 10.3 Ví dụ câu hỏi cần match sang `elevator_cv`

- "Hôm nay có bao nhiêu lần té ngã?"
- "Camera 1 đông nhất lúc nào?"
- "Có sự kiện nào bất thường trong 24 giờ qua?"
- "Hôm nay mật độ người cao nhất là mấy giờ?"

## 10.4 Ví dụ câu hỏi cần match sang `elevator_llm`

- "Hệ thống này dùng để làm gì?"
- "Cảnh báo crowd nghĩa là gì?"
- "Quy trình xử lý khi có cảnh báo fall là gì?"
- "Cách vận hành module chatbot trong đề tài thang máy AI"

---

## 11. Prompt tích hợp đề xuất cho chatbot khi đưa 2 file `.md`

Khi bạn gửi:

1. file `.md` về module CV này
2. file `.md` về project Elevator AI tổng thể

thì chatbot nên được yêu cầu làm theo logic sau:

### Mục tiêu của chatbot

- hiểu module CV hiện tại là một service con;
- xác định các điểm nối giữa CV service và project Elevator AI;
- đề xuất thay đổi nhỏ nhất để tích hợp thay vì viết lại toàn bộ;
- ưu tiên tái sử dụng API và database hiện có;
- giữ tách biệt `elevator_cv` và `elevator_llm`.

### Prompt gợi ý

```text
Đây là 2 tài liệu:
1. Tài liệu module CV hiện tại đang chạy trên Jetson Nano.
2. Tài liệu project Elevator AI tổng thể.

Hãy phân tích để:
- chỉ ra module CV hiện tại đang làm gì;
- chỉ ra project Elevator AI hiện tại đang có gì;
- map các điểm ghép giữa 2 hệ thống;
- đề xuất kiến trúc tích hợp sao cho module CV trở thành một phần của đề tài,
  không viết lại từ đầu nếu không cần;
- giữ database tách riêng: elevator_cv cho dữ liệu camera, elevator_llm cho chatbot/LLM;
- chỉ rõ backend nào gọi backend nào, frontend nào gọi API nào, chatbot nào query DB nào;
- cuối cùng tạo một kế hoạch thực hiện theo từng bước để tích hợp hoàn chỉnh.
```

---

## 12. Kế hoạch tạo file `.md` tổng hợp cuối cùng sau khi ghép xong

Sau khi module CV đã được gắn vào project Elevator AI, bạn nên tạo một file tổng hợp cuối cùng với cấu trúc như sau:

### 12.1 Phần A - Tổng quan hệ thống

- mục tiêu đề tài;
- kiến trúc tổng thể;
- các module chính.

### 12.2 Phần B - Module CV

- camera;
- detect/pose/track/face;
- events;
- density;
- API;
- database CV.

### 12.3 Phần C - Module Chatbot/LLM

- intent routing;
- semantic matcher;
- FAQ;
- embeddings;
- database LLM.

### 12.4 Phần D - Tích hợp

- frontend gọi API nào;
- backend gọi service nào;
- chatbot query DB nào;
- mapping route và dữ liệu.

### 12.5 Phần E - Hướng phát triển tiếp theo

- event nâng cao;
- snapshot event;
- face recognition production;
- timeline ảnh;
- dashboard đẹp hơn;
- SQL tool đầy đủ cho chatbot.

---

## 13. Kết luận ngắn gọn

Module CV hiện tại đã đạt mức **service CV chạy độc lập trên Jetson Nano**, có thể:

- chạy camera;
- infer bằng TensorRT;
- expose stream;
- cung cấp API status/events/density;
- ghi dữ liệu vào `elevator_cv`.

Để trở thành một phần của đề tài Elevator AI, module này cần được tích hợp theo mô hình:

- **CV service** độc lập;
- **`elevator_cv`** là nguồn dữ liệu camera;
- **`elevator_llm`** là nguồn dữ liệu chatbot/LLM;
- **chatbot** phải dùng SQL để đọc `elevator_cv`;
- **frontend Elevator AI** phải gọi API của CV service để hiển thị camera, timeline, density.

Đây là cách tích hợp đúng, sạch và phù hợp nhất để biến phần CV đang chạy riêng lẻ thành một phần chính thức của đề tài thang máy AI.
