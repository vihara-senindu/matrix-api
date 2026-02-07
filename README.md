# 🧮 Matrix Calculator API

A clean, production-ready **REST API** for performing common **matrix operations** such as addition, subtraction, multiplication, determinant, inverse, rank, transpose, scalar multiplication, and solving linear systems.

Built with **FastAPI** and **NumPy**, this API is designed to be:

* easy to integrate with frontend apps (React, Vue, etc.)
* mathematically correct
* well-validated
* developer-friendly

---

## 🚀 Features

* Matrix addition, subtraction, multiplication
* Scalar multiplication
* Transpose
* Determinant
* Inverse
* Rank
* Solve linear systems (Ax = b)
* Strong input validation
* Consistent JSON responses
* CORS-enabled (frontend friendly)
* Interactive Swagger documentation

---

## 🛠️ Tech Stack

* **Backend:** FastAPI
* **Math Engine:** NumPy
* **Server:** Uvicorn
* **Language:** Python 3.9+

---

## 📦 Installation

### 1️⃣ Clone or create project folder

```bash
mkdir matrix-api
cd matrix-api
```

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

* **Windows**

```bash
venv\Scripts\activate
```

* **macOS / Linux**

```bash
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install fastapi uvicorn numpy
```

---

## ▶️ Run the API

```bash
uvicorn matrix_api:app --reload --port 8000
```

API will be available at:

* **Base URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📘 API Response Format

All endpoints return a consistent JSON structure:

```json
{
  "success": true,
  "result": null,
  "error": null
}
```

* `success` → operation status
* `result` → calculation output
* `error` → error message (if any)

---

## 📐 API Endpoints

### ➕ Add Matrices

**POST** `/add`

```json
{
  "A": [[1,2],[3,4]],
  "B": [[5,6],[7,8]]
}
```

---

### ➖ Subtract Matrices

**POST** `/subtract`

```json
{
  "A": [[5,6],[7,8]],
  "B": [[1,2],[3,4]]
}
```

---

### ✖ Multiply Matrices

**POST** `/multiply`

```json
{
  "A": [[1,2],[3,4]],
  "B": [[5,6],[7,8]]
}
```

---

### 🔢 Scalar Multiplication

**POST** `/scalar`

```json
{
  "A": [[1,2],[3,4]],
  "scalar": 3
}
```

---

### 🔁 Transpose

**POST** `/transpose`

```json
{
  "matrix": [[1,2,3],[4,5,6]]
}
```

---

### 🧮 Determinant

**POST** `/determinant`

```json
{
  "matrix": [[1,2],[3,4]]
}
```

---

### 🔄 Inverse

**POST** `/inverse`

```json
{
  "matrix": [[4,7],[2,6]]
}
```

> ⚠️ Matrix must be square and non-singular

---

### 📊 Rank

**POST** `/rank`

```json
{
  "matrix": [[1,2],[2,4]]
}
```

---

### 📐 Solve Linear System (Ax = b)

**POST** `/solve`

```json
{
  "A": [[2,1],[1,3]],
  "b": [1,2]
}
```

Returns solution vector or least-squares solution if applicable.

---

## ❌ Error Handling

Example error response:

```json
{
  "success": false,
  "result": null,
  "error": "matrix must be square"
}
```

Common errors handled:

* incompatible matrix sizes
* non-invertible matrices
* invalid input formats

---

## 🌐 CORS Support

CORS is enabled to allow frontend applications (React, Vue, etc.) to consume the API.

---

## 🧪 Testing

You can test all endpoints using:

* Swagger UI → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* `curl`
* Postman
* Your Matrix Calculator frontend

---

## 🚀 Deployment Tips

For production:

* Use **Gunicorn + Uvicorn workers**
* Add **API authentication** if public
* Add **rate limiting (Redis)**
* Use **Docker** for portability

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👨‍💻 Author

Matrix Calculator API — built for educational and production use.

If you want:

* Express.js version
* Docker setup
* Cloud deployment (Railway / Render)
* API authentication

Just ask 👍
