from typing import List, Optional, Union, Tuple
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import logging
import time
import threading

RATE_LIMIT_WINDOW = 60  

RATE_LIMIT_MAX = 120    

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("matrix_api")

app = FastAPI(title="Matrix Calculator API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  

    allow_credentials=True,
    allow_methods=["*"],  

    allow_headers=["*"],
)

_rate_lock = threading.Lock()
_rate_store = {}  

class MatrixModel(BaseModel):
    matrix: List[List[float]]

    @validator("matrix")
    def non_empty_and_rectangular(cls, v):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("matrix must be a non-empty 2D list")
        row_len = len(v[0])
        if row_len == 0:
            raise ValueError("matrix rows must be non-empty")
        for row in v:
            if not isinstance(row, list):
                raise ValueError("matrix must be a 2D list")
            if len(row) != row_len:
                raise ValueError("all rows in the matrix must have the same length")
            for x in row:
                if not isinstance(x, (int, float)):
                    raise ValueError("matrix must contain only numeric values")
        return v

class TwoMatricesModel(BaseModel):
    A: List[List[float]]
    B: List[List[float]]

    @validator("A", "B")
    def validate_matrix(cls, v):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("matrix must be a non-empty 2D list")
        row_len = len(v[0])
        if row_len == 0:
            raise ValueError("matrix rows must be non-empty")
        for row in v:
            if not isinstance(row, list) or len(row) != row_len:
                raise ValueError("all rows in the matrix must have the same length")
            for x in row:
                if not isinstance(x, (int, float)):
                    raise ValueError("matrix must contain only numeric values")
        return v

class ScalarModel(BaseModel):
    A: List[List[float]]
    scalar: float

    @validator("A")
    def validate_matrix(cls, v):

        if not v or not isinstance(v, list) or not isinstance(v[0], list):
            raise ValueError("A must be a non-empty 2D list")
        row_len = len(v[0])
        for row in v:
            if len(row) != row_len:
                raise ValueError("all rows in A must have the same length")
            for x in row:
                if not isinstance(x, (int, float)):
                    raise ValueError("A must contain only numeric values")
        return v

class SolveModel(BaseModel):
    A: List[List[float]]
    b: Union[List[float], List[List[float]]]

    @validator("A")
    def validate_A(cls, v):
        if not v or not isinstance(v, list) or not isinstance(v[0], list):
            raise ValueError("A must be a non-empty 2D list")
        row_len = len(v[0])
        for row in v:
            if len(row) != row_len:
                raise ValueError("all rows in A must have the same length")
            for x in row:
                if not isinstance(x, (int, float)):
                    raise ValueError("A must contain only numeric values")
        return v

    @validator("b")
    def validate_b(cls, v):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("b must be a non-empty list or 2D list")

        if isinstance(v[0], list):

            col_len = len(v[0])
            for row in v:
                if not isinstance(row, list) or len(row) != col_len:
                    raise ValueError("b must be a rectangular 2D list if provided as 2D")
                for x in row:
                    if not isinstance(x, (int, float)):
                        raise ValueError("b must contain only numeric values")
        else:

            for x in v:
                if not isinstance(x, (int, float)):
                    raise ValueError("b must contain only numeric values")
        return v

def to_numpy(mat: List[List[float]]) -> np.ndarray:
    try:
        arr = np.array(mat, dtype=float)
    except Exception as e:
        raise ValueError("could not convert matrix to numeric array: %s" % e)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    return arr

def to_list(arr: np.ndarray) -> List:

    if np.isscalar(arr):
        return float(arr)
    return arr.tolist()

def standard_response(result=None, error: Optional[str] = None):
    return JSONResponse(status_code=200 if error is None else 400, content={
        "success": error is None,
        "result": result if error is None else None,
        "error": error
    })

@app.middleware("http")
async def log_and_rate_limit(request: Request, call_next):
    start = time.time()
    client_ip = request.client.host if request.client else "unknown"

    now = time.time()
    with _rate_lock:
        timestamps = _rate_store.get(client_ip, [])

        timestamps = [t for t in timestamps if t > now - RATE_LIMIT_WINDOW]
        if len(timestamps) >= RATE_LIMIT_MAX:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS, content={
                "success": False,
                "result": None,
                "error": "rate limit exceeded"
            })
        timestamps.append(now)
        _rate_store[client_ip] = timestamps

    try:
        body = await request.json()
    except Exception:
        body = None
    logger.info(f"{client_ip} -> {request.method} {request.url.path} body={body}")

    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception("Unhandled error during request")
        raise
    finally:
        elapsed = (time.time() - start) * 1000
        logger.info(f"{client_ip} <- {request.method} {request.url.path} done in {elapsed:.1f}ms")

    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={
        "success": False,
        "result": None,
        "error": exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    })

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unexpected error")
    return JSONResponse(status_code=500, content={
        "success": False,
        "result": None,
        "error": "internal server error"
    })

@app.post("/add")
async def add_matrices(payload: TwoMatricesModel):
    A = to_numpy(payload.A)
    B = to_numpy(payload.B)
    if A.shape != B.shape:
        raise HTTPException(status_code=400, detail="matrices must have the same shape")
    C = A + B
    return standard_response(result=to_list(C))

@app.post("/subtract")
async def subtract_matrices(payload: TwoMatricesModel):
    A = to_numpy(payload.A)
    B = to_numpy(payload.B)
    if A.shape != B.shape:
        raise HTTPException(status_code=400, detail="matrices must have the same shape")
    C = A - B
    return standard_response(result=to_list(C))

@app.post("/multiply")
async def multiply_matrices(payload: TwoMatricesModel):
    A = to_numpy(payload.A)
    B = to_numpy(payload.B)
    if A.shape[1] != B.shape[0]:
        raise HTTPException(status_code=400, detail="A.columns must equal B.rows for multiplication")
    C = A.dot(B)
    return standard_response(result=to_list(C))

@app.post("/scalar")
async def scalar_multiply(payload: ScalarModel):
    A = to_numpy(payload.A)
    s = float(payload.scalar)
    C = A * s
    return standard_response(result=to_list(C))

@app.post("/transpose")
async def transpose_matrix(payload: MatrixModel):
    A = to_numpy(payload.matrix)
    return standard_response(result=to_list(A.T))

@app.post("/determinant")
async def determinant_matrix(payload: MatrixModel):
    A = to_numpy(payload.matrix)
    if A.shape[0] != A.shape[1]:
        raise HTTPException(status_code=400, detail="matrix must be square to compute determinant")
    try:
        det = float(np.linalg.det(A))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"error computing determinant: {e}")
    return standard_response(result=det)

@app.post("/inverse")
async def inverse_matrix(payload: MatrixModel):
    A = to_numpy(payload.matrix)
    if A.shape[0] != A.shape[1]:
        raise HTTPException(status_code=400, detail="matrix must be square to compute inverse")
    try:
        inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise HTTPException(status_code=400, detail="matrix is singular (non-invertible)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return standard_response(result=to_list(inv))

@app.post("/rank")
async def rank_matrix(payload: MatrixModel):
    A = to_numpy(payload.matrix)
    r = int(np.linalg.matrix_rank(A))
    return standard_response(result=r)

@app.post("/solve")
async def solve_system(payload: SolveModel):
    A = to_numpy(payload.A)
    b_raw = payload.b

    if isinstance(b_raw[0], list):
        b = np.array(b_raw, dtype=float)
    else:
        b = np.array(b_raw, dtype=float).reshape((-1, 1))

    if A.shape[0] != b.shape[0]:
        raise HTTPException(status_code=400, detail="A.rows must equal b.rows")

    try:

        if A.shape[0] == A.shape[1]:
            detA = float(np.linalg.det(A))
            if abs(detA) > 1e-12:
                x = np.linalg.solve(A, b)
                return standard_response(result=to_list(x))
            else:

                x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                return standard_response(result={
                    "least_squares": True,
                    "solution": to_list(x),
                    "residuals": to_list(residuals),
                    "rank": int(rank)
                })
        else:

            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            return standard_response(result={
                "least_squares": True,
                "solution": to_list(x),
                "residuals": to_list(residuals),
                "rank": int(rank)
            })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"error solving system: {e}")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/info")
async def info():
    return {
        "name": "Matrix Calculator API",
        "version": "1.0",
        "endpoints": [
            "/add", "/subtract", "/multiply", "/scalar",
            "/transpose", "/determinant", "/inverse", "/rank", "/solve"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("matrix_api:app", host="0.0.0.0", port=8000, reload=True)

