# backend/app/services/eigenfaces.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Optional, Dict
import io
import numpy as np
from sqlalchemy.orm import Session
from sklearn.decomposition import PCA

from app.models.eigenmodel import EigenModel
from app.models.embedding import FaceEmbedding

# ---------- Numpy <-> bytes helpers ----------

def np_to_bytes(arr: np.ndarray) -> bytes:
    with io.BytesIO() as buf:
        np.save(buf, arr, allow_pickle=False)
        return buf.getvalue()

def bytes_to_np(b: bytes) -> np.ndarray:
    with io.BytesIO(b) as buf:
        buf.seek(0)
        return np.load(buf, allow_pickle=False)

# ---------- Fit / Project (scikit-learn) ----------

def fit_eigenfaces(images_flat: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Train PCA (Eigenfaces) using scikit-learn from flattened grayscale faces.
    images_flat: (N, D)
    Returns: (mean[D], components[K, D], pca_obj)
    """
    if images_flat.ndim != 2:
        raise ValueError("images_flat must be 2D (N, D)")
    N, _ = images_flat.shape
    if N < n_components:
        raise ValueError(f"Need at least n_components={n_components} samples, got {N}")

    # sklearn PCA centers internally; we still keep mean_ and components_ for storage
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=False, random_state=0)
    pca.fit(images_flat.astype(np.float32, copy=False))

    mean = pca.mean_.astype(np.float32)
    components = pca.components_.astype(np.float32)  # (K, D)
    return mean, components, pca

def project(vec_flat: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """
    Project one flattened face into eigen-space using stored mean & components.
    vec_flat: (D,), mean: (D,), components: (K, D) -> (K,)
    Equivalent to sklearn PCA.transform for a single sample.
    """
    x = vec_flat.astype(np.float32, copy=False)
    return ((x - mean) @ components.T).astype(np.float32)

# ---------- Persist / Load model ----------

def save_eigenmodel(db: Session, mean: np.ndarray, components: np.ndarray, n_components: int) -> EigenModel:
    """
    Upsert a single row eigen-model (id = 1 by convention) to EigenModel table.
    """
    payload = {
        "mean": np_to_bytes(mean.astype(np.float32, copy=False)),
        "components": np_to_bytes(components.astype(np.float32, copy=False)),
        "n_components": int(n_components),
    }
    model = db.query(EigenModel).filter(EigenModel.id == 1).one_or_none()
    if model is None:
        model = EigenModel(id=1, **payload)
        db.add(model)
    else:
        model.mean = payload["mean"]
        model.components = payload["components"]
        model.n_components = payload["n_components"]
    db.commit()
    db.refresh(model)
    return model

def load_latest_eigenmodel(db: Session) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load most recent eigen-model. Returns (mean[D], components[K,D], K).
    """
    model = (
        db.query(EigenModel)
        .order_by(EigenModel.created_at.desc())
        .first()
    )
    if model is None:
        raise RuntimeError("No EigenModel found. Train it first.")
    mean = bytes_to_np(model.mean).astype(np.float32)
    components = bytes_to_np(model.components).astype(np.float32)
    return mean, components, int(model.n_components)

# ---------- Gallery & Matching ----------

def load_gallery_embeddings(db: Session) -> List[Tuple[int, np.ndarray]]:
    rows = (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.method == "eigenfaces")
        .all()
    )
    gallery: List[Tuple[int, np.ndarray]] = []
    for r in rows:
        vec = bytes_to_np(r.vector).astype(np.float32)
        gallery.append((int(r.user_id) if r.user_id is not None else -1, vec))
    return gallery

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def nearest_by_euclidean(
    query_vec: np.ndarray,
    gallery: List[Tuple[int, np.ndarray]],
) -> Tuple[Optional[int], float]:
    if not gallery:
        return None, float("inf")
    best_uid, best_d = None, float("inf")
    for uid, gvec in gallery:
        d = euclidean(query_vec, gvec)
        if d < best_d:
            best_uid, best_d = uid, d
    return best_uid, best_d

def nearest_by_cosine(
    query_vec: np.ndarray,
    gallery: List[Tuple[int, np.ndarray]],
) -> Tuple[Optional[int], float]:
    if not gallery:
        return None, float("-inf")
    best_uid, best_s = None, float("-inf")
    for uid, gvec in gallery:
        s = cosine_sim(query_vec, gvec)
        if s > best_s:
            best_uid, best_s = uid, s
    return best_uid, best_s

# ---------- Thresholding / Confidence ----------

def distance_to_confidence(dist: float, lo: float, hi: float) -> float:
    x = (dist - lo) / max(hi - lo, 1e-6)
    conf = 1.0 - np.clip(x, 0.0, 1.0)
    return float(conf)

def decide_eigenfaces(
    score: float,
    mode: str = "euclidean",
    threshold: float = 300.0,
) -> bool:
    if mode == "euclidean":
        return score <= threshold
    elif mode == "cosine":
        return score >= threshold
    else:
        raise ValueError("mode must be 'euclidean' or 'cosine'")

# ---------- High-level helpers you’ll call from your API ----------

def train_and_persist(
    db: Session,
    all_user_images_flat: Dict[int, List[np.ndarray]],
    n_components: int = 64,
) -> None:
    """
    Fit sklearn PCA on ALL training images, store model, then compute & upsert per-user embeddings
    (average of that user's projected samples).
    - all_user_images_flat: {user_id: [vec_flat, ...]} where vec_flat is (D,)
    """
    stack = np.vstack([np.stack(imgs, axis=0) for imgs in all_user_images_flat.values()])
    mean, comps, _ = fit_eigenfaces(stack, n_components)
    save_eigenmodel(db, mean, comps, n_components)

    for uid, samples in all_user_images_flat.items():
        proj_list = [project(vec_flat, mean, comps) for vec_flat in samples]
        user_vec = np.mean(np.stack(proj_list, axis=0), axis=0).astype(np.float32)
        upsert_face_embedding(db, uid, user_vec, method="eigenfaces", dim=n_components)

def upsert_face_embedding(db: Session, user_id: int, vec: np.ndarray, method: str, dim: int) -> None:
    vec_bytes = np_to_bytes(vec.astype(np.float32, copy=False))
    row = (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.user_id == user_id, FaceEmbedding.method == method)
        .one_or_none()
    )
    if row is None:
        db.add(FaceEmbedding(user_id=user_id, method=method, dim=dim, vector=vec_bytes))
    else:
        row.dim = dim
        row.vector = vec_bytes
    db.commit()

def recognize_eigenfaces(
    db: Session,
    query_vec_flat: np.ndarray,
    mode: str = "euclidean",
    threshold: float = 300.0,
    conf_band: Tuple[float, float] = (150.0, 600.0),
) -> Tuple[Optional[int], float, bool]:
    mean, comps, _ = load_latest_eigenmodel(db)
    q = project(query_vec_flat, mean, comps)

    gallery = load_gallery_embeddings(db)
    if mode == "euclidean":
        uid, score = nearest_by_euclidean(q, gallery)
    elif mode == "cosine":
        uid, score = nearest_by_cosine(q, gallery)
    else:
        raise ValueError("mode must be 'euclidean' or 'cosine'")

    accepted = decide_eigenfaces(score, mode=mode, threshold=threshold)
    return uid if accepted else None, score, accepted
