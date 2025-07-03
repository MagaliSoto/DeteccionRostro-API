"""
Este servidor FastAPI permite detectar y validar rostros humanos en imágenes recibidas mediante una API REST.
Utiliza los siguientes componentes:

- **InsightFace (buffalo_l)** para extraer embeddings faciales de alta calidad.
- **MediaPipe Pose** para analizar la orientación del cuerpo (frente, perfil, espaldas).
- **Galería por track_id** para comparar la similitud entre embeddings faciales de la misma persona en distintas capturas.
- Se actualiza y valida la galería si el rostro es consistente y la orientación corporal es similar.
- Permite detectar si un rostro nuevo coincide con uno previamente visto usando distancia de coseno.
- Filtra detecciones poco confiables según la orientación del cuerpo o mala visibilidad.

Endpoints:
- `/test_connection/` → Verifica si el servidor está activo.
- `/detectar_rostro/` → Recibe una imagen y un `track_id`, y responde con la caja del rostro detectado y su orientación, o `None` si no se valida.

Este sistema está diseñado para integrarse en flujos de análisis de video o seguimiento multi-cámara, donde se desea confirmar identidades a lo largo del tiempo.
"""

import cv2
import numpy as np
import mediapipe as mp
import insightface
from insightface.app import FaceAnalysis
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from numpy.linalg import norm
from typing import List

app = FastAPI()

# Modelo InsightFace
face_model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_model.prepare(ctx_id=0)

# Pose para orientación
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Galerías
gallery: dict[int, List[np.ndarray]] = {}
pose_gallery: dict[int, float] = {}  # Almacena la posición horizontal del cuerpo

# CORS
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

def mejorar_imagen(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img

def obtener_orientacion_y_pose(img: np.ndarray) -> tuple[str, float | None]:
    res = pose.process(img)
    if not res.pose_landmarks:
        return "desconocido", None
    lm = res.pose_landmarks.landmark
    nariz = lm[mp_pose.PoseLandmark.NOSE]
    izq  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    der  = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    if nariz.visibility < 0.5 or izq.visibility < 0.5 or der.visibility < 0.5:
        return "desconocido", None
    centro_h = (izq.x + der.x) / 2
    cuerpo_x = centro_h  # centro del torso
    if nariz.visibility < 0.2:
        return "espaldas", cuerpo_x
    if abs(nariz.x - centro_h) > 0.1:
        return "perfil", cuerpo_x
    return "frente", cuerpo_x

def distancia_coseno(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

def embedding_medio(embs: List[np.ndarray]) -> np.ndarray:
    m = np.mean(np.stack(embs), axis=0)
    return m / norm(m)

@app.get("/test_connection/")
def test_connection():
    return {"status": "Servidor activo y escuchando"}

@app.post("/detectar_rostro/")
async def detectar_rostro(
    track_id: int = Query(..., description="ID interno de tracking"),
    file: UploadFile = File(...),
):
    contenido = await file.read()
    img = cv2.imdecode(np.frombuffer(contenido, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = mejorar_imagen(img)

    # --- 1) Orientación y pose ---
    orient, cuerpo_x = obtener_orientacion_y_pose(img_rgb)
    if orient == "espaldas":
        return JSONResponse(content={"box": None, "orientacion": orient})

    # --- 2) Detección de caras ---
    faces = face_model.get(img_rgb)
    if not faces:
        return JSONResponse(content={"box": None, "orientacion": orient})

    # --- 3) Nueva identidad (primera vez) ---
    if track_id not in gallery:
        f0 = faces[0]
        gallery[track_id] = [f0.normed_embedding]
        if cuerpo_x is not None:
            pose_gallery[track_id] = cuerpo_x
        x1, y1, x2, y2 = map(int, f0.bbox)
        return JSONResponse(content={"box": [x1, y1, x2, y2], "orientacion": orient})

    # --- 4) Comparar con galería ---
    target = embedding_medio(gallery[track_id])
    best_face = None
    best_dist = float("inf")
    for f in faces:
        d = distancia_coseno(f.normed_embedding, target)
        if d < best_dist:
            best_dist = d
            best_face = f

    # Umbrales
    th_perfil = 0.6
    th_frente = 0.5  # más estricto que antes para evitar confusiones

    # --- 5) Lógica según orientación ---
    if orient == "perfil":
        if best_dist > th_perfil:
            return JSONResponse(content={"box": None, "orientacion": orient})
        x1, y1, x2, y2 = map(int, best_face.bbox)
        return JSONResponse(content={"box": [x1, y1, x2, y2], "orientacion": orient})

    elif orient == "frente":
        # Caso normal: solo aceptar si se parece
        if best_dist <= th_frente:
            accept = True
        else:
            # Caso especial: ¿el cuerpo está en el mismo lugar?
            prev_pose = pose_gallery.get(track_id)
            accept = cuerpo_x is not None and prev_pose is not None and abs(cuerpo_x - prev_pose) < 0.05

        if accept:
            x1, y1, x2, y2 = map(int, best_face.bbox)
            gallery[track_id].append(best_face.normed_embedding)
            if len(gallery[track_id]) > 5:
                gallery[track_id].pop(0)
            if cuerpo_x is not None:
                pose_gallery[track_id] = cuerpo_x
            return JSONResponse(content={"box": [x1, y1, x2, y2], "orientacion": orient})
        else:
            return JSONResponse(content={"box": None, "orientacion": orient})

    # Orientación desconocida
    return JSONResponse(content={"box": None, "orientacion": orient})


if __name__ == "__main__":
    #conda activate servP
    #python DeteccionPersonasServer/server.py
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
