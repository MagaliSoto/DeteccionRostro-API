# 👁️‍🗨️ Detección y Validación de Rostros - API FastAPI

Este repositorio contiene un **servidor FastAPI** que permite detectar, validar y comparar rostros humanos en imágenes recibidas mediante una API REST.  
Está diseñado para integrarse en sistemas de **análisis de video en tiempo real** o **seguimiento multi-cámara**.

---

## ⚙️ Características principales

- Detección facial con **InsightFace (buffalo_l)**.  
- Análisis de orientación corporal con **MediaPipe Pose** (frente, perfil o espaldas).  
- Comparación de rostros por **track_id** usando distancia de coseno.  
- Validación de consistencia entre capturas (pose y similitud).  
- Filtro de detecciones poco confiables según visibilidad o pose.

---

## 🚀 Endpoints

| Método | Ruta | Descripción |
|:-------|:------|:-------------|
| `GET` | `/test_connection/` | Verifica si el servidor está activo. |
| `POST` | `/detectar_rostro/` | Recibe una imagen y un `track_id`, devuelve la caja del rostro detectado y su orientación. |

---

## 🧩 Dependencias principales

- `fastapi`
- `uvicorn`
- `insightface`
- `mediapipe`
- `opencv-python`
- `numpy`

---

## ▶️ Ejecución

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
````

2. Ejecutar el servidor:

   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

3. Probar conexión:

   ```bash
   curl http://localhost:8000/test_connection/
   ```

---

## 📦 Descripción rápida del archivo

**`server.py`**
Contiene toda la lógica de:

* Carga de modelos (InsightFace y MediaPipe).
* Gestión de galerías de embeddings faciales.
* Cálculo de similitud entre rostros.
* Validación de orientación corporal.
* Exposición de endpoints de la API.

---

📍 **Autor:** *Magali Soto*
🧠 **Propósito:** detección y validación de rostros en flujos de video.

