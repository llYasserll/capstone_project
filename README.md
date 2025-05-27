## 🚀 Setup del entorno virtual (Windows)

### 1. Crear entorno virtual
```bash
py -m venv venv
# O si tienes múltiples versiones de Python instaladas:
py -3.12 -m venv venv
```

### 2. Activar entorno virtual
```bash
venv\Scripts\activate
```

---

## ⚙️ Instalación de dependencias

### Si tienes GPU NVIDIA (CUDA 12.1 compatible):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Si NO tienes GPU (solo CPU):
```bash
pip install torch torchvision torchaudio
```

### Instalar Accelerate
```bash
pip install accelerate
```

### Instalar Transformers
```bash
pip install transformers
```

---

## 🔐 Autenticación con Hugging Face

1. Crea una cuenta en [huggingface.co](https://huggingface.co/)
2. Ve a [Access Tokens](https://huggingface.co/settings/tokens) y crea un token personal
3. Inicia sesión desde la terminal con:
```bash
huggingface-cli login
```
Pega tu token cuando lo solicite.

---

### Instalar FastAPI y Uvicorn
```bash
pip install uvicorn fastapi
```

---

## 🧪 Uso del modelo

Ejecuta el script principal con:

```bash
python test.py
```

### Para consumir la API usa el comando y envia los parámetros
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Asegúrate de que el modelo descargado sea compatible con tu sistema (CUDA o CPU) y de haber iniciado sesión correctamente con Hugging Face.

---


