import os
import json
import pdfplumber
import time
from mistralai import Mistral


api_key = "bx9is5zsh1M9imucHxVdogfG0cdqQ8HN" 
model = "mistral-small-latest"               

client = Mistral(api_key=api_key)

input_dir = r"C:\Users\Yasser\Desktop\capstone_project\tesis"  # Carpeta con PDFs
output_file = "dataset_preprocesado_mistral.jsonl"             # Dataset final

# ======= Extraer texto de PDF =======
def extraer_texto_pdf(ruta_pdf):
    texto = ""
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            contenido = pagina.extract_text()
            if contenido:
                texto += contenido + "\n"
    return texto

# ======= Preprocesar texto con Mistral API oficial =======
def preprocesar_con_mistral(texto_tesis):
    prompt = f"""
Clasifica el siguiente texto de tesis en formato JSON con las claves:
"problema", "objetivos", "hipótesis", "variables", "conclusiones".

Texto:
\"\"\"{texto_tesis}\"\"\"

Devuelve solo un objeto JSON válido.
"""

    try:
        chat_response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        print(f" Error API Mistral: {e}")
        return None

# ======= Procesar todos los PDFs =======
dataset = []

for archivo in os.listdir(input_dir):
    if archivo.endswith(".pdf"):
        ruta_pdf = os.path.join(input_dir, archivo)
        print(f"📄 Procesando archivo: {archivo}")

        texto_tesis = extraer_texto_pdf(ruta_pdf)

        if len(texto_tesis.strip()) < 100:
            print(" Archivo vacío o inválido, ignorando.")
            continue

        resultado_mistral = preprocesar_con_mistral(texto_tesis)
        
        if resultado_mistral:
            dataset.append({
                "input": texto_tesis,
                "output": resultado_mistral
            })
            print("✅ Procesado correctamente.")
        
        time.sleep(1)  # Pausa para respetar límites de API

# ======= Guardar dataset en JSONL =======
with open(output_file, "w", encoding="utf-8") as f_out:
    for ejemplo in dataset:
        json.dump(ejemplo, f_out, ensure_ascii=False)
        f_out.write("\n")

print(f" Dataset final guardado en: {output_file}")
