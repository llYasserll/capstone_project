from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

texto_tesis = """
Esta investigación tiene como finalidad analizar el impacto del comercio electrónico en las pymes de Lima durante 2024. Se plantea como hipótesis que la adopción de plataformas digitales mejora significativamente las ventas. Las variables consideradas son: adopción digital, volumen de ventas, y satisfacción del cliente. Finalmente, se concluye que el comercio electrónico incrementa la competitividad de las pymes.
"""

prompt = f"""
Extrae un objeto JSON con las siguientes claves: 
- "Formulación del problema"
- "Objetivos generales"
- "Hipótesis"
- "Variables"
- "Conclusiones"

Texto:
\"\"\"{texto_tesis}\"\"\"

Devuelve **solo el JSON** sin explicaciones.
"""

print("⏳ Generando...")
result = pipe(prompt, max_new_tokens=512, do_sample=False, temperature=0.7)[0]["generated_text"]

# Extraer solo la parte que parezca JSON
try:
    json_start = result.find("{")
    json_text = result[json_start:]
    parsed = json.loads(json_text)
    print("\n✅ JSON extraído:\n")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
except Exception as e:
    print("⚠️ Error extrayendo JSON:")
    print(result)
