from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "meta-llama/Llama-3.1-8B"

# Carga tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",         # Para usar GPU automáticamente
    torch_dtype=torch.float16, # Reduce memoria usando 16 bits
    low_cpu_mem_usage=True
)

# Pipeline para generación de texto fácil
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


# Prueba generando texto
prompt = "cuanto es 2+2"

result = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)

print(result[0]['generated_text'])
