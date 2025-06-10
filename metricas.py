import matplotlib.pyplot as plt


metricas = ['BLEU', 'ROUGE', 'BERTScore', 'Cosine Similarity']
valores = [0.78, 0.82, 0.70, 0.88] 

plt.figure(figsize=(8, 5))
plt.bar(metricas, valores, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
plt.ylim(0, 1)
plt.title('Evaluación de Modelo LLaMA 3.1 – Métricas de Generación')
plt.ylabel('Puntaje (0 - 1)')
plt.xlabel('Métrica')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(valores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.show()
