# PlagiarismDetection

**Profesorado:** 
- Miguel González Mendoza
- Raúl Monroy Borja
- Ariel Ortíz Ramírez
- Jorge Adolfo Ramírez Uresti

**Materia:** Desarrollo de aplicaciones avanzadas de ciencias computacionales

**Fecha de entrega:** 26 de mayo del 2024

**Integrantes del equipo 9:**

- Eduardo Joel Cortez Valente A01746664
- Fernando Ortiz Saldaña A01376737
- Maximiliano Benítez Ahumada A01752791

## Requisitos

- Python 3.6 o superior
- Las siguientes bibliotecas de Python:
  - `nltk`
  - `scikit-learn`
  - `logging`

## Instalación

1. Clona este repositorio en tu máquina local.
2. Instala las dependencias necesarias:

```bash
pip3 -r install requirements.txt
```

3. Descarga los recursos necesarios de NLTK:

```bash
nltk.download('punkt')
nltk.download('wordnet')
```

## Uso

**Inicialización**

Crea una instancia de la clase similarityCalculation especificando la ruta a la carpeta con archivos .txt y el umbral de similitud para considerar un texto como plagio.

```python
from similarity_calculation import similarityCalculation

# Ruta a la carpeta con archivos .txt y umbral de similitud (por ejemplo, 0.8 para 80%)
calc = similarityCalculation('DataBase/', 0.8)
```

**Detección de Plagio**

Para detectar plagio, utiliza el método plagiarismDetection pasando la ruta del archivo que deseas verificar.

```python
is_plagiarism, most_similar_file, similarity_score = calc.plagiarismDetection('path/to/input_file.txt')

if is_plagiarism:
    print(f'Plagio detectado. Archivo más similar: {most_similar_file} con una similitud de {similarity_score}')
else:
    print('No se detectó plagio.')
```

Puedes ver un ejemplo del funcionamiento en el archivo `main.py`
