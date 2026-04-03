# 1. Usamos una versión ligera de Python 3.13 (la que usaste en tu Mac)
FROM python:3.13-slim

# 2. Definimos la carpeta de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiamos solo el archivo de requerimientos primero (optimiza el caché de Docker)
COPY requirements.txt .

# 4. Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiamos todo el resto de tu código al contenedor
COPY . .

# 6. Comando por defecto al encender el contenedor
CMD ["python", "main.py"]