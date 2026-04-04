# 1. Usamos la imagen base oficial de AWS Lambda para Python 3.13
FROM public.ecr.aws/lambda/python:3.13

# 2. Copiamos el archivo de dependencias a la ruta raíz de Lambda
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# 3. Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiamos todo el código fuente al contenedor
COPY . ${LAMBDA_TASK_ROOT}

# 5. Le indicamos a Lambda cuál es la función que debe ejecutar como punto de entrada
CMD [ "main.handler" ]