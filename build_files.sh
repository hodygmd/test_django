# Asegúrate de que pip esté instalado
python3.9 -m ensurepip

# Actualiza pip a la última versión
python3.9 -m pip install --upgrade pip

# Instala las dependencias desde requirements.txt
python3.9 -m pip install -r requirements.txt

# Recopila los archivos estáticos
python3.9 manage.py collectstatic --noinput
