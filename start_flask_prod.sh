eval "$(conda shell.bash hook)"
conda activate aloe
gunicorn -w 4 --bind 0.0.0.0:8082 wsgi

