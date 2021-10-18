web: gunicorn app:app --log-level=debug
worker: celery  -A  app.celery worker --beat 