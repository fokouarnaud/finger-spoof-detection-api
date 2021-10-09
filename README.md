# finger-spoof-detection-api
pip list --format=freeze > requirements.txt

celery   -A  app.celery worker  --loglevel=info

heroku ps:scale  worker=1