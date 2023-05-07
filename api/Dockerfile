FROM python:3.9-slim 

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./



RUN pip install --no-cache-dir -r requirements.txt 

RUN python -c "import nltk; nltk.download('omw-1.4'); nltk.download('wordnet')"

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app