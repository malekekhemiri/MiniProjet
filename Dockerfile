FROM python:3.8

WORKDIR /app

COPY ./app/svm_service.py /app/
COPY ./app/vgg_service.py /app/
COPY ./app/svm_model.pkl /app/
COPY ./app/vgg_model.pb /app/

RUN pip install Flask flask-cors scikit-learn joblib librosa tensorflow

EXPOSE 5001 5002

CMD ["python", "svm_service.py", "vgg_service.py"]  