FROM python:3.12
COPY ./start.sh ./app/start.sh
WORKDIR /app
CMD ["./start.sh"]