FROM python:3.13-slim
EXPOSE 8000
COPY ./starter.sh ./app/starter.sh
WORKDIR /app
CMD ["./starter.sh"]