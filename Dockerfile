FROM python:3.12
COPY ./starter.sh ./app/starter.sh
WORKDIR /app
CMD ["./starter.sh"]