FROM python:3.12
EXPOSE 8000
COPY ./starter.sh ./app/starter.sh
WORKDIR /app
CMD ["./starter.sh"]