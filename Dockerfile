FROM python:3.12
EXPOSE 8000
RUN mkdir -p /data/models /data/nlp_cache
COPY ./starter.sh ./app/starter.sh
WORKDIR /app
CMD ["./starter.sh"]