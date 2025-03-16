FROM python:3.12
#COPY . ./app
RUN mkdir /app
WORKDIR /app
RUN git clone https://github.com/Botusau/lt_dombo.git
CMD ["./start.sh"]