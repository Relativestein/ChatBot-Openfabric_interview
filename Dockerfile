FROM openfabric/tee-python-cpu:latest

RUN mkdir application
WORKDIR /application
COPY . .
RUN poetry install -vvv --no-dev
RUN pip install langchain
RUN pip install google-generativeai
RUN pip install wikipedia
EXPOSE 5500
CMD ["sh","start.sh"]