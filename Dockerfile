FROM python:3.9-slim as base

FROM base AS python-deps

RUN pip install pipenv

COPY Pipfile .
COPY Pipfile.lock .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy

FROM base AS runtime

COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

WORKDIR /app

COPY covid19app.py covid19model.h5 ./
COPY /ref_images ./ref_images

EXPOSE 3500

CMD ["streamlit", "run", "covid19app.py", "--server.port", "3500", "--server.address", "0.0.0.0"]