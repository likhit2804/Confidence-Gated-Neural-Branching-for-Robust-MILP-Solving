FROM condaforge/mambaforge:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MAMBA_NO_BANNER=1

WORKDIR /workspace

COPY docker/requirements.txt /tmp/requirements.txt
RUN conda install -y -c conda-forge \
    python=3.10 \
    ecole \
    pytorch \
    pyscipopt \
    scipy \
    pytest \
    && pip install torch-geometric \
    && ln -s /opt/conda/lib/libscip.so.10.1 /opt/conda/lib/libscip.so.10.0 \
    && conda clean -afy

COPY implementation /workspace/implementation

CMD ["python", "-m", "pytest", "implementation/tests", "-vv"]
