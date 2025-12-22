FROM python:3.14.2-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && apt-get clean

# setup user
RUN groupadd -r diamodel \
    && useradd -ms /bin/bash -g diamodel -G sudo diamodel
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN chown -R diamodel /home/diamodel/

USER diamodel
ENV HOME=/home/diamodel
WORKDIR /home/diamodel/diamodel

RUN curl -fsSL https://claude.ai/install.sh | bash
ENV CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

# install dependencies
COPY --chown=diamodel ./pyproject.toml ./
ENV PATH=$HOME/.local/bin:$PATH
RUN pip install ".[dev]" && \
    python -c "import cmdstanpy; cmdstanpy.install_cmdstan(cores=4)"

COPY --chown=diamodel . .

# switch to editable model
RUN pip install -e ".[dev]" --no-deps
