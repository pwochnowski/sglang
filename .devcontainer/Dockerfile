FROM lmsysorg/sglang:dev

# Create non-root user with specified UID and GID
# NOTE: Replace with your own UID and GID. This is a workaround from https://github.com/microsoft/vscode-remote-release/issues/49#issuecomment-489060908.
ARG HOST_UID=1003
ARG HOST_GID=1003
RUN groupadd -g $HOST_GID devuser && \
    useradd -m -u $HOST_UID -g $HOST_GID -s /bin/zsh devuser

# Give devuser sudo access
RUN apt-get update && apt-get install -y sudo && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/devuser && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Set up oh-my-zsh for devuser
RUN for f in .oh-my-zsh .zshrc .vimrc .tmux.conf; do \
        [ -e /root/$f ] && cp -r /root/$f /home/devuser/$f; \
    done && \
    [ -f /home/devuser/.zshrc ] && sed -i 's|/root/.oh-my-zsh|/home/devuser/.oh-my-zsh|g' /home/devuser/.zshrc; \
    chown -R devuser:devuser /home/devuser/

# Set workspace directory and ownership
WORKDIR /sgl-workspace/sglang
RUN chown -R devuser:devuser /sgl-workspace

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN curl -fsSL https://claude.ai/install.sh | sh
