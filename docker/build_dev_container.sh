#!/bin/bash

docker run -itd --shm-size 32g --gpus all \
  -e http_proxy="http://127.0.0.1:8118" -e https_proxy="http://127.0.0.1:8118" -e HTTP_PROXY="http://127.0.0.1:8118" -e HTTPS_PROXY="http://127.0.0.1:8118" -e NO_PROXY="localhost,127.0.0.1" \
  -v ~/.claude:/root/.claude \
  -v /data/:/root/.cache/huggingface \
  -v $HOME/repos/sglang:/sgl-workspace/sglang \
  --ipc=host \
  --network=host --privileged \
  --name sglang_dev lmsysorg/sglang:dev /bin/zsh

docker exec -it sglang_dev /bin/zsh
