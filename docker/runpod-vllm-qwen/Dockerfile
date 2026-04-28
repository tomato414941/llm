ARG VLLM_IMAGE=vllm/vllm-openai:cu129-nightly@sha256:96d12f16910e0b1087edad26988e04b2e48a680f0999b952a7ec58bc5d0ac18a
FROM ${VLLM_IMAGE}

COPY entrypoint.sh /opt/runpod-vllm-qwen/entrypoint.sh
RUN chmod +x /opt/runpod-vllm-qwen/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/opt/runpod-vllm-qwen/entrypoint.sh"]
