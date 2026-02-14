# RunPod Quick Guide

## 1. Create the pod
1. Go to `RunPod -> Pods -> Deploy a Pod`.
2. Pick the `Runpod Pytorch 2.4.0` template.
3. Choose your GPU count and instance type.
4. Keep `SSH terminal access` enabled.

![RunPod deploy pod settings](../media/runpod_1.png)

## 2. Override the container image
1. Click `Edit` in the pod template section.
2. Set `Container image` to `ghcr.io/piercefreeman/vllm-bootstrap:latest`.
3. Keep `Expose TCP Ports` as `22`.
4. Change `Expose HTTP Ports` from `8888` to `8000` if you want to leverage the runpod public proxy service. If you do this, you should make sure to add a `VLLM_ACCESS_KEY` env param that will protect your vllm service. Don't give away that compute for free, you know?
5. Click `Set Overrides`, then deploy.

![RunPod template overrides](../media/runpod_2.png)

## 3. Verify the server is up
If everything worked:
1. Click into the running pod.
2. Go to `HTTP Services`.
3. Click the `:8000` service to open the `vllm-bootstrap` home health screen (`/`).

You can also open `http://<your-runpod-endpoint>:8000/docs` for API docs.

## Note on Docker pull time
The initial Docker image pull can take a while. We have seen pulls take up to **8 minutes**.
If a pull goes past **10 minutes**, the pod likely has a network issue. Tear it down and boot a different pod.
