# neureal-ai-agent
RL agent using private and shared world models

Requirements:
* Linux/Windows: CUDA Drivers installed ([compatability matrix](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#deployment-consideration-forward))
* Linux: [Docker Engine](https://docs.docker.com/engine/install/), [nvidia-docker2 installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker), "nvidia-smi" docker test on previous link needs to work.
* Windows: [WSL nbody benchmark needs to work](https://docs.docker.com/desktop/windows/wsl/#gpu-support)

```
$ git clone https://github.com/neureal/neureal-ai-agent.git
$ cd neureal-ai-agent
$ ./build.sh build
$ ./build.sh dev
```
