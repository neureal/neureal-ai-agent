# neureal-ai-agent
RL agent using private and shared world models

Requirements:
* Both: CUDA Drivers installed
* Linux: [Docker Engine](https://docs.docker.com/engine/install/), [nvidia-docker2 installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker), "nvidia-smi" docker test on previous link should work.
* Windows: [WSL nbody benchmark should work](https://docs.docker.com/desktop/windows/wsl/#gpu-support)

```
$ git clone https://github.com/neureal/neureal-ai-agent.git
$ cd neureal-ai-agent
$ ./build.sh build
$ ./build.sh dev
```
