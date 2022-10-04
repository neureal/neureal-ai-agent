# neureal-ai-agent
RL agent using private and shared world models

Requirements:
* Linux/Windows: CUDA Drivers installed ([compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#deployment-consideration-forward))
* Linux: [Docker Engine](https://docs.docker.com/engine/install/), [nvidia-docker2 installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker), "nvidia-smi" docker test on previous link needs to work.
* Windows (untested): [WSL nbody benchmark needs to work](https://docs.docker.com/desktop/windows/wsl/#gpu-support) ["Does nvidia-docker2 support Windows?"](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#is-microsoft-windows-supported)

```
$ git clone https://github.com/neureal/neureal-ai-agent.git
$ cd neureal-ai-agent
$ ./build.sh build
$ ./build.sh dev
root@f3537fad7977:/app# time python agent.py
...
RUN gym-PG-CartPole-r-dev-a0-22-10-03-11-10-04
tracing -> GeneralAI PG
tracing -> GeneralAI pg_actor
tracing -> GeneralAI pg_learner_onestep
RUNNING
gym-PG-CartPole-r-dev-a0-22-10-03-11-10-04    [CPU-float64]    A[inAio+D512-Ï7_net02AT+D256_outAio+D512-Öc2_Ld7x16]-new
time:   0:00:00:21    steps:2712    t/s:0.00781960    ms:256     |     attn: net io out ar    al:16    am:4     |     a-clk:0.001    a-spd:160.0    aug:SP     |     action:4e-06

real	0m28.174s
user	0m42.520s
sys	0m9.661s
root@f3537fad7977:/app#
```
