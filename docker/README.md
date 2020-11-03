# README

## Build 

```bash
docker build -t pvnet_clean:latest .
```

## Run

To run the docker
Add the following to your ~/.bashrc

```bash
export PVNET_DOCKER=pvnet_clean:latest
export PVNET_GIT=$HOME/gits/clean-pvnet  # update
source $PVNET_GIT/docker/setup_dev.bash
```

run it with:

```bash
pvnet_docker
```
