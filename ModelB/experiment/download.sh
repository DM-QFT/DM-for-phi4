#!/bin/bash
# Zenodo 13908799: colloid trajectories, ~6 GB, 26 files, 4 parallel streams
cd /work/gw97/w97000/project/DM/ModelB/experiment
awk '{print $2}' md5sums.txt | xargs -P 4 -I{} sh -c 'curl -sSL -C - -o "{}" "https://zenodo.org/api/records/13908799/files/{}/content"; echo "done {} $(date +%T)"'
md5sum -c md5sums.txt > md5check.log 2>&1
echo ALLDONE
