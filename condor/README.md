### Job submission commands

```
# Obtain a VOMS proxy certificate
voms-proxy-init -rfc -voms cms --valid 168:00

# List available config files (each .yaml file corresponds to one job)
ls ./configs

# Submit jobs too HTCondor under afs system
condor_submit submit.sub
```
