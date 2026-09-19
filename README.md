This Paper is under review, please do not distribute the codebase. 


### Correlation computation 
This code will generate ideal queries and run correlation. Modify the parameters in the following code to run on different datasets with different ideal queries.
```bash
python direct-correlation-idealq.py --dataset msmarco_passage --method dfo --name dfo_trial
```


### Other findings

To compute correlation between Restricted AP and Real AP, do the following
```bash
python direct-correlation-ap.py --dataset msmarco_passage
```

To compute the different spread of EQs, and the angle with respect to the ideal queries, use the following code.
```bash
python angle_range.py --dataset msmarco_passage_v2 --name leastsq_0.1
```

### Plot 
Spherical sweep of 360 degree from ideal query to EQs. This will generate all different sweeps. 
```bash
python spherical_sweep.py --dataset trec678rb --idealq_name leastsq_0.1
```
Across different queries these are aggregated with the following code. Please modify the parameters to run on different datasets with different ideal queries.
```bash
python spherical_sweep_per_dataset.py --dataset trec678rb  --idealq_name leastsq_0.1
```
