python spherical_sweep_per_dataset.py --dataset msmarco_passage  --idealq_name leastsq_0.1
python spherical_sweep_per_dataset.py --dataset msmarco_passage  --idealq_name leastsq_1
python spherical_sweep_per_dataset.py --dataset msmarco_passage  --idealq_name dfo_final_iter


python spherical_sweep_per_dataset.py --dataset msmarco_document  --idealq_name leastsq_0.1
python spherical_sweep_per_dataset.py --dataset msmarco_document  --idealq_name leastsq_1
python spherical_sweep_per_dataset.py --dataset msmarco_document  --idealq_name dfo_final_iter

python spherical_sweep_per_dataset.py --dataset msmarco_passage_v2  --idealq_name leastsq_0.1
python spherical_sweep_per_dataset.py --dataset msmarco_passage_v2  --idealq_name leastsq_1
python spherical_sweep_per_dataset.py --dataset msmarco_passage_v2  --idealq_name dfo

python spherical_sweep_per_dataset.py --dataset trec678rb  --idealq_name leastsq_0.1
python spherical_sweep_per_dataset.py --dataset trec678rb  --idealq_name leastsq_1
python spherical_sweep_per_dataset.py --dataset trec678rb  --idealq_name dfo_final
