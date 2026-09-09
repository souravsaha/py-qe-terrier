# python direct-correlation-idealq.py --dataset msmarco_passage --method dfo --name dfo_trial
#python direct-correlation-idealq.py --dataset trec678rb --method leastsq --name leastsq_1 --l2_lambda 1.0
#python direct-correlation-idealq.py --dataset trec678rb --method leastsq --name leastsq_0.1 --l2_lambda 0.1

# for k in 50 150 200 250 300 350 400 450 500
# do
#   python direct-correlation-idealq.py \
#     --dataset msmarco_document \
#     --method lda \
#     --name lda_$k \
#     --k $k
# done

# python direct-correlation-idealq.py --dataset msmarco_document --method leastsq --l2_lambda 1.0 --name leastsq_1_sphereproj
# python direct-correlation-idealq.py --dataset msmarco_passage --method leastsq --l2_lambda 1.0 --name leastsq_1_hyde

# python direct-correlation-sep_vs_cosine.py --dataset trec678rb --method leastsq --l2_lambda 1 --name leastsq_1
# python direct-correlation-sep_vs_cosine.py --dataset msmarco_passage --method leastsq --l2_lambda 0.1 --name leastsq_0.1
# python direct-correlation-sep_vs_cosine.py --dataset msmarco_document --method dfo --name dfo_final_iter
# python direct-correlation-corrsep_vs_cosine.py --dataset trec678rb --method leastsq --name leastsq_0.1 --l2_lambda 0.1

# python direct-correlation-idealq.py --dataset trec678rb --method rocchio --name oracle_rocchio
# python only_ideal_query_construction.py --dataset msmarco_passage --method leastsq --l2_lambda 0.0 --name leastsq_0

## -- AP VS AP --
# python direct-correlation-ap.py --dataset msmarco_passage
# python direct-correlation-ap.py --dataset msmarco_document
# python direct-correlation-ap.py --dataset msmarco_passage_v2
# python direct-correlation-ap.py --dataset trec678rb

# -- NARROW BAND --
# python angle_range.py --dataset msmarco_passage_v2 --name leastsq_0.1
# python angle_range.py --dataset msmarco_passage_v2 --name leastsq_1
# python angle_range.py --dataset msmarco_passage_v2 --name dfo

## -- SIMILARITY VS AP --
#python direct-correlation-idealq.py --dataset trec678rb --name dfo-FINAL --idealq_name dfo_final
#python direct-correlation-idealq.py --dataset trec678rb --name leastsq_0.1-FINAL --idealq_name leastsq_0.1
#python direct-correlation-idealq.py --dataset trec678rb --name leastsq_1-FINAL --idealq_name leastsq_1
#
#python direct-correlation-idealq.py --dataset msmarco_passage --name dfo-FINAL --idealq_name dfo_final_iter
#python direct-correlation-idealq.py --dataset msmarco_passage --name leastsq_0.1-FINAL --idealq_name leastsq_0.1
#python direct-correlation-idealq.py --dataset msmarco_passage --name leastsq_1-FINAL --idealq_name leastsq_1

#python direct-correlation-idealq.py --dataset msmarco_document --name dfo-FINAL --idealq_name dfo_final_iter
#python direct-correlation-idealq.py --dataset msmarco_document --name leastsq_0.1-FINAL --idealq_name leastsq_0.1
#python direct-correlation-idealq.py --dataset msmarco_document --name leastsq_1-FINAL --idealq_name leastsq_1

#python direct-correlation-idealq.py --dataset msmarco_passage_v2 --name dfo-FINAL --idealq_name dfo
#python direct-correlation-idealq.py --dataset msmarco_passage_v2 --name leastsq_0.1-FINAL --idealq_name leastsq_0.1
#python direct-correlation-idealq.py --dataset msmarco_passage_v2 --name leastsq_1-FINAL --idealq_name leastsq_1


## -- SEPARABILITY FINAL --
#python direct-correlation-sep_vs_cosine.py --dataset msmarco_passage --name dfo-FINAL --idealq_name dfo_final_iter
#python direct-correlation-sep_vs_cosine.py --dataset msmarco_passage --name leastsq_0.1-FINAL --idealq_name leastsq_0.1
#python direct-correlation-sep_vs_cosine.py --dataset msmarco_passage --name leastsq_1-FINAL --idealq_name leastsq_1

#python direct-correlation-sep_vs_cosine.py --dataset msmarco_document --name dfo-FINAL --idealq_name dfo_final_iter
#python direct-correlation-sep_vs_cosine.py --dataset msmarco_document --name leastsq_0.1-FINAL --idealq_name leastsq_0.1
#python direct-correlation-sep_vs_cosine.py --dataset msmarco_document --name leastsq_1-FINAL --idealq_name leastsq_1

# python direct-correlation-sep_vs_cosine.py --dataset msmarco_passage_v2 --name dfo-FINAL --idealq_name dfo
# python direct-correlation-sep_vs_cosine.py --dataset msmarco_passage_v2 --name leastsq_0.1-FINAL --idealq_name leastsq_0.1
# python direct-correlation-sep_vs_cosine.py --dataset msmarco_passage_v2 --name leastsq_1-FINAL --idealq_name leastsq_1

#python direct-correlation-sep_vs_cosine.py --dataset trec678rb --name dfo-FINAL --idealq_name dfo_final
# python direct-correlation-sep_vs_cosine.py --dataset trec678rb --name leastsq_0.1-FINAL --idealq_name leastsq_0.1
# python direct-correlation-sep_vs_cosine.py --dataset trec678rb --name leastsq_1-FINAL --idealq_name leastsq_1

## -- COORDINATE SEPARABILITY FINAL --
# python direct-correlation-corrsep_vs_cosine.py --dataset msmarco_passage --name dfo-FINAL --idealq_name dfo_final_iter
# python direct-correlation-corrsep_vs_cosine.py --dataset msmarco_passage --name leastsq_0.1-FINAL --idealq_name leastsq_0.1
# python direct-correlation-corrsep_vs_cosine.py --dataset msmarco_passage --name leastsq_1-FINAL --idealq_name leastsq_1

# python direct-correlation-corrsep_vs_cosine.py --dataset msmarco_document --name dfo-FINAL --idealq_name dfo_final_iter
# python direct-correlation-corrsep_vs_cosine.py --dataset msmarco_document --name leastsq_0.1-FINAL --idealq_name leastsq_0.1
# python direct-correlation-corrsep_vs_cosine.py --dataset msmarco_document --name leastsq_1-FINAL --idealq_name leastsq_1

python direct-correlation-corrsep_vs_cosine.py --dataset msmarco_passage_v2 --name dfo-FINAL --idealq_name dfo
python direct-correlation-corrsep_vs_cosine.py --dataset msmarco_passage_v2 --name leastsq_0.1-FINAL --idealq_name leastsq_0.1
python direct-correlation-corrsep_vs_cosine.py --dataset msmarco_passage_v2 --name leastsq_1-FINAL --idealq_name leastsq_1

#python direct-correlation-corrsep_vs_cosine.py --dataset trec678rb --name dfo-FINAL --idealq_name dfo_final
#python direct-correlation-corrsep_vs_cosine.py --dataset trec678rb --name leastsq_0.1-FINAL --idealq_name leastsq_0.1
#python direct-correlation-corrsep_vs_cosine.py --dataset trec678rb --name leastsq_1-FINAL --idealq_name leastsq_1
