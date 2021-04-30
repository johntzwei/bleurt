# cand file (tr16): /nas/home/jwei/metrics-quantification/data/train_2016_test_2017-19/candidates
# bleurt checkpoint (tr16): /nas/home/jwei/metrics-quantification/experiments/vanilla_train_2016/pairwise_val_3/export/bleurt_best/1618706529

python score_group_attrs.py\
    --candidate_file /nas/home/jwei/metrics-quantification/data/train_2016_test_2017-19/dev_2016_de-en/val_cands\
    --reference_file /nas/home/jwei/metrics-quantification/data/train_2016_test_2017-19/dev_2016_de-en/val_refs\
    --scores_file /nas/home/jwei/metrics-quantification/experiments/group_loss_train_2016/pw_val_dev_scores\
    --bleurt_checkpoint /nas/home/jwei/metrics-quantification/experiments/group_loss_train_2016/pairwise_val_2/export/bleurt_best/1618667255
