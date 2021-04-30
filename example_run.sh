BE_ROOT=/nas/home/jwei/metrics-quantification

CUDA_VISIBLE_DEVICES=1 python -m bleurt.finetune \
    -init_checkpoint=$BE_ROOT/bert-base/bert-base \
    -bert_config_file=$BE_ROOT/bert-base/bert_config.json \
    -vocab_file=$BE_ROOT/bert-base/vocab.txt \
    -model_dir=$BE_ROOT/experiments/group_loss_train_2016-19/alphas/alpha08 \
    -train_set=$BE_ROOT/data/train_2016-18_test_2019/dev_2018/train_2016-17.jsonl \
    -dev_set=$BE_ROOT/data/train_2016-18_test_2019/dev_2018/dev_2018.jsonl \
    -num_train_steps=30000
