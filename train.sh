# TRAIN ON PARALLEL CORPUS

python -u train.py \
    --train-src=$PATH_OF_SOURCE_SIDE_OF_PARALLEL_CORPUS \
    --train-tgt=$PATH_OF_TARGET_SIDE_OF_PARALLEL_CORPUS \
    --dev-src=$PATH_OF_SOURCE_SIDE_OF_DEV_SET \
    --dev-tgt=$PATH_OF_TARGET_SIDE_OF_DEV_SET \
    --dev-hter=$PATH_OR_SENTENCE_LEVEL_SCORES_OF_DEV_SET \
    --dev-tags=$PATH_OF_WORD_LEVEL_TAGS_OF_DEV_SET \
    --wwm \
    --pretrained-model-path=$PATH_OF_PRETRAINED_MODEL \
    --save-model-path=$PATH_OF_SAVED_MODEL
