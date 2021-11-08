# PREDICT ON DEV & TEST SET

# these values may be set larger for better performance
PREDICT_N=40
PREDICT_M=6

python -u predict.py \
    --test-src=$PATH_OF_SOURCE_SIDE_OF_DEV_SET \
    --test-tgt=$PATH_OF_TARGET_SIDE_OF_DEV_SET \
    --threshold-tune=$PATH_OF_WORD_LEVEL_TAGS_OF_DEV_SET \
    --wwm \
    --mc-dropout \
    --predict-n=$PREDICT_N \
    --predict-m=$PREDICT_M \
    --checkpoint=$PATH_OF_SAVED_CHECKPOINT \
    --word-output=$PATH_OF_DEV_OUTPUT_OF_WORD_LEVEL_TAGS \
    --sent-output=$PATH_OF_DEV_OUTPUT_OF_SENT_LEVEL_SCORE \
    --score-output=$PATH_OF_DEV_OUTPUT_OF_WORD_LEVEL_SCORE \
    --threshold-output=$PATH_OF_THRESHOLD

python -u predict.py \
    --test-src=$PATH_OF_SOURCE_SIDE_OF_TEST_SET \
    --test-tgt=$PATH_OF_TARGET_SIDE_OF_DEV_SET \
    --wwm \
    --mc-dropout \
    --predict-n=$PREDICT_N \
    --predict-m=$PREDICT_M \
    --checkpoint=$PATH_OF_SAVED_CHECKPOINT \
    --word-output=$PATH_OF_TEST_OUTPUT_OF_WORD_LEVEL_TAGS \
    --sent-output=$PATH_OF_TEST_OUTPUT_OF_SENT_LEVEL_SCORE \
    --score-output=$PATH_OF_TEST_OUTPUT_OF_WORD_LEVEL_SCORE \
    --threshold=$PATH_OF_THRESHOLD
