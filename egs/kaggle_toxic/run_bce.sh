start=$(date +%s)

if [ $1 == "rnn" ]; then
    python -u main.py -c egs/kaggle_toxic/configs/bce_cfgs/rnn.train.config
    python -u main.py -c egs/kaggle_toxic/configs/bce_cfgs/rnn.decode.config

elif [ $1 == "cnn" ]; then
    python -u main.py -c egs/kaggle_toxic/configs/bce_cfgs/cnn.train.config
    python -u main.py -c egs/kaggle_toxic/configs/bce_cfgs/cnn.decode.config

elif [ $1 == "dpcnn" ]; then
    python -u main.py -c egs/kaggle_toxic/configs/bce_cfgs/dpcnn.train.config
    python -u main.py -c egs/kaggle_toxic/configs/bce_cfgs/dpcnn.decode.config

elif [ $1 == "rcnn" ]; then
    python -u main.py -c egs/kaggle_toxic/configs/bce_cfgs/rcnn.train.config
    python -u main.py -c egs/kaggle_toxic/configs/bce_cfgs/rcnn.decode.config
fi

python egs/kaggle_toxic/submit_bce.py -t data/test.csv --decode_result egs/kaggle_toxic/result/bce_model_result.csv \
--output egs/kaggle_toxic/submission.csv

end=$(date +%s)
runtime=$(python -c "print('%u:%02u' % ((${end} - ${start})/60, (${end} - ${start})%60))")
echo "Runtime was $runtime"
