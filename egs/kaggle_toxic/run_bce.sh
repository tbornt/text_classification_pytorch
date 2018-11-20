start=$(date +%s)

if [ $1 == "rnn" ]; then
    python -u main.py -c egs/kaggle_toxic/configs/bce_cfgs/rnn.train.config
    python -u main.py -c egs/kaggle_toxic/configs/bce_cfgs/rnn.decode.config

elif [ $1 == "cnn" ]; then
    python -u main.py -c egs/kaggle_toxic/configs/bce_cfgs/cnn.train.config
    python -u main.py -c egs/kaggle_toxic/configs/bce_cfgs/cnn.decode.config

elif [ $1 == "dpcnn" ]; then
    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/identityhate.train.config
    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/identityhate.decode.config

elif [ $1 == "rcnn" ]; then
    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/identityhate.train.config
    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/identityhate.decode.config

elif [ $1 == "transformer" ]; then
    python -u main.py -c egs/kaggle_toxic/configs/transformer_cfgs/identityhate.train.config
    python -u main.py -c egs/kaggle_toxic/configs/transformer_cfgs/identityhate.decode.config
fi

python egs/kaggle_toxic/submit.py -t data/test.csv --decode_result egs/kaggle_toxic/result/bce_model_result.csv \
--output egs/kaggle_toxic/submission.csv

end=$(date +%s)
runtime=$(python -c "print('%u:%02u' % ((${end} - ${start})/60, (${end} - ${start})%60))")
echo "Runtime was $runtime"
