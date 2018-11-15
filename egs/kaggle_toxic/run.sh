start=$(date +%s)

if [ $1 == "rnn" ]; then
    python -u main.py -c egs/kaggle_toxic/configs/rnn_cfgs/identityhate.train.config
    python -u main.py -c egs/kaggle_toxic/configs/rnn_cfgs/identityhate.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/rnn_cfgs/insult.train.config
    python -u main.py -c egs/kaggle_toxic/configs/rnn_cfgs/insult.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/rnn_cfgs/obscene.train.config
    python -u main.py -c egs/kaggle_toxic/configs/rnn_cfgs/obscene.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/rnn_cfgs/severetoxic.train.config
    python -u main.py -c egs/kaggle_toxic/configs/rnn_cfgs/severetoxic.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/rnn_cfgs/threat.train.config
    python -u main.py -c egs/kaggle_toxic/configs/rnn_cfgs/threat.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/rnn_cfgs/toxic.train.config
    python -u main.py -c egs/kaggle_toxic/configs/rnn_cfgs/toxic.decode.config

elif [ $1 == "cnn" ]; then
    python -u main.py -c egs/kaggle_toxic/configs/cnn_cfgs/identityhate.train.config
    python -u main.py -c egs/kaggle_toxic/configs/cnn_cfgs/identityhate.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/cnn_cfgs/insult.train.config
    python -u main.py -c egs/kaggle_toxic/configs/cnn_cfgs/insult.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/cnn_cfgs/obscene.train.config
    python -u main.py -c egs/kaggle_toxic/configs/cnn_cfgs/obscene.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/cnn_cfgs/severetoxic.train.config
    python -u main.py -c egs/kaggle_toxic/configs/cnn_cfgs/severetoxic.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/cnn_cfgs/threat.train.config
    python -u main.py -c egs/kaggle_toxic/configs/cnn_cfgs/threat.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/cnn_cfgs/toxic.train.config
    python -u main.py -c egs/kaggle_toxic/configs/cnn_cfgs/toxic.decode.config

elif [ $1 == "dpcnn" ]; then
    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/identityhate.train.config
    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/identityhate.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/insult.train.config
    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/insult.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/obscene.train.config
    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/obscene.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/severetoxic.train.config
    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/severetoxic.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/threat.train.config
    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/threat.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/toxic.train.config
    python -u main.py -c egs/kaggle_toxic/configs/dpcnn_cfgs/toxic.decode.config

elif [ $1 == "rcnn" ]; then
    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/identityhate.train.config
    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/identityhate.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/insult.train.config
    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/insult.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/obscene.train.config
    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/obscene.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/severetoxic.train.config
    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/severetoxic.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/threat.train.config
    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/threat.decode.config

    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/toxic.train.config
    python -u main.py -c egs/kaggle_toxic/configs/rcnn_cfgs/toxic.decode.config
fi

python egs/kaggle_toxic/submit.py -t data/test.csv \
--toxic egs/kaggle_toxic/result/toxic_result.csv \
--severetoxic egs/kaggle_toxic/result/severe_toxic_result.csv \
--obscene egs/kaggle_toxic/result/obscene_result.csv \
--threat egs/kaggle_toxic/result/threat_result.csv \
--insult egs/kaggle_toxic/result/insult_result.csv \
--identityhate egs/kaggle_toxic/result/identity_hate_result.csv \
--output egs/kaggle_toxic/submission.csv

end=$(date +%s)
runtime=$(python -c "print('%u:%02u' % ((${end} - ${start})/60, (${end} - ${start})%60))")
echo "Runtime was $runtime"
