python -d main.py -c egs/kaggle_toxic/configs/identityhate.train.config
python -d main.py -c egs/kaggle_toxic/configs/identityhate.decode.config

python -d main.py -c egs/kaggle_toxic/configs/insult.train.config
python -d main.py -c egs/kaggle_toxic/configs/insult.decode.config

python -d main.py -c egs/kaggle_toxic/configs/obscene.train.config
python -d main.py -c egs/kaggle_toxic/configs/obscene.decode.config

python -d main.py -c egs/kaggle_toxic/configs/severetoxic.train.config
python -d main.py -c egs/kaggle_toxic/configs/severetoxic.decode.config

python -d main.py -c egs/kaggle_toxic/configs/threat.train.config
python -d main.py -c egs/kaggle_toxic/configs/threat.decode.config

python -d main.py -c egs/kaggle_toxic/configs/toxic.train.config
python -d main.py -c egs/kaggle_toxic/configs/toxic.decode.config

python egs/kaggle_toxic/submit.py -t data/test.csv \
--toxic egs/kaggle_toxic/result/toxic_result.csv \
--severetoxic egs/kaggle_toxic/result/severe_toxic_result.csv \
--obscene egs/kaggle_toxic/result/obscene_result.csv \
--threat egs/kaggle_toxic/result/threat_result.csv \
--insult egs/kaggle_toxic/result/insult_result.csv \
--identityhate egs/kaggle_toxic/result/identity_hate_result.csv \
--output egs/kaggle_toxic/submission.csv
