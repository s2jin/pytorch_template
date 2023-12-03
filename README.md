
#### train command

```
python run.py --task general -td data/klueTC_small/training.jsonl -vd data/klueTC_small/valid.jsonl --label_path data/klueTC_small/label_list.txt -s checkpoint
```

#### predict command

```
python run.py --task general --label_path data/klueTC_small/label_list.txt --predict -ed data/klueTC_small/test.jsonl -s prediction_test.jsonl -w checkpoint/*/trained_model/
```
