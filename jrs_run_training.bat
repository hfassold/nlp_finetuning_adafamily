REM call conda activate textclassify

python nlp_finetuning_lightning_google.py --max_epochs 10 --nr_frozen_epochs 0 --batch_size 64 --learning_rate 1e-4 --seed 42
