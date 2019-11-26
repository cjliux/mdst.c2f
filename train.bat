python scripts/run.py --mode train --data_path ../data/woz/mwoz20 --config ./protos/trade_mwoz2.0.yaml -dec TRADE -bsz 32 -dr 0.2 -lr 0.001 -le 1 

python scripts/run.py --mode train -dec SAEe3tm --config ./protos/saee3tm_mwoz2.0.yaml -bsz 32 -dr 0.2 -lr 0.001 -le 1 --run_id saee2 --max_epoch 16 --patience 0
