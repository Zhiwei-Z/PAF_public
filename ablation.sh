python launcher_scripts/execute.py -c configs/ant/base.json -t configs/ant/tune_offset.json -r 4 -p 4 -n ablation_ant &&
python launcher_scripts/execute.py -c configs/hc/base.json -t configs/hc/tune_offset.json -r 4 -p 4 -n ablation_cheetah && 
python launcher_scripts/execute.py -c configs/hopper/base.json -t configs/hopper/tune_offset.json -r 4 -p 4 -n ablation_hopper &&
python launcher_scripts/execute.py -c configs/walker/base.json -t configs/walker/tune_offset.json -r 4 -p 4 -n ablation_walker