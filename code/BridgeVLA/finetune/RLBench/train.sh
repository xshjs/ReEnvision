cd /PATH_TO_BRIDGEVLA/finetune
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0

cd /PATH_TO_BRIDGEVLA/finetune/RLBench


port=MASTER_PORT
GPUS_PER_NODE=NUMBER_OF_GPU_PER_MACHINE
NNODES=NUMBER_OF_MACHINE
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$port \
   train.py \
   $@ 
