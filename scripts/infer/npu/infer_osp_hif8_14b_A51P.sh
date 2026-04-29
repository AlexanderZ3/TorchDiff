pkill -9 -f infer.py
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export TOKENIZERS_PARALLELISM=false

export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=0
export ACL_DEVICE_SYNC_TIMEOUT=3600

# [A51P] 单卡指定卡号，默认0号卡，按需修改
# export ASCEND_VISIBLE_DEVICES=0

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29505}
NPRC_PER_NODE=${NPRC_PER_NODE:-1}   # [A51P] 单卡：原为8，改为1
NNODES=${NNODES:-1}
WORLD_SIZE=$(($NNODES * $NPRC_PER_NODE))

torchrun \
  --nproc_per_node=${NPRC_PER_NODE} \
  --nnodes=${NNODES} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  infer/infer_osp_A51P.py \
  --config configs/infer/npu/osp_hif8_14b_A51P.yaml
