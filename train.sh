#########################################################################
# File Name: train.sh
# Author: frank
# mail: 1216451203@qq.com
# Created Time: 2025年07月07日 星期一 11时30分22秒
#########################################################################
#!/bin/bash
get_dir() {
    local start_index=${1:-1}       # 默认起始索引为1
    local parent_dir=${2:-"./"}     # 默认父目录为当前目录

    # 获取当前日期部分
    local date_part=$(date "+%Y-%m-%d")
    local index=$start_index
    local dir_name

    # 循环直到找到不存在的目录名
    while true; do
        dir_name="${date_part}_${index}"

        # 检查目录是否已存在
        if [ ! -d "${parent_dir}/${dir_name}" ]; then
            echo "${parent_dir}/$dir_name"  # 返回可用的目录名
            return 0
        fi

        ((index++))  # 递增索引
    done
}






DIR=$(get_dir 1 "results/")
echo $DIR
mkdir -p $DIR

log="$DIR/log"
tf_logs="$DIR/tf_log"
model_dir="$DIR/model"
checkpoint_path="results/checkpoint_57000"

#while [ True ]; do
#    num=`ls ${model_dir}/check* | wc -l`
#    if [ $num -gt 100 ]; then
#        ls ${model_dir}/check* -tr | head -1 | xargs rm
#    fi
#    sleep 3
#done  &
#model_dir="/path/to/your/model_dir"  # 替换为您的实际目录
cleanup_pid=""

# 启动清理函数
start_cleanup() {
    while true; do
        num=$(ls ${model_dir}/check* 2>/dev/null | wc -l)
        if [ $num -gt 100 ]; then
            # 删除最旧的一个checkpoint
            oldest=$(ls ${model_dir}/check* -tr | head -1)
            echo "删除旧checkpoint: $oldest"
            rm -rf "$oldest"
        fi
        sleep 3
    done
}

# 捕获中断信号
trap 'stop_cleanup' INT TERM

stop_cleanup() {
    echo "接收到中断信号，停止清理进程..."
    kill -TERM $cleanup_pid 2>/dev/null
    exit 0
}

# 在子shell中启动清理并记录PID
(start_cleanup) &
cleanup_pid=$!

echo "后台清理进程已启动，PID: $cleanup_pid"

if [! -d Wave.tar ];then
    wget https://github-1324907443.cos.ap-shanghai.myqcloud.com/tacotron2/Wave.tar
    tar xvf Wave.tar
fi

#train
train="learning_rate=5e-5"
#tune
tune="learning_rate=5e-6"
python train.py  -l ${tf_logs} -o ${model_dir}  -c ${checkpoint_path} \
    --hparams ${train} &> $log

# 等待清理进程
wait $cleanup_pid
