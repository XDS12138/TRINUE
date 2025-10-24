#!/bin/bash

# 多文件日志查看工具
# 用于方便地查看和分析训练日志

# 默认实验目录
DEFAULT_EXP_DIR="experiments/train"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    echo "使用方法: $0 [选项] [实验目录]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  -t, --train         查看训练进度日志"
    echo "  -l, --loss          查看损失日志"
    echo "  -m, --metrics       查看指标日志"
    echo "  -e, --error         查看错误和警告"
    echo "  -d, --debug         查看调试信息"
    echo "  -a, --all           查看所有日志（按时间排序）"
    echo "  -f, --follow        实时跟踪日志（类似 tail -f）"
    echo "  -n NUM              显示最后 NUM 行（默认 50）"
    echo "  -s, --search TERM   搜索特定内容"
    echo "  --epoch NUM         查看特定 epoch 的日志"
    echo "  --summary           显示训练总结"
    echo ""
    echo "示例:"
    echo "  $0 -t                      # 查看最新实验的训练日志"
    echo "  $0 -l -f                   # 实时跟踪损失变化"
    echo "  $0 -e -n 100               # 查看最后100行错误日志"
    echo "  $0 --epoch 14              # 查看第14个epoch的所有日志"
    echo "  $0 -s \"depth.*collapse\"    # 搜索深度崩溃相关信息"
}

# 获取最新的实验目录
get_latest_experiment() {
    local base_dir=$1
    if [ -d "$base_dir" ]; then
        latest=$(ls -t "$base_dir" | head -n 1)
        if [ -n "$latest" ]; then
            echo "$base_dir/$latest"
        fi
    fi
}

# 检查日志目录是否存在
check_log_dir() {
    local exp_dir=$1
    local log_dir="$exp_dir/logs"
    
    if [ ! -d "$log_dir" ]; then
        echo -e "${RED}错误: 日志目录不存在: $log_dir${NC}"
        exit 1
    fi
    
    echo "$log_dir"
}

# 主要参数
FOLLOW=false
LINES=50
SEARCH_TERM=""
EPOCH_NUM=""
VIEW_TYPE=""
SHOW_SUMMARY=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--train)
            VIEW_TYPE="train"
            shift
            ;;
        -l|--loss)
            VIEW_TYPE="loss"
            shift
            ;;
        -m|--metrics)
            VIEW_TYPE="metrics"
            shift
            ;;
        -e|--error)
            VIEW_TYPE="error"
            shift
            ;;
        -d|--debug)
            VIEW_TYPE="debug"
            shift
            ;;
        -a|--all)
            VIEW_TYPE="all"
            shift
            ;;
        -f|--follow)
            FOLLOW=true
            shift
            ;;
        -n)
            LINES="$2"
            shift 2
            ;;
        -s|--search)
            SEARCH_TERM="$2"
            shift 2
            ;;
        --epoch)
            EPOCH_NUM="$2"
            shift 2
            ;;
        --summary)
            SHOW_SUMMARY=true
            shift
            ;;
        *)
            EXP_DIR="$1"
            shift
            ;;
    esac
done

# 确定实验目录
if [ -z "$EXP_DIR" ]; then
    EXP_DIR=$(get_latest_experiment "$DEFAULT_EXP_DIR")
    if [ -z "$EXP_DIR" ]; then
        echo -e "${RED}错误: 找不到实验目录${NC}"
        exit 1
    fi
    echo -e "${CYAN}使用最新实验: $EXP_DIR${NC}"
fi

# 获取日志目录
LOG_DIR=$(check_log_dir "$EXP_DIR")

# 显示训练总结
if [ "$SHOW_SUMMARY" = true ]; then
    echo -e "${GREEN}=== 训练总结 ===${NC}"
    echo ""
    
    # 从train.log提取关键信息
    if [ -f "$LOG_DIR/train.log" ]; then
        echo -e "${YELLOW}训练配置:${NC}"
        grep -E "Experiment:|Epochs:|Batch size:|Learning rate:" "$LOG_DIR/train.log" | head -n 10
        echo ""
        
        echo -e "${YELLOW}最新进度:${NC}"
        grep -E "Epoch [0-9]+/" "$LOG_DIR/train.log" | tail -n 5
        echo ""
    fi
    
    # 从metrics.log提取最佳指标
    if [ -f "$LOG_DIR/metrics.log" ]; then
        echo -e "${YELLOW}最佳验证指标:${NC}"
        grep -E "PSNR|SSIM" "$LOG_DIR/metrics.log" | sort -k6 -nr | head -n 5
        echo ""
    fi
    
    # 检查错误
    if [ -f "$LOG_DIR/error.log" ]; then
        error_count=$(grep -c "ERROR" "$LOG_DIR/error.log" 2>/dev/null || echo "0")
        warn_count=$(grep -c "WARNING" "$LOG_DIR/error.log" 2>/dev/null || echo "0")
        echo -e "${YELLOW}错误统计:${NC}"
        echo -e "  错误: ${RED}$error_count${NC}"
        echo -e "  警告: ${YELLOW}$warn_count${NC}"
        
        if [ "$error_count" -gt 0 ]; then
            echo ""
            echo -e "${RED}最近的错误:${NC}"
            grep "ERROR" "$LOG_DIR/error.log" | tail -n 3
        fi
    fi
    
    exit 0
fi

# 根据类型查看日志
case $VIEW_TYPE in
    train)
        LOG_FILE="$LOG_DIR/train.log"
        COLOR=$GREEN
        ;;
    loss)
        LOG_FILE="$LOG_DIR/loss.log"
        COLOR=$YELLOW
        ;;
    metrics)
        LOG_FILE="$LOG_DIR/metrics.log"
        COLOR=$BLUE
        ;;
    error)
        LOG_FILE="$LOG_DIR/error.log"
        COLOR=$RED
        ;;
    debug)
        LOG_FILE="$LOG_DIR/debug.log"
        COLOR=$PURPLE
        ;;
    all)
        LOG_FILE="$LOG_DIR/*.log"
        COLOR=$CYAN
        ;;
    *)
        echo -e "${RED}错误: 请指定要查看的日志类型${NC}"
        show_help
        exit 1
        ;;
esac

# 执行查看操作
if [ -n "$SEARCH_TERM" ]; then
    # 搜索模式
    echo -e "${COLOR}=== 搜索 '$SEARCH_TERM' in $VIEW_TYPE logs ===${NC}"
    if [ "$VIEW_TYPE" = "all" ]; then
        grep -n "$SEARCH_TERM" $LOG_FILE | head -n "$LINES"
    else
        grep -n "$SEARCH_TERM" "$LOG_FILE" | head -n "$LINES"
    fi
elif [ -n "$EPOCH_NUM" ]; then
    # 查看特定epoch
    echo -e "${COLOR}=== Epoch $EPOCH_NUM logs ===${NC}"
    if [ "$VIEW_TYPE" = "all" ]; then
        grep -E "Epoch $EPOCH_NUM[^0-9]|epoch.*$EPOCH_NUM[^0-9]" $LOG_FILE | head -n "$LINES"
    else
        grep -E "Epoch $EPOCH_NUM[^0-9]|epoch.*$EPOCH_NUM[^0-9]" "$LOG_FILE" | head -n "$LINES"
    fi
elif [ "$FOLLOW" = true ]; then
    # 实时跟踪模式
    echo -e "${COLOR}=== 实时跟踪 $VIEW_TYPE logs ===${NC}"
    echo -e "${CYAN}按 Ctrl+C 停止${NC}"
    if [ "$VIEW_TYPE" = "all" ]; then
        tail -f $LOG_FILE
    else
        tail -f "$LOG_FILE"
    fi
else
    # 普通查看模式
    echo -e "${COLOR}=== 最后 $LINES 行 $VIEW_TYPE logs ===${NC}"
    if [ "$VIEW_TYPE" = "all" ]; then
        # 对于all类型，合并所有日志并按时间排序
        cat $LOG_FILE | sort -k1,2 | tail -n "$LINES"
    else
        tail -n "$LINES" "$LOG_FILE"
    fi
fi 