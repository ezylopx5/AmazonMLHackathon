#!/bin/bash
# Real-time system monitoring during training

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "📊 Starting system monitoring..."
echo "Logs will be saved to: $LOG_DIR"

# Function to log system stats
log_stats() {
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # GPU stats (if available)
    if command -v nvidia-smi &> /dev/null; then
        gpu_stats=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits)
        echo "[$timestamp] GPU: $gpu_stats" >> "$LOG_DIR/gpu_monitoring.log"
    fi
    
    # CPU and Memory
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    
    echo "[$timestamp] CPU: ${cpu_usage}%, Memory: ${mem_usage}%" >> "$LOG_DIR/system_monitoring.log"
    
    # Disk usage for project directory
    disk_usage=$(df -h "$PROJECT_DIR" | tail -1 | awk '{print $5}')
    echo "[$timestamp] Disk: $disk_usage" >> "$LOG_DIR/disk_monitoring.log"
    
    # Python processes
    python_processes=$(ps aux | grep python | grep -E "(train|infer|validate)" | wc -l)
    echo "[$timestamp] Python ML processes: $python_processes" >> "$LOG_DIR/process_monitoring.log"
}

# Enhanced monitoring with alerts
monitor_with_alerts() {
    while true; do
        log_stats
        
        # Check for high memory usage
        mem_percent=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
        if [ "$mem_percent" -gt 90 ]; then
            echo "🚨 HIGH MEMORY USAGE: ${mem_percent}%" | tee -a "$LOG_DIR/alerts.log"
            
            # Log top memory processes
            echo "Top memory processes:" >> "$LOG_DIR/alerts.log"
            ps aux --sort=-%mem | head -10 >> "$LOG_DIR/alerts.log"
        fi
        
        # Check GPU memory if available
        if command -v nvidia-smi &> /dev/null; then
            gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{printf "%.0f", $1/$2 * 100}')
            if [ "$gpu_mem" -gt 95 ]; then
                echo "🚨 HIGH GPU MEMORY: ${gpu_mem}%" | tee -a "$LOG_DIR/alerts.log"
            fi
        fi
        
        # Check disk space
        disk_percent=$(df "$PROJECT_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')
        if [ "$disk_percent" -gt 85 ]; then
            echo "🚨 LOW DISK SPACE: ${disk_percent}% used" | tee -a "$LOG_DIR/alerts.log"
        fi
        
        sleep 30  # Monitor every 30 seconds
    done
}

# Training progress monitoring
monitor_training() {
    echo "📈 Monitoring training progress..."
    
    # Watch for training logs
    if [ -f "$LOG_DIR/training.log" ]; then
        tail -f "$LOG_DIR/training.log" | while read line; do
            if echo "$line" | grep -q "Epoch\|SMAPE\|Loss"; then
                echo "[$timestamp] $line" | tee -a "$LOG_DIR/training_progress.log"
            fi
        done &
    fi
    
    # Monitor checkpoint creation
    checkpoint_dir="$PROJECT_DIR/checkpoints"
    if [ -d "$checkpoint_dir" ]; then
        inotifywait -m -e create "$checkpoint_dir" 2>/dev/null | while read path action file; do
            if [[ "$file" == *.pt ]]; then
                size=$(du -h "$path$file" | cut -f1)
                timestamp=$(date '+%Y-%m-%d %H:%M:%S')
                echo "[$timestamp] New checkpoint: $file ($size)" | tee -a "$LOG_DIR/checkpoint_monitoring.log"
            fi
        done &
    fi
}

# Summary report generation
generate_report() {
    echo "📋 Generating monitoring report..."
    
    report_file="$LOG_DIR/monitoring_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "Amazon ML Challenge 2025 - Monitoring Report"
        echo "============================================="
        echo "Generated: $(date)"
        echo ""
        
        echo "System Information:"
        echo "- OS: $(uname -a)"
        echo "- Python: $(python --version 2>&1)"
        echo "- PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
        echo ""
        
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Information:"
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
            echo ""
        fi
        
        echo "Peak Resource Usage:"
        if [ -f "$LOG_DIR/system_monitoring.log" ]; then
            echo "- Peak CPU: $(grep "CPU:" "$LOG_DIR/system_monitoring.log" | awk -F'CPU: ' '{print $2}' | awk -F'%' '{print $1}' | sort -n | tail -1)%"
            echo "- Peak Memory: $(grep "Memory:" "$LOG_DIR/system_monitoring.log" | awk -F'Memory: ' '{print $2}' | awk -F'%' '{print $1}' | sort -n | tail -1)%"
        fi
        
        if [ -f "$LOG_DIR/alerts.log" ]; then
            echo ""
            echo "Alerts Summary:"
            cat "$LOG_DIR/alerts.log" | tail -20
        fi
        
        echo ""
        echo "Training Progress:"
        if [ -f "$LOG_DIR/training_progress.log" ]; then
            tail -20 "$LOG_DIR/training_progress.log"
        fi
        
    } > "$report_file"
    
    echo "📄 Report saved to: $report_file"
}

# Signal handlers
cleanup() {
    echo ""
    echo "🔄 Stopping monitoring..."
    generate_report
    kill $(jobs -p) 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main monitoring loop
case "${1:-full}" in
    "stats")
        echo "📊 Logging system stats once..."
        log_stats
        ;;
    "training")
        echo "📈 Starting training monitoring..."
        monitor_training
        wait
        ;;
    "report")
        generate_report
        ;;
    "full"|*)
        echo "🔄 Starting full monitoring (Press Ctrl+C to stop and generate report)"
        monitor_with_alerts &
        monitor_training &
        wait
        ;;
esac