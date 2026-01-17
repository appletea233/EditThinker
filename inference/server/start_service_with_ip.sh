#!/bin/bash

# Unified Multimodal API Service Startup Script
# Supports selecting and starting different services

set -e

# ==============================================================================
#                           USER CONFIGURATION
# ==============================================================================

# 1. Environment Paths
# Path to save the server IP address
SERVER_IP_SAVE_PATH=$2

# Path to the Conda environment to check against
# Please replace with your own conda environment path
CONDA_ENV_PATH="/path/to/your/conda/env"  # Replace with your conda environment path


# 2. Service Performance Settings (Defaults)
# Default task queue size for standard services
DEFAULT_QUEUE_SIZE=100
# Default task timeout in seconds
DEFAULT_TIMEOUT=300

# Settings for heavier models (e.g., Flux Fill, LongCat)
HEAVY_TASK_QUEUE_SIZE=50
HEAVY_TASK_TIMEOUT=600

# ==============================================================================
#                        END OF USER CONFIGURATION
# ==============================================================================

# Color Definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}   Multimodal API Service Launcher${NC}"
echo -e "${CYAN}========================================${NC}"

# Display Help
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  ./start_service.sh [Service ID]"
    echo -e ""
    echo -e "${YELLOW}Service IDs:${NC}"
    echo -e "  1 - Qwen-Image Generation Service (Port: 8000)"
    echo -e "  2 - Qwen-Image Lightning Generation Service (Port: 8001)"
    echo -e "  3 - FLUX.1-Krea-dev Generation Service (Port: 8002)"
    echo -e "  4 - Qwen-Image-Edit Service (Port: 8003)"
    echo -e "  5 - Qwen-Image-Edit Lightning Service (Port: 8004)"
    echo -e "  6 - FLUX.1-Kontext-dev Edit Service (Port: 8005)"
    echo -e "  7 - FLUX.1-Fill-dev Fill Service (Port: 8006)"
    echo -e "  8 - LongCat-Image-Edit Service (Port: 8010)"
    echo -e "  9 - OmniGen2-Image-Edit Service (Port: 8007)"
    echo -e "  9 - OmniGen2-Image-Edit Service (Port: 8007)"
    echo -e ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  ./start_service.sh 1    # Start Qwen-Image Generation"
    echo -e "  ./start_service.sh 2    # Start Qwen-Image Lightning"
    echo -e "  ./start_service.sh      # Interactive selection"
    echo -e ""
    exit 0
fi

# Check Conda Environment
check_conda_env() {
    echo -e "${YELLOW}1. Checking Conda environment...${NC}"
    
    if [[ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV_PATH" ]] || [[ "$CONDA_PREFIX" == "$CONDA_ENV_PATH" ]]; then
        echo -e "${GREEN}✓ Conda environment activated: $CONDA_ENV_PATH${NC}"
    else
        echo -e "${RED}✗ You must activate the conda environment first:${NC}"
        echo -e "${YELLOW}conda activate $CONDA_ENV_PATH${NC}"
        exit 1
    fi
}

# Check Python Dependencies
check_python_deps() {
    echo -e "\n${YELLOW}2. Checking Python dependencies...${NC}"
    python -c "import fastapi, uvicorn, torch, diffusers, PIL" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Basic dependencies checked${NC}"
    else
        echo -e "${RED}✗ Missing necessary Python packages${NC}"
        exit 1
    fi
}

# Check GPU Availability
check_gpu() {
    echo -e "\n${YELLOW}3. Checking GPU status...${NC}"
    python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}')"
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo -e "${GREEN}✓ GPU is available${NC}"
    else
        echo -e "${YELLOW}⚠ GPU not available, using CPU mode (performance will be slow)${NC}"
    fi
}

# Check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}✗ Port $port is already in use${NC}"
        echo -e "${YELLOW}Process Info:${NC}"
        lsof -Pi :$port -sTCP:LISTEN
        return 1
    else
        echo -e "${GREEN}✓ Port $port is available${NC}"
        return 0
    fi
}

# Show Service List
show_services() {
    echo -e "\n${YELLOW}4. Available Services:${NC}"
    echo -e "${BLUE}----------------------------------------${NC}"
    echo "1. Qwen-Image Generation Service (Port: 8000)"
    echo "2. Qwen-Image Lightning Generation Service (Port: 8001)"
    echo "3. FLUX.1-Krea-dev Generation Service (Port: 8002)"
    echo "4. Qwen-Image-Edit Service (Port: 8003)"
    echo "5. Qwen-Image-Edit Lightning Service (Port: 8004)"
    echo "6. FLUX.1-Kontext-dev Edit Service (Port: 8005)"
    echo "7. FLUX.1-Fill-dev Fill Service (Port: 8006)"
    echo "8. LongCat-Image-Edit Service (Port: 8010)"
    echo "9. OmniGen2-Image-Edit Service (Port: 8007)"
    echo -e "${BLUE}----------------------------------------${NC}"
}

# Get Service Information
get_service_info() {
    local choice=$1
    case $choice in
        1) echo "qwen_generation_api.py,Qwen-Image Generation,8000" ;;
        2) echo "qwen_generation_lightning_api.py,Qwen-Image Lightning Generation,8001" ;;
        3) echo "flux_generation_api.py,FLUX.1-Krea-dev Generation,8002" ;;
        4) echo "qwen_edit_api.py,Qwen-Image-Edit,8003" ;;
        5) echo "qwen_edit_lightning_api.py,Qwen-Image-Edit Lightning,8004" ;;
        6) echo "flux_edit_api.py,FLUX.1-Kontext-dev Edit,8005" ;;
        7) echo "flux_fill_api.py,FLUX.1-Fill-dev Fill,8006" ;;
        8) echo "longcat_edit_api.py,LongCat-Image-Edit,8010" ;;
        9) echo "omnigen2_edit_api.py,OmniGen2-Image-Edit,8007" ;;
        *) echo "" ;;
    esac
}

# Setup Environment Variables
setup_env_vars() {
    local service_name=$1
    
    # Calculate available GPUs
    local gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
    export NUM_GPUS_TO_USE=${NUM_GPUS_TO_USE:-$gpu_count}

    # Set queue and timeout based on service type using variables defined at the top
    if [[ $service_name == *"lightning"* ]]; then
        export TASK_QUEUE_SIZE=${TASK_QUEUE_SIZE:-$DEFAULT_QUEUE_SIZE}
        export TASK_TIMEOUT=${TASK_TIMEOUT:-$DEFAULT_TIMEOUT}
    elif [[ $service_name == *"flux"* ]]; then
        if [[ $service_name == *"fill"* ]]; then
            export TASK_QUEUE_SIZE=${TASK_QUEUE_SIZE:-$HEAVY_TASK_QUEUE_SIZE}
            export TASK_TIMEOUT=${TASK_TIMEOUT:-$HEAVY_TASK_TIMEOUT}
        else
            export TASK_QUEUE_SIZE=${TASK_QUEUE_SIZE:-$HEAVY_TASK_QUEUE_SIZE}
            export TASK_TIMEOUT=${TASK_TIMEOUT:-$HEAVY_TASK_TIMEOUT}
        fi
    elif [[ $service_name == *"longcat"* ]]; then
        export TASK_QUEUE_SIZE=${TASK_QUEUE_SIZE:-$HEAVY_TASK_QUEUE_SIZE}
        export TASK_TIMEOUT=${TASK_TIMEOUT:-$HEAVY_TASK_TIMEOUT}
    elif [[ $service_name == *"omnigen2"* ]]; then
        export TASK_QUEUE_SIZE=${TASK_QUEUE_SIZE:-$HEAVY_TASK_QUEUE_SIZE}
        export TASK_TIMEOUT=${TASK_TIMEOUT:-$HEAVY_TASK_TIMEOUT}
    else
        export TASK_QUEUE_SIZE=${TASK_QUEUE_SIZE:-$DEFAULT_QUEUE_SIZE}
        export TASK_TIMEOUT=${TASK_TIMEOUT:-$DEFAULT_TIMEOUT}
    fi
    
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    echo -e "${YELLOW}Environment Configuration:${NC}"
    echo -e "   GPUs: $NUM_GPUS_TO_USE"
    echo -e "   Queue Size: $TASK_QUEUE_SIZE"
    echo -e "   Task Timeout: $TASK_TIMEOUT seconds"
}

# Start Service
start_service() {
    local api_file=$1
    local service_name=$2
    local port=$3
    local run_mode=$4
    
    echo -e "\n${YELLOW}Checking service file...${NC}"
    if [ ! -f "apis/$api_file" ]; then
        echo -e "${RED}✗ Service file not found: apis/$api_file${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Service file exists: apis/$api_file${NC}"
    
    # Check port
    echo -e "\n${YELLOW}Checking port $port...${NC}"
    if ! check_port $port; then
        echo -e "${RED}Port check failed. Please resolve port conflict and retry.${NC}"
        exit 1
    fi
    
    case $run_mode in
        1)
            echo -e "\n${GREEN}Starting $service_name (Foreground)...${NC}"
            echo -e "${BLUE}Access URL: http://localhost:$port${NC}"
            echo -e "${BLUE}API Docs: http://localhost:$port/docs${NC}"
            echo -e "${BLUE}Press Ctrl+C to stop the service${NC}"
            echo -e "${CYAN}========================================${NC}"
            python apis/$api_file
            ;;
        2)
            local log_file="${api_file%.*}_$(date +%Y%m%d_%H%M%S).log"
            echo -e "\n${GREEN}Starting $service_name (Background)...${NC}"
            echo -e "${BLUE}Log file: $log_file${NC}"
            echo -e "${BLUE}Access URL: http://localhost:$port${NC}"
            echo -e "${BLUE}View logs: tail -f $log_file${NC}"
            echo -e "${CYAN}========================================${NC}"
            
            nohup python apis/$api_file > "$log_file" 2>&1 &
            local pid=$!
            echo -e "${GREEN}✓ Service started, PID: $pid${NC}"
            echo -e "${YELLOW}To stop service: kill $pid${NC}"
            
            # Wait a few seconds to verify start
            sleep 5
            if ps -p $pid > /dev/null; then
                echo -e "${GREEN}✓ Service running normally${NC}"
                # Attempt health check
                sleep 3
                if curl -s http://localhost:$port/health > /dev/null 2>&1; then
                    echo -e "${GREEN}✓ Service health check passed${NC}"
                else
                    echo -e "${YELLOW}⚠ Service might still be initializing, check logs for details${NC}"
                fi
            else
                echo -e "${RED}✗ Service failed to start, check log: $log_file${NC}"
                exit 1
            fi
            ;;
        3)
            echo -e "\n${GREEN}✓ Configuration check passed, service not started${NC}"
            echo -e "${YELLOW}Manual start command: python apis/$api_file${NC}"
            ;;
        *)
            echo -e "${RED}Invalid run mode${NC}"
            exit 1
            ;;
    esac
}

# Main Function
main() {
    local service_choice=$1
    
    # Run Checks
    check_conda_env
    check_python_deps
    check_gpu
    
    # Show Services
    show_services
    
    # Select Service
    if [ -z "$service_choice" ]; then
        echo -e "\n${YELLOW}Please select a service to start [1-9]:${NC}"
        read -p "Selection: " service_choice
    else
        echo -e "\n${BLUE}Service selected via argument: $service_choice${NC}"
    fi
    
    # Get Info
    service_info=$(get_service_info $service_choice)
    if [ -z "$service_info" ]; then
        echo -e "${RED}Invalid service selection${NC}"
        exit 1
    fi
    
    IFS=',' read -r api_file service_name port <<< "$service_info"
    
    echo -e "\n${BLUE}Selected: $service_name${NC}"
    echo -e "${BLUE}Port: $port${NC}"
    echo -e "${BLUE}File: apis/$api_file${NC}"
    
    # Setup Env Vars
    echo -e "\n${YELLOW}5. Setting up environment variables...${NC}"
    setup_env_vars $api_file
    
    # Start Service (Default to Foreground Mode 1)
    echo -e "\n${YELLOW}6. Starting service...${NC}"
    start_service "$api_file" "$service_name" "$port" 1
}

# Get and Save Server IP
save_server_ip() {
    local ip_file="$SERVER_IP_SAVE_PATH"
    
    echo -e "\n${YELLOW}7. Obtaining Server IP Address...${NC}"
    
    local main_ip=""
    
    # Method 1: ip route (Most accurate for outgoing)
    if command -v ip >/dev/null 2>&1; then
        main_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')
        if [ -n "$main_ip" ]; then
            echo -e "${BLUE}IP via default route: $main_ip${NC}"
        fi
    fi
    
    # Method 2: Traffic analysis (Fallback)
    if [ -z "$main_ip" ]; then
        local max_rx=0
        local best_ip=""
        
        while IFS= read -r line; do
            if [[ $line =~ ^[a-zA-Z0-9]+: ]]; then
                interface=$(echo "$line" | cut -d: -f1)
            elif [[ $line =~ inet\ ([0-9.]+) ]]; then
                ip_addr="${BASH_REMATCH[1]}"
                if [[ $ip_addr != "127.0.0.1" ]]; then
                    rx_bytes=$(ifconfig "$interface" 2>/dev/null | grep "RX packets" | awk '{print $5}')
                    if [[ $rx_bytes =~ ^[0-9]+$ ]] && [ $rx_bytes -gt $max_rx ]; then
                        max_rx=$rx_bytes
                        best_ip=$ip_addr
                    fi
                fi
            fi
        done < <(ifconfig)
        
        if [ -n "$best_ip" ]; then
            main_ip=$best_ip
            echo -e "${BLUE}IP via traffic analysis: $main_ip (RX: $max_rx bytes)${NC}"
        fi
    fi
    
    # Method 3: Hostname
    if [ -z "$main_ip" ]; then
        main_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
        if [ -n "$main_ip" ]; then
            echo -e "${BLUE}IP via hostname: $main_ip${NC}"
        fi
    fi
    
    # Method 4: ifconfig first inet
    if [ -z "$main_ip" ]; then
        main_ip=$(ifconfig 2>/dev/null | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
        if [ -n "$main_ip" ]; then
            echo -e "${BLUE}IP via first interface: $main_ip${NC}"
        fi
    fi
    
    if [ -n "$main_ip" ]; then
        echo -e "${GREEN}✓ Detected Server IP: $main_ip${NC}"
        
        local ip_dir=$(dirname "$ip_file")
        mkdir -p "$ip_dir"
        
        if [ -f "$ip_file" ] && grep -q "^$main_ip$" "$ip_file"; then
            echo -e "${YELLOW}⚠ IP address $main_ip already exists, skipping append${NC}"
        else
            echo "$main_ip" >> "$ip_file"
            echo -e "${GREEN}✓ IP address appended to: $ip_file${NC}"
        fi
        
        echo -e "${BLUE}File Content:${NC}"
        cat "$ip_file"
    else
        echo -e "${RED}✗ Unable to determine Server IP address${NC}"
        return 1
    fi
}

# Run Main
save_server_ip
main $1

echo -e "\n${CYAN}========================================${NC}"
echo -e "${CYAN}Service Launcher Script Completed${NC}"
echo -e "${CYAN}========================================${NC}"