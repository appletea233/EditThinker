#!/bin/bash

# User-level Nginx Load Balancer Configuration Script
# No sudo required, uses a local Nginx instance

set -e

# Color Definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
DEFAULT_BACKEND_PORT=8006
DEFAULT_PROXY_PORT=8080

# Function to show usage
show_help() {
    echo -e "${CYAN}Usage: $0 [OPTIONS]${NC}"
    echo -e "Options:"
    echo -e "  -i, --ip-file <path>    Path to server IPs file (Required)"
    echo -e "  -d, --base-dir <path>   Base directory for logs/config (Required)"
    echo -e "  -b, --backend-port <port> Backend port (Default: $DEFAULT_BACKEND_PORT)"
    echo -e "  -p, --proxy-port <port>   Local proxy port (Default: $DEFAULT_PROXY_PORT)"
    echo -e "  -h, --help              Show help"
    echo -e "${YELLOW}Recommended Example:${NC}"
    echo -e "  $0 -i ./ip_server2.txt -d /tmp/nginx_user_data -b 8006 -p 8080"
}

# Parse arguments
IP_FILE=""
BASE_DIR=""
BACKEND_PORT="$DEFAULT_BACKEND_PORT"
PROXY_PORT="$DEFAULT_PROXY_PORT"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--ip-file) IP_FILE="$2"; shift 2 ;;
        -d|--base-dir) BASE_DIR="$2"; shift 2 ;;
        -b|--backend-port) BACKEND_PORT="$2"; shift 2 ;;
        -p|--proxy-port) PROXY_PORT="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; show_help; exit 1 ;;
    esac
done

if [ -z "$IP_FILE" ] || [ -z "$BASE_DIR" ]; then
    echo -e "${RED}Error: --ip-file and --base-dir are required.${NC}"; show_help; exit 1
fi

# 1. Permission Check
echo -e "${YELLOW}Checking write permissions for $BASE_DIR...${NC}"
mkdir -p "$BASE_DIR" 2>/dev/null || {
    echo -e "${RED}Error: Cannot create directory $BASE_DIR. Permission denied.${NC}"
    echo -e "${YELLOW}Tip: Use a directory you own, like /tmp/nginx_data or /home/yourname/nginx${NC}"
    exit 1
}
if [ ! -w "$BASE_DIR" ]; then
    echo -e "${RED}Error: Directory $BASE_DIR is not writable.${NC}"; exit 1
fi

# Convert BASE_DIR to absolute path
BASE_DIR=$(cd "$BASE_DIR" && pwd)
NGINX_CONF_DIR="$BASE_DIR/conf"
NGINX_LOG_DIR="$BASE_DIR/logs"
NGINX_PID_FILE="$BASE_DIR/nginx.pid"
SCRIPTS_DIR="$BASE_DIR/scripts"

# Read Server IPs
read_server_ips() {
    echo -e "\n${YELLOW}1. Reading server IP list...${NC}"
    if [ ! -f "$IP_FILE" ]; then echo -e "${RED}✗ IP file not found: $IP_FILE${NC}"; exit 1; fi
    SERVER_IPS=($(grep -v '^$' "$IP_FILE"))
    if [ ${#SERVER_IPS[@]} -eq 0 ]; then echo -e "${RED}✗ No valid IPs found${NC}"; exit 1; fi
    echo -e "${GREEN}✓ Found ${#SERVER_IPS[@]} server IPs${NC}"
}

# Create Directories
create_nginx_dirs() {
    echo -e "\n${YELLOW}2. Creating directories...${NC}"
    mkdir -p "$NGINX_CONF_DIR" "$NGINX_LOG_DIR" "$SCRIPTS_DIR"
    
    # Use specific local tmp dir
    NGINX_TMP_DIR="/tmp/nginx_${USER}_${PROXY_PORT}_tmp"
    rm -rf "$NGINX_TMP_DIR" # Clean old temp
    mkdir -p "$NGINX_TMP_DIR"/{client_body,proxy,fastcgi,uwsgi,scgi}
    
    echo -e "${GREEN}✓ Config Dir: $NGINX_CONF_DIR${NC}"
}

# Generate Main Config
generate_main_config() {
    echo -e "\n${YELLOW}3. Generating config...${NC}"
    cat > "$NGINX_CONF_DIR/nginx.conf" << EOF
worker_processes auto;
# Use local error log
error_log $NGINX_LOG_DIR/error.log warn;
pid $NGINX_PID_FILE;

events { worker_connections 1024; }

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    client_body_temp_path $NGINX_TMP_DIR/client_body;
    proxy_temp_path $NGINX_TMP_DIR/proxy;
    fastcgi_temp_path $NGINX_TMP_DIR/fastcgi;
    uwsgi_temp_path $NGINX_TMP_DIR/uwsgi;
    scgi_temp_path $NGINX_TMP_DIR/scgi;
    
    access_log $NGINX_LOG_DIR/access.log;
    sendfile on;
    keepalive_timeout 65;
    
    include $NGINX_CONF_DIR/multimodal_api.conf;
}
EOF
}

# Generate Load Balancer Config
generate_load_balancer_config() {
    echo -e "\n${YELLOW}4. Generating LB config...${NC}"
    cat > "$NGINX_CONF_DIR/multimodal_api.conf" << EOF
upstream multimodal_backend {
EOF
    for ip in "${SERVER_IPS[@]}"; do
        echo "    server $ip:$BACKEND_PORT max_fails=3 fail_timeout=30s;" >> "$NGINX_CONF_DIR/multimodal_api.conf"
    done
    cat >> "$NGINX_CONF_DIR/multimodal_api.conf" << EOF
    keepalive 32;
}
server {
    listen $PROXY_PORT;
    server_name localhost;
    access_log $NGINX_LOG_DIR/api_access.log;
    error_log $NGINX_LOG_DIR/api_error.log;
    
    client_max_body_size 500M;
    proxy_connect_timeout 300s;
    proxy_send_timeout 900s;
    proxy_read_timeout 900s;
    
    location / {
        proxy_pass http://multimodal_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header Connection "";
    }
    location /nginx-health { return 200 "healthy\n"; }
}
EOF
}

# Test Config
test_config() {
    echo -e "\n${YELLOW}5. Testing configuration...${NC}"
    NGINX_BIN=$(which nginx 2>/dev/null || echo "/usr/sbin/nginx")
    
    if [ ! -f "$NGINX_CONF_DIR/nginx.conf" ]; then
        echo -e "${RED}FATAL: Config file missing${NC}"; exit 1
    fi

    # Use -g to prevent /var/log permission error
    if "$NGINX_BIN" -t -c "$NGINX_CONF_DIR/nginx.conf" -g "error_log stderr;"; then
        echo -e "${GREEN}✓ Syntax OK${NC}"
    else
        echo -e "${RED}✗ Syntax Error${NC}"; exit 1
    fi
}

# Start Nginx
start_nginx() {
    echo -e "\n${YELLOW}6. Starting Nginx...${NC}"
    NGINX_BIN=$(which nginx 2>/dev/null || echo "/usr/sbin/nginx")
    
    if [ -f "$NGINX_PID_FILE" ]; then
        kill -TERM $(cat "$NGINX_PID_FILE") 2>/dev/null || true
        rm -f "$NGINX_PID_FILE"
        sleep 1
    fi
    
    if netstat -tlnp 2>/dev/null | grep -q ":$PROXY_PORT "; then
        echo -e "${RED}Error: Port $PROXY_PORT is already in use.${NC}"; exit 1
    fi

    # Start with global override for error_log
    "$NGINX_BIN" -c "$NGINX_CONF_DIR/nginx.conf" -g "error_log stderr;"
    
    sleep 2
    if [ -f "$NGINX_PID_FILE" ] && kill -0 $(cat "$NGINX_PID_FILE") 2>/dev/null; then
        echo -e "${GREEN}✓ Nginx running (PID: $(cat $NGINX_PID_FILE))${NC}"
    else
        echo -e "${RED}✗ Start failed. Logs:${NC}"
        cat "$NGINX_LOG_DIR/error.log" 2>/dev/null
        exit 1
    fi
}

# Create Management Scripts
create_management_scripts() {
    echo -e "\n${YELLOW}7. Creating scripts...${NC}"
    NGINX_BIN=$(which nginx 2>/dev/null || echo "/usr/sbin/nginx")
    
    # Start script
    cat > "$SCRIPTS_DIR/start_nginx.sh" << EOF
#!/bin/bash
"$NGINX_BIN" -c "$NGINX_CONF_DIR/nginx.conf" -g "error_log stderr;"
echo "Started."
EOF
    
    # Stop script
    cat > "$SCRIPTS_DIR/stop_nginx.sh" << EOF
#!/bin/bash
"$NGINX_BIN" -s quit -c "$NGINX_CONF_DIR/nginx.conf" -g "error_log stderr;"
echo "Stopped."
EOF
    
    chmod +x "$SCRIPTS_DIR"/*.sh
    echo -e "${GREEN}✓ Scripts created in $SCRIPTS_DIR${NC}"
}

# Get Local IP
get_local_ip() {
    local ip=""
    # Method 1: ip route (Best for finding the actual outgoing IP)
    if command -v ip >/dev/null 2>&1; then
        ip=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')
    fi
    
    # Method 2: hostname -I (Fallback)
    if [ -z "$ip" ] && command -v hostname >/dev/null 2>&1; then
        ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    
    # Method 3: ifconfig (Fallback)
    if [ -z "$ip" ] && command -v ifconfig >/dev/null 2>&1; then
         ip=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -n1 | awk '{print $2}' | tr -d 'addr:')
    fi

    echo "${ip:-127.0.0.1}"
}

# Show Summary
show_usage_summary() {
    local host_ip=$(get_local_ip)
    
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}     Setup Complete${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo -e "${GREEN}Load Balancer Access:${NC}"
    echo -e "  - Local Access:   http://localhost:$PROXY_PORT/"
    echo -e "  - Network Access: http://$host_ip:$PROXY_PORT/  <-- (Use this IP)"
    echo -e "  - Docs:           http://$host_ip:$PROXY_PORT/docs"
    echo -e ""
    echo -e "${GREEN}Backend Servers:${NC}"
    for ip in "${SERVER_IPS[@]}"; do
        echo -e "  - $ip:$BACKEND_PORT"
    done
    echo -e ""
    echo -e "${YELLOW}Management Commands:${NC}"
    echo -e "  cd $SCRIPTS_DIR"
    echo -e "  ./start_nginx.sh"
    echo -e "  ./stop_nginx.sh"
    echo -e "  tail -f $NGINX_LOG_DIR/api_access.log"
    echo -e "${CYAN}========================================${NC}"
}

# Get Local IP
get_local_ip() {
    local ip=""
    # Method 1: ip route (Best for finding the actual outgoing IP)
    if command -v ip >/dev/null 2>&1; then
        ip=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')
    fi
    
    # Method 2: hostname -I (Fallback)
    if [ -z "$ip" ] && command -v hostname >/dev/null 2>&1; then
        ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    
    # Method 3: ifconfig (Fallback)
    if [ -z "$ip" ] && command -v ifconfig >/dev/null 2>&1; then
         ip=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -n1 | awk '{print $2}' | tr -d 'addr:')
    fi

    echo "Please use ip:port as: "
    echo "${ip:-127.0.0.1}:${DEFAULT_PROXY_PORT}"
}

# Run
read_server_ips
create_nginx_dirs
generate_main_config
generate_load_balancer_config
test_config
start_nginx
create_management_scripts
show_usage_summary
get_local_ip