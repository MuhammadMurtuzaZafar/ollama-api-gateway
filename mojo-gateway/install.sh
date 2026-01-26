#!/bin/bash
# EdgeLLM Installation Script
# Usage: curl -fsSL https://raw.githubusercontent.com/user/edgellm/main/install.sh | bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
EDGELLM_VERSION="${EDGELLM_VERSION:-latest}"
INSTALL_DIR="${EDGELLM_HOME:-$HOME/.edgellm}"
BIN_DIR="${EDGELLM_BIN:-$HOME/.local/bin}"
REPO_URL="https://github.com/umerkhan95/EdgeLLM.git"

echo -e "${BLUE}"
echo "  _____ ____   ____ _____   _     _     __  __ "
echo " | ____|  _ \ / ___| ____| | |   | |   |  \/  |"
echo " |  _| | | | | |  _|  _|   | |   | |   | |\/| |"
echo " | |___| |_| | |_| | |___  | |___| |___| |  | |"
echo " |_____|____/ \____|_____| |_____|_____|_|  |_|"
echo -e "${NC}"
echo "EdgeLLM Installer - High-performance LLM inference for edge devices"
echo ""

# Detect OS and architecture
detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Linux)
            case "$ARCH" in
                x86_64) PLATFORM="linux-64" ;;
                aarch64|arm64) PLATFORM="linux-aarch64" ;;
                *) echo -e "${RED}Unsupported architecture: $ARCH${NC}"; exit 1 ;;
            esac
            ;;
        Darwin)
            case "$ARCH" in
                x86_64)
                    echo -e "${YELLOW}Warning: macOS x86_64 requires Docker for Mojo${NC}"
                    PLATFORM="osx-64"
                    USE_DOCKER=true
                    ;;
                arm64) PLATFORM="osx-arm64" ;;
                *) echo -e "${RED}Unsupported architecture: $ARCH${NC}"; exit 1 ;;
            esac
            ;;
        *)
            echo -e "${RED}Unsupported OS: $OS${NC}"
            exit 1
            ;;
    esac

    echo -e "${GREEN}Detected platform: $PLATFORM${NC}"
}

# Check for required tools
check_dependencies() {
    echo "Checking dependencies..."

    # Check for git
    if ! command -v git &> /dev/null; then
        echo -e "${RED}Git is required but not installed.${NC}"
        echo "Install git: https://git-scm.com/downloads"
        exit 1
    fi

    # Check for pixi
    if ! command -v pixi &> /dev/null; then
        echo -e "${YELLOW}Pixi not found. Installing...${NC}"
        curl -fsSL https://pixi.sh/install.sh | bash
        export PATH="$HOME/.pixi/bin:$PATH"

        if ! command -v pixi &> /dev/null; then
            echo -e "${RED}Failed to install Pixi. Please install manually:${NC}"
            echo "curl -fsSL https://pixi.sh/install.sh | bash"
            exit 1
        fi
    fi

    echo -e "${GREEN}All dependencies satisfied${NC}"
}

# Clone or update repository
clone_repo() {
    echo "Setting up EdgeLLM..."

    if [ -d "$INSTALL_DIR" ]; then
        echo "Updating existing installation..."
        cd "$INSTALL_DIR"
        git pull origin main 2>/dev/null || true
    else
        echo "Cloning EdgeLLM repository..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR/mojo-gateway"
    fi
}

# Build EdgeLLM
build_edgellm() {
    echo "Building EdgeLLM..."
    cd "$INSTALL_DIR/mojo-gateway" 2>/dev/null || cd "$INSTALL_DIR"

    # Install dependencies and build
    pixi install
    pixi run build-cli

    if [ ! -f "bin/edgellm" ]; then
        echo -e "${RED}Build failed: bin/edgellm not found${NC}"
        exit 1
    fi

    echo -e "${GREEN}Build successful${NC}"
}

# Install to PATH
install_binary() {
    echo "Installing EdgeLLM to $BIN_DIR..."

    mkdir -p "$BIN_DIR"

    # Create wrapper script
    cat > "$BIN_DIR/edgellm" << 'WRAPPER'
#!/bin/bash
EDGELLM_HOME="${EDGELLM_HOME:-$HOME/.edgellm}"
exec "$EDGELLM_HOME/mojo-gateway/bin/edgellm" "$@"
WRAPPER

    chmod +x "$BIN_DIR/edgellm"

    # Add to PATH if needed
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        echo ""
        echo -e "${YELLOW}Add this to your shell profile (.bashrc, .zshrc, etc.):${NC}"
        echo -e "  export PATH=\"$BIN_DIR:\$PATH\""
        echo ""
    fi
}

# Create models directory
setup_models_dir() {
    mkdir -p "$HOME/.edgellm/models"
    echo -e "${GREEN}Models directory: $HOME/.edgellm/models${NC}"
}

# Verify installation
verify_installation() {
    echo ""
    echo "Verifying installation..."

    if "$BIN_DIR/edgellm" --help > /dev/null 2>&1; then
        echo -e "${GREEN}EdgeLLM installed successfully!${NC}"
        echo ""
        echo "Usage:"
        echo "  edgellm models        # List available models"
        echo "  edgellm pull <model>  # Download a model"
        echo "  edgellm run <model>   # Run interactive chat"
        echo "  edgellm serve <model> # Start API server"
        echo ""
        echo "Example:"
        echo "  edgellm pull smollm-135m"
        echo "  edgellm run smollm-135m"
        return 0
    else
        echo -e "${RED}Installation verification failed${NC}"
        return 1
    fi
}

# Uninstall function
uninstall() {
    echo "Uninstalling EdgeLLM..."
    rm -rf "$INSTALL_DIR"
    rm -f "$BIN_DIR/edgellm"
    echo -e "${GREEN}EdgeLLM uninstalled${NC}"
}

# Main installation flow
main() {
    case "${1:-install}" in
        install)
            detect_platform
            check_dependencies
            clone_repo
            build_edgellm
            install_binary
            setup_models_dir
            verify_installation
            ;;
        uninstall)
            uninstall
            ;;
        update)
            detect_platform
            clone_repo
            build_edgellm
            echo -e "${GREEN}EdgeLLM updated successfully${NC}"
            ;;
        *)
            echo "Usage: $0 {install|uninstall|update}"
            exit 1
            ;;
    esac
}

main "$@"
