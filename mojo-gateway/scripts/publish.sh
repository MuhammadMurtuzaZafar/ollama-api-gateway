#!/bin/bash
# EdgeLLM Package Publishing Script
# Publishes to prefix.dev (Pixi's package registry)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "EdgeLLM Package Publisher"
echo "========================="
echo ""

# Check for rattler-build
if ! command -v rattler-build &> /dev/null; then
    echo -e "${YELLOW}Installing rattler-build...${NC}"
    pixi global install rattler-build
fi

# Check for authentication
if [ -z "$PREFIX_DEV_TOKEN" ]; then
    echo -e "${YELLOW}Warning: PREFIX_DEV_TOKEN not set${NC}"
    echo "To publish, set your prefix.dev API token:"
    echo "  export PREFIX_DEV_TOKEN=your-token-here"
    echo ""
    echo "Get your token at: https://prefix.dev/settings/tokens"
    echo ""
fi

# Build the package
echo "Building package..."
cd "$PROJECT_DIR"

mkdir -p dist

rattler-build build \
    --recipe conda-recipe/recipe.yaml \
    --output-dir dist/ \
    2>&1 | tee dist/build.log

# Find the built package
PACKAGE=$(find dist -name "edgellm-*.conda" | head -1)

if [ -z "$PACKAGE" ]; then
    echo -e "${RED}Build failed: No package found in dist/${NC}"
    exit 1
fi

echo -e "${GREEN}Package built: $PACKAGE${NC}"

# Upload to prefix.dev
if [ -n "$PREFIX_DEV_TOKEN" ]; then
    echo ""
    echo "Uploading to prefix.dev..."

    rattler-build upload prefix \
        --channel edgellm \
        --api-key "$PREFIX_DEV_TOKEN" \
        "$PACKAGE"

    echo -e "${GREEN}Package published successfully!${NC}"
    echo ""
    echo "Install with:"
    echo "  pixi add edgellm --channel https://prefix.dev/edgellm"
else
    echo ""
    echo -e "${YELLOW}Skipping upload (no token)${NC}"
    echo ""
    echo "To upload manually:"
    echo "  rattler-build upload prefix --channel edgellm --api-key YOUR_TOKEN $PACKAGE"
fi

echo ""
echo "Local package available at: $PACKAGE"
echo ""
echo "To install locally:"
echo "  pixi add --path $PACKAGE"
