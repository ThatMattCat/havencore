#!/bin/bash

# Setup script for HavenCore volume directories
# This script creates the necessary directory structure for bind mounts
# and sets appropriate permissions

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_status "Setting up HavenCore volume directories..."
print_status "Project root: $PROJECT_ROOT"

# Create volumes directory structure
VOLUMES_DIR="$PROJECT_ROOT/volumes"

print_status "Creating volume directories..."
mkdir -p "$VOLUMES_DIR/postgres_data/data"
mkdir -p "$VOLUMES_DIR/qdrant_storage" 
mkdir -p "$VOLUMES_DIR/models"

print_status "Setting permissions for PostgreSQL data directory..."
# PostgreSQL in Docker runs as user postgres (UID 999)
# We need to ensure the host directory is writable by this user
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # On Linux, set proper ownership
    if command -v sudo >/dev/null 2>&1; then
        print_status "Setting ownership to UID 999 (postgres user in container)..."
        sudo chown -R 999:999 "$VOLUMES_DIR/postgres_data/data"
    else
        print_warning "sudo not available. You may need to manually set ownership:"
        print_warning "  sudo chown -R 999:999 $VOLUMES_DIR/postgres_data/data"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # On macOS, Docker Desktop handles permissions differently
    print_status "Detected macOS. Setting permissive permissions..."
    chmod -R 755 "$VOLUMES_DIR/postgres_data/data"
else
    print_warning "Unknown OS. Setting permissive permissions..."
    chmod -R 755 "$VOLUMES_DIR/postgres_data/data"
fi

# Set permissions for other directories
print_status "Setting permissions for other volume directories..."
chmod -R 755 "$VOLUMES_DIR/qdrant_storage"
chmod -R 755 "$VOLUMES_DIR/models"

# Create .gitkeep files to preserve directory structure in git
print_status "Creating .gitkeep files..."
touch "$VOLUMES_DIR/postgres_data/.gitkeep"
touch "$VOLUMES_DIR/qdrant_storage/.gitkeep"
touch "$VOLUMES_DIR/models/.gitkeep"

# Verify directory structure
print_status "Verifying directory structure..."
if [[ -d "$VOLUMES_DIR/postgres_data/data" && -d "$VOLUMES_DIR/qdrant_storage" && -d "$VOLUMES_DIR/models" ]]; then
    print_status "✓ Volume directories created successfully!"
    print_status "Directory structure:"
    ls -la "$VOLUMES_DIR/"
    ls -la "$VOLUMES_DIR/postgres_data/"
else
    print_error "Failed to create some directories. Please check permissions."
    exit 1
fi

print_status "Volume setup complete!"
print_status ""
print_status "Next steps:"
print_status "1. Configure your .env file: cp .env.tmpl .env"
print_status "2. Edit .env with your settings"
print_status "3. Start HavenCore: docker compose up -d"