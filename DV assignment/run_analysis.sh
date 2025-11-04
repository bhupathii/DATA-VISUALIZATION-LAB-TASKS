#!/bin/bash

# Ride Sharing Analysis - Run Script
# This script runs the complete analysis

echo "=========================================="
echo "Ride Sharing Dataset Analysis"
echo "=========================================="
echo ""

# Step 1: Generate dataset (if not exists)
if [ ! -f "ride_sharing_dataset.csv" ]; then
    echo "Step 1: Generating dataset..."
    python3 generate_dataset.py
    echo ""
else
    echo "Step 1: Dataset already exists. Skipping generation."
    echo ""
fi

# Step 2: Run analysis
echo "Step 2: Running complete analysis..."
python3 ride_sharing_analysis.py

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Check the output files in the current directory."

