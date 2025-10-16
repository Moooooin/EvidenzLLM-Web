#!/bin/bash

# EvidenzLLM Web Chat - Test Runner
# Runs all test suites sequentially

echo "========================================================================"
echo "EvidenzLLM Web Chat - Running All Tests"
echo "========================================================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Counter for passed/failed tests
PASSED=0
FAILED=0

# Array of test files
TESTS=(
    "test_model_basic.py"
    "test_retrieval_manual.py"
    "test_gemini_api.py"
    "test_pipeline_structure.py"
    "test_flask_api.py"
    "test_frontend.py"
)

# Run each test
for test in "${TESTS[@]}"; do
    echo "Running $test..."
    echo "------------------------------------------------------------------------"
    
    if python "tests/$test"; then
        echo -e "${GREEN}✓ $test PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ $test FAILED${NC}"
        ((FAILED++))
        # Uncomment to stop on first failure
        # exit 1
    fi
    
    echo ""
done

# Summary
echo "========================================================================"
echo "Test Summary"
echo "========================================================================"
echo "Total tests: $((PASSED + FAILED))"
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
    exit 1
else
    echo "Failed: 0"
    echo ""
    echo -e "${GREEN}ALL TESTS PASSED ✓${NC}"
    exit 0
fi
