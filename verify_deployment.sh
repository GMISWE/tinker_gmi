#!/bin/bash
# Quick deployment verification script for GMI Wrapper V3

set -e

NAMESPACE="slime-gmi"
SERVICE_NAME="gmi-wrapper"
API_KEY="slime-dev-key"

echo "========================================"
echo "GMI Wrapper V3 Deployment Verification"
echo "========================================"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl."
    exit 1
fi

# Check namespace
echo "1. Checking namespace..."
if kubectl get namespace $NAMESPACE &> /dev/null; then
    echo "   ✓ Namespace '$NAMESPACE' exists"
else
    echo "   ❌ Namespace '$NAMESPACE' not found"
    exit 1
fi

# Check service
echo ""
echo "2. Checking service..."
if kubectl get svc $SERVICE_NAME -n $NAMESPACE &> /dev/null; then
    echo "   ✓ Service '$SERVICE_NAME' exists"
    kubectl get svc $SERVICE_NAME -n $NAMESPACE
else
    echo "   ❌ Service '$SERVICE_NAME' not found in namespace '$NAMESPACE'"
    exit 1
fi

# Check pods
echo ""
echo "3. Checking pods..."
PODS=$(kubectl get pods -n $NAMESPACE -l app=$SERVICE_NAME -o jsonpath='{.items[*].metadata.name}')
if [ -z "$PODS" ]; then
    echo "   ❌ No pods found for service '$SERVICE_NAME'"
    exit 1
fi

echo "   ✓ Found pods: $PODS"

for POD in $PODS; do
    STATUS=$(kubectl get pod $POD -n $NAMESPACE -o jsonpath='{.status.phase}')
    echo "   - $POD: $STATUS"

    if [ "$STATUS" != "Running" ]; then
        echo "   ⚠ Pod is not running. Showing logs:"
        kubectl logs $POD -n $NAMESPACE --tail=20
    fi
done

# Setup port-forward
echo ""
echo "4. Setting up port-forward..."
echo "   Starting port-forward in background..."

# Kill any existing port-forward
pkill -f "port-forward.*$SERVICE_NAME" || true
sleep 2

kubectl port-forward -n $NAMESPACE svc/$SERVICE_NAME 8000:8000 &
PF_PID=$!

# Wait for port-forward to be ready
sleep 3

# Test health endpoint
echo ""
echo "5. Testing health endpoint..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "   ✓ Health endpoint accessible"
    curl -s http://localhost:8000/health | python3 -m json.tool
else
    echo "   ❌ Health endpoint not accessible"
    kill $PF_PID || true
    exit 1
fi

# Test with API key
echo ""
echo "6. Testing authentication..."
HEALTH_STATUS=$(curl -s -H "x-api-key: $API_KEY" http://localhost:8000/health | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")

if [ "$HEALTH_STATUS" = "healthy" ]; then
    echo "   ✓ Authentication working"
    echo "   ✓ GMI Wrapper is healthy"
else
    echo "   ❌ Health check failed: status=$HEALTH_STATUS"
fi

# Check Ray status
echo ""
echo "7. Checking Ray status..."
RAY_INIT=$(curl -s http://localhost:8000/health | python3 -c "import sys, json; print(json.load(sys.stdin)['ray_initialized'])")

if [ "$RAY_INIT" = "True" ]; then
    echo "   ✓ Ray is initialized"
else
    echo "   ⚠ Ray is not initialized"
fi

# Show active clients
ACTIVE_CLIENTS=$(curl -s http://localhost:8000/health | python3 -c "import sys, json; print(json.load(sys.stdin)['active_training_clients'])")
FUTURES_COUNT=$(curl -s http://localhost:8000/health | python3 -c "import sys, json; print(json.load(sys.stdin)['futures_count'])")

echo "   Active training clients: $ACTIVE_CLIENTS"
echo "   Pending futures: $FUTURES_COUNT"

# Summary
echo ""
echo "========================================"
echo "Deployment verification complete!"
echo "========================================"
echo ""
echo "To run tests, use:"
echo "  export GMI_BASE_URL=http://localhost:8000"
echo "  export GMI_API_KEY=$API_KEY"
echo "  cd tinker_gmi && python test_gmi_wrapper.py"
echo ""
echo "Port-forward is running with PID: $PF_PID"
echo "To stop port-forward: kill $PF_PID"
echo ""
