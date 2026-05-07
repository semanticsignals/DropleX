#!/bin/bash
# Monitor benchmark logs from the Android app

echo "========================================="
echo "Benchmark Monitor for TabletCap2"
echo "========================================="
echo "Monitoring BENCHMARK logs... Press Ctrl+C to stop"
echo ""

# Clear previous logs and start monitoring
adb logcat -c
adb logcat -s BENCHMARK:E | while read line; do
    echo "$line"
done
