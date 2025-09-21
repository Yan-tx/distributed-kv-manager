#!/bin/bash

CRAIL_HOME=/root/crail
CRAIL_ADDR="crail://192.168.100.10:9060"

set -e

echo "=== Creating directory ==="
$CRAIL_HOME/bin/crail fs -fs $CRAIL_ADDR -mkdir /testdir

echo "=== Creating empty file ==="
# Create an empty file by uploading empty content
echo -n "" | $CRAIL_HOME/bin/crail fs -fs $CRAIL_ADDR -put - /testdir/emptyfile

echo "=== Writing to file ==="
echo "hello crail" | $CRAIL_HOME/bin/crail fs -fs $CRAIL_ADDR -put - /testdir/hello.txt

echo "=== Reading file content ==="
$CRAIL_HOME/bin/crail fs -fs $CRAIL_ADDR -cat /testdir/hello.txt

echo
echo "=== Listing directory contents ==="
$CRAIL_HOME/bin/crail fs -fs $CRAIL_ADDR -ls /testdir
# /root/crail/bin/crail fs -fs "crail://192.168.100.10:9060" -ls /crail/kvcache/

echo "=== Copying file ==="
$CRAIL_HOME/bin/crail fs -fs $CRAIL_ADDR -cp /testdir/hello.txt /testdir/hello_copy.txt

echo "=== Deleting file ==="
$CRAIL_HOME/bin/crail fs -fs $CRAIL_ADDR -rm /testdir/hello_copy.txt

echo "=== Deleting directory recursively ==="
$CRAIL_HOME/bin/crail fs -fs $CRAIL_ADDR -rm -r /testdir
# /root/crail/bin/crail fs -fs "crail://192.168.100.10:9060" -rm -r /crail/kvcache

echo "=== Test completed ==="
