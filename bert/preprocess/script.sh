#!/bin/bash
while true
do
	timeout 300 python extract_facemesh.py > fm_dump.txt
	echo "restart facemesh ..."
	sleep 1
done
