#!/bin/bash
for i in {1..50}
do
    echo $i
    python compute_burial_matches.py --indices=burial_event_ids.txt --num_batches=50 --batch_index=$i --writedb
done
