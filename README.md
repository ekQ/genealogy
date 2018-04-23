# Genealogical Network Inference and Analysis
Code for the following paper:

Malmi, E., Gionis, A., Solin, A. "Computationally Inferred Genealogical Networks Uncover Long-Term Trends in Assortative Mating". In Proceedings of the Web Conference (WWW 2018), Lyon, France, 2018.

## Data
To get access to the data, email firstname.lastname@gmail.com (first author), and you will receive a link to a download site.

## Running Genealogical Network Inference

First, link the burial records into birth records by running
```
python compute_matches.py --indices=birth_event_ids.txt --num_batches=50 --batch_index=$SLURM_ARRAY_TASK_ID 
                          --output=reconstructions/all_parent_edges
```
for `batch_index` values from 1 to 50 (can be run in parallel) in order to link the HisKi dataset in 50 batches.

Second, run
```
sh serial_burial_db_write.sh
```

Third, run the following command

```
python compute_matches.py --indices=birth_event_ids.txt --num_batches=50 --batch_index=1 \
                          --output=reconstructions/all_parent_edges
```
for `batch_index` values from 1 to 50 (can be run in parallel) in order to link the HisKi dataset in 50 batches.
