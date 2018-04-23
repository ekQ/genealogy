# Genealogical Network Inference and Analysis
Code for the following paper:

Malmi, E., Gionis, A., Solin, A. "Computationally Inferred Genealogical Networks Uncover Long-Term Trends in Assortative Mating". In Proceedings of the Web Conference (WWW 2018), Lyon, France, 2018.

## Data
To get access to the data, email firstname.lastname@gmail.com (first author), and you will receive a link to a download site.

## Running Genealogical Network Inference

Once you have downloaded and copied the HisKi database into the data directory, do the following three steps.

First, link the burial records into birth records in, e.g., 50 batches by running
```
python compute_burial_matches.py --indices=burial_event_ids.txt --num_batches=50 --batch_index=1 \
                                 --output=reconstructions/burial_birth_links_single
```
for `batch_index` values from 1 to 50 (can be run in parallel).

Second, store the burial-birth links into the HisKi DB by running
```
sh serial_burial_db_write.sh
```

Third, infer the parents of each person in the HisKi DB in 50 batches by running
```
python compute_matches.py --indices=birth_event_ids.txt --num_batches=50 --batch_index=1 \
                          --output=reconstructions/all_parent_edges
```
for `batch_index` values from 1 to 50 (can be run in parallel).

## Running Assortative Mating Analysis

(The code and documentation to be added.)
