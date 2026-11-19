# Datasets
This folder contains the four MWEiC datasets for Galician MWEs resulting from combining the two creation variants (MWEGenCorp, SentDisjCorp) and two syntactic patterns (noun-adjective, verb-object). Each dataset is partitioned into training, development, and test splits, following an 70/10/20 distribution, with balanced development and test splits, although the final proportions may vary across configurations.

Each TSV file follows a WiC structure with both sentences and their classification label, together with additional information for the MWE.

| Header | Description |
|----------|------|
| mwe_id | Unique identifier for the MWE |
| mwe | Lemmatized MWE |
| label | Boolean indicating whether the MWE has the same meaning in both context sentences |
| id_ex{n} | Unique identifier for context sentence *n* |
| sent_ex{n} | Context sentence *n* |
| head_type_ex{n} | Head's type |
| head_token_ex{n} | Head's token |
| dependent_type_ex{n} | Dependent's type |
| dependent_token_ex{n} | Dependent's token |
| head_index_ex{n} | Token index of the head |
| head_start_char_index_ex{n} | Starting character index of the head |
| head_end_char_index_ex{n} | Ending character index of the head |
| dependent_index_ex{n} | Token index of the dependent |
| dependent_start_char_index_ex{n} | Starting character index of the dependent |
| dependent_end_char_index_ex{n} | Ending character index of the dependent |

Note: *n* denotes the context sentence number (i.e., 1 or 2). All token and character indices start at 0.