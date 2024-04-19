# Dubo-SQL v1

## Preparation
Before running inference on the dev set, we fine tune a model with the BIRD-SQL training data. The notebook `fine_tuning_dubo_v1.ipynb` shows how we fine tuned the model.

## Execution

1. Copy `env.template` to `.env` and fill in the OpenAI API key.

2. Create a virtual environment and install its dependencies

```
python3 -m venv env && . env/bin/activate
pip install -r requirements.txt
```

3. Execute the code. For example, this code will execute against the `dev` databases:

```
python dubo_bird.py --set dev --json dev.json --db dev_databases
```

Or if your data is in the `test` directory:

```
python dubo_bird.py --set test --json test.json --db test_databases
```

The command-line tool expects a directory structure like the one used in the BIRD `dev` data:
```
.
└── dev
   ├── dev.json
   ├── dev.sql
   ├── dev_databases
   ├── dev_tables.json
   └── dev_tied_append.json
```

After execution, the tool produces the file `./predict_{SET}.json`, which can be used for evaluation.

## How it works

The `dubo_bird.py` script runs a fine-tuned OpenAI model against the BIRD data set.
The script will
- look for the input JSON file at `./{SET}/{JSON_FILE}`
- look for the databases to execute against at `./{SET}/{DB_DIR}`
- save output to `./predict_{SET}.json`


usage: dubo_bird.py [-h] [--set SET] [--json JSON] [--db DB] [--dryrun DRYRUN]

  --set SET        Set value for the directory containing input JSON and databases.
  --json JSON      Set file name for input JSON.
  --db DB          Set subdirectory name containing sqlite database directories.
  --dryrun DRYRUN  Set value for DRYRUN variable.

## About us

Dubo is built by [Mercator Technologies](https://mercator.tech/), a Y Combinator-backed, San Francisco-based startup focused on building an AI-assisted data analytics platform for writing faster, cheaper, and smarter SQL.
