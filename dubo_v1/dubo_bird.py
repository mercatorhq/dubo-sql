# Execute with python dubo_bird.py --set bird-data-train --json train.json --db train_databases
import argparse
import glob
import json
import os
import re
import sqlite3
import time
from typing import List, Optional

import openai
from openai.error import Timeout as OpenAITimeout
import pandas as pd
from requests.exceptions import Timeout as RequestsTimeout
import sqlglot
from sqlglot import expressions, Expression
import tiktoken

OPENAI_API_KEY = os.getenv('OPENAI_KEY')

parser = argparse.ArgumentParser(description="Set command-line arguments.")
parser.add_argument(
    "--set", type=str, default="dev", help="Set value for SET variable."
)
parser.add_argument(
    "--json", type=str, default="dev.json", help="Set value for JSON_FILE variable."
)
parser.add_argument(
    "--db", type=str, default="dev_databases", help="Set value for DB_DIR variable."
)
parser.add_argument(
    "--dryrun", type=bool, default=False, help="Set value for DRYRUN variable."
)
args = parser.parse_args()

SET = args.set
JSON_FILE = "./" + SET + "/" + args.json
DB_DIR = "./" + SET + "/" + args.db

if not os.path.exists(DB_DIR):
    raise ValueError(f"DB_DIR {DB_DIR} does not exist.")
if not os.path.exists(JSON_FILE):
    raise ValueError(f"JSON_FILE {JSON_FILE} does not exist.")

dryrun = args.dryrun

DEFAULT_MODEL = (
    "ft:gpt-3.5-turbo-0613:mercator:dev:8GFLwXLG"  # version trained with train+dev data
)
# DEFAULT_MODEL = "ft:gpt-3.5-turbo-0613:mercator:bird-qna-take-two:8FwjK0wH"  # version trained with train data only

encoding = tiktoken.get_encoding("cl100k_base")


def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def get_create_table_and_data(db_path: str, num_rows: int = 5) -> List[str]:
    MAX_TOKENS = 2550  # The limit required so OpenAI doesn't complain after we reformat
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    while num_rows >= 0:
        # Query the sqlite_master table to get the CREATE TABLE statements
        cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = cursor.fetchall()

        output_statements = []

        for table_name, create_statement in tables:
            # "INTEGER" -> "INT"
            create_statement = create_statement.replace("INTEGER", "INT")

            # remove comments
            create_statement = re.sub(
                r"--.*$", "", create_statement, flags=re.MULTILINE
            )
            create_statement = "\n".join(
                [line for line in create_statement.split("\n") if line.strip()]
            )

            # Condense whitespace
            create_statement = " ".join(create_statement.split())

            # First, add the create statement
            output_statements.append(create_statement + ";")

            # Fetch sample data
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT ?", (num_rows,))
            sample_rows = cursor.fetchall()

            # For each row, create an INSERT INTO statement
            for row in sample_rows:
                formatted_values = []
                for idx, value in enumerate(row):
                    if isinstance(value, str):
                        formatted_value = value.replace("\n", " ")
                        formatted_value = formatted_value.replace("'", '"')
                        formatted_value = formatted_value[:100]
                        formatted_values.append(f"'{formatted_value}'")
                    elif value is None:
                        formatted_values.append("NULL")
                    else:
                        formatted_values.append(str(value))
                values_str = ",".join(formatted_values)

                # Check if table_name contains a space or dash and wrap it in double quotes if it does
                if " " in table_name or "-" in table_name:
                    formatted_table_name = f'"{table_name}"'
                else:
                    formatted_table_name = table_name

                insert_statement = (
                    f"INSERT INTO {formatted_table_name} VALUES ({values_str});"
                )
                output_statements.append(insert_statement)

        msgs = [{"role": "user", "content": "\n".join(output_statements)}]
        token_count = num_tokens_from_messages(msgs)

        if token_count < MAX_TOKENS:
            cursor.close()
            conn.close()
            return output_statements
        elif num_rows > 0:
            num_rows -= 1
            continue
        else:
            final_statements = []
            for statement in output_statements:
                final_statements.append(statement)
                msgs = [{"role": "user", "content": "\n".join(final_statements)}]
                token_count = num_tokens_from_messages(msgs)

                if token_count > MAX_TOKENS:
                    cursor.close()
                    conn.close()
                    final_statements.pop()
                    return final_statements
    cursor.close()
    conn.close()
    raise ValueError(f"Even with 0 rows, token count is too high!")


def clean_creates(sql_text: str) -> str:
    """While these fields might be useful for some purposes, I've honestly
    needed them so rarely as a data scientist that we are going to exclude them
    """

    def replace_(node: Expression) -> Optional[Expression]:
        if isinstance(
            node,
            (
                expressions.ColumnConstraint,
                expressions.PrimaryKey,
            ),
        ):
            return None
        return node

    return str(sqlglot.parse_one(sql_text).transform(replace_))


def hard_replace__clean_creates(sql_text: str):
    """The backticks and double-quotes are always equivalent in bird
    # but sqlglot cannot yet handle the backticks
    """
    try:
        return clean_creates(
            sql_text.replace("`", '"')
            .replace("WITHOUT ROWID", "")
            .replace("on update cascade", "")
            .replace("ON UPDATE CASCADE", "")
            .replace("on delete cascade", "")
            .replace("ON DELETE CASCADE", "")
            .replace("references staff", "")
        )  # .sql()
    except Exception:
        raise


def read_in_all_sqlite_dbs():
    """Read in all the sqlite databases from the bird data"""
    dirs = glob.glob(DB_DIR + "/*")
    statements = []
    for d in dirs:
        if os.path.isfile(d):
            continue
        dbname = d.split("/")[-1]
        sqlite_db_path = os.path.join(d, dbname + ".sqlite")
        assert os.path.exists(d), f"DB {d} does not exist!"
        ddl_list = get_create_table_and_data(sqlite_db_path)
        for ddl in ddl_list:
            statements.append((dbname, hard_replace__clean_creates(ddl)))

    return statements


def make_x(tables, db_id, ideal_sql):
    """Make the x and y for the training data"""
    return tables, db_id, ideal_sql


ddl_statements = read_in_all_sqlite_dbs()

df_ddl = pd.DataFrame(ddl_statements)
df_ddl.columns = ["db_id", "ddl"]
df_ddl = df_ddl.groupby("db_id")["ddl"].agg("\n".join).reset_index(name="ddl")


def format_ddl(ddl_str):
    formatted_ddls = []

    # Split the ddl_str by "CREATE TABLE"
    create_tables = re.split(r"(?i)CREATE TABLE", ddl_str)

    for ct in create_tables:
        if not ct.strip():
            continue

        # Extract table name from the current CREATE TABLE section
        table_name_match = re.search(r'^\s*("?[\w\s-]+"?|[\w\s-]+)', ct)

        # table_name = table_name_match.group(1) if table_name_match else "Unknown Table"
        table_name = (
            table_name_match.group(1).strip() if table_name_match else "Unknown Table"
        )

        # Split the current section at "INSERT INTO"
        splits = ct.split("INSERT INTO")

        # Extract column names and remove the table name from it
        # columns = splits[0].replace(table_name, "").strip().replace("(", "").replace(")", "")
        # columns = re.sub(r'^\s*' + re.escape(table_name), '', splits[0]).strip().replace("(", "").replace(")", "")
        columns = re.sub(r"^\s*" + re.escape(table_name), "", splits[0]).strip()
        if columns.startswith("(") and columns.endswith(")"):
            columns = columns[1:-1]

        columns = " ".join(columns.split())

        # Process INSERT statements
        cleaned_table_name = table_name.strip('"')
        insert_statements = [
            split.replace(f"{cleaned_table_name} VALUES", "").strip()
            for split in splits[1:]
        ]

        # Remove parentheses from the INSERT statements
        insert_statements = [
            stmt.replace("(", "").replace(")", "") for stmt in insert_statements
        ]

        # Combine the statements for the current table and append to formatted_ddls
        # formatted_ddl = "# Table: " + table_name + "\n" + columns + "\n" + "\n".join(insert_statements)
        if "VARBINARY" in ct:
            formatted_ddl = table_name + " (" + columns + ");"
        else:
            formatted_ddl = (
                table_name
                + " ("
                + columns
                + ");\nINSERT INTO "
                + table_name
                + " VALUES\n("
                + ")\n(".join(insert_statements)
                + ");"
            )
        formatted_ddls.append(formatted_ddl)

    # Join all the formatted sections with newline characters
    return "\n".join(formatted_ddls)


# Apply the formatting function to the 'ddl' column
df_ddl["ddl"] = df_ddl["ddl"].apply(format_ddl)

df_question = pd.read_json(JSON_FILE)
joined_df = pd.merge(df_question, df_ddl, on=["db_id"])

df = pd.DataFrame(
    joined_df.apply(
        lambda x: make_x(
            x["ddl"]
            + "\n## The user has asked:\n"
            + x["question"]
            + "\nNOTE: "
            + x["evidence"],
            x["db_id"],
            x["SQL"],
        ),
        axis=1,
    ).tolist()
)
df.columns = ["user_prompt", "db_id", "ideal_assistant_response"]

MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds


def create_response(model, msgs, response_max_tokens=1000):
    num_tokens = num_tokens_from_messages(msgs) + response_max_tokens

    if num_tokens > 4096:
        model = "gpt-3.5-turbo-16k"

    for attempt in range(MAX_RETRIES):
        if dryrun:
            return {"choices": [{"message": {"content": "SELECT 1"}}]}
        try:
            return openai.ChatCompletion.create(
                model=model,
                messages=msgs,
                temperature=0.0,
                n=1,
                max_tokens=response_max_tokens,
                request_timeout=60,
            )
        except (OpenAITimeout, RequestsTimeout):
            if attempt < MAX_RETRIES - 1:  # i.e. if not the last attempt
                print(f"Timeout occurred, retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                raise


def grab_response_from_chatgpt(model, user_msg):
    msgs = [{"role": "user", "content": user_msg}]
    return create_response(model, msgs)


def error_correction_from_chatgpt(model, user_msg, predicted_sql, error_message):
    msgs = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": predicted_sql},
        {
            "role": "user",
            "content": f"That SQL produced this error message: \"{error_message}\". Write a new query. If you receieved a 'no such column' error, consider whether you pulled the column from the correct table. Don't apologize. Respond only with SQL.",
        },
    ]
    return create_response(model, msgs)


def check_execution_accuracy(predicted_query, ideal_query, _conn, model, user_msg):
    is_equal = False
    _predicted_set = set()

    cursor = _conn.cursor()
    try:
        cursor.execute(ideal_query)
    except sqlite3.OperationalError as _e:
        print(f"Error with ideal SQL: {_e}")
    ideal_res = cursor.fetchall()

    try:
        cursor.execute(predicted_query)
        _predicted_res = cursor.fetchall()
        _predicted_set = set(_predicted_res)

        if _predicted_set == set(ideal_res):
            is_equal = True

    except sqlite3.OperationalError as _e:
        corrected_sql_response = error_correction_from_chatgpt(
            model, user_msg, predicted_query, str(_e)
        )
        corrected_sql = corrected_sql_response["choices"][0]["message"]["content"]  # type: ignore

        try:
            cursor.execute(corrected_sql)
            corrected_res = cursor.fetchall()
            corrected_set = set(corrected_res)

            if corrected_set == set(ideal_res):
                is_equal = True
                _predicted_set = corrected_set
                predicted_query = corrected_sql

        except Exception as inner_e:
            print(str(inner_e))

    except Exception as _e:
        print(f"Unexpected error encountered: {_e}")

    return is_equal, _predicted_set, predicted_query


predicted_res = []
first_predicted_queries = []
final_predicted_queries = []
execution_accuracies = []
query_changes = []

df_pred = df.copy()

for index, row in df_pred.iterrows():
    print(f"processing row {index} of {len(df_pred)}")
    try:
        completion = grab_response_from_chatgpt(
            model=DEFAULT_MODEL, user_msg=row["user_prompt"]
        )
    except Exception as e:
        if "Read timed out" in str(e):
            print("The request to OpenAI API timed out. Retrying...")
            completion = grab_response_from_chatgpt(
                model=DEFAULT_MODEL, user_msg=row["user_prompt"]
            )
        else:
            print(f"An unexpected error occurred: {e}")
    first_predicted_query = completion["choices"][0]["message"]["content"]  # type: ignore
    db_name = row["db_id"]

    conn = sqlite3.connect(f"{DB_DIR}/{db_name}/{db_name}.sqlite", timeout=30)
    execution_accuracy, predicted_set, final_predicted_query = check_execution_accuracy(
        first_predicted_query,
        row["ideal_assistant_response"],
        conn,
        DEFAULT_MODEL,
        row["user_prompt"],
    )
    conn.close()

    execution_accuracies.append(1 if execution_accuracy else 0)
    first_predicted_queries.append(first_predicted_query)
    final_predicted_queries.append(final_predicted_query)
    predicted_res.append(predicted_set)


df_pred["first_predicted_query"] = first_predicted_queries
df_pred["final_predicted_query"] = final_predicted_queries
df_pred["predicted_res"] = predicted_res
df_pred["execution_accuracy"] = execution_accuracies

df_pred["formatted_string"] = (
    df_pred["final_predicted_query"] + "\t----- bird -----\t" + df_pred["db_id"]
)

result_json = df_pred["formatted_string"].to_dict()

with open(f"predict_{SET}.json", "w") as outfile:
    json.dump(result_json, outfile, indent=4)
