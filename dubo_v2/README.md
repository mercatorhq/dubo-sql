# Dubo-SQL v2

## Preparation

Before running inference on the dev set, we calculate embeddings for all the questions in the test set using `BIRD_train_embeddings.ipynb`

## Execution

Our main results for performance on the dev set come from `dubo-v2-full_dev_set.ipynb`. The baseline for ablation studies using a consistent, random sample of 500 question-answer pairs comes from `dubo_v2.ipynb`. The other notebooks contain other entries in the ablation study.

## API Key

Each notebook requires you to enter an OpenAI API key at the top of the notebook.