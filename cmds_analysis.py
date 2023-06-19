import pandas as pd
import numpy as np
import re

# read dataset
'''df = pd.read_csv('./dataset-with-workflow-7.csv')

# get actions
df_actions = df['actions']

# convert to list of lists
df_actions_list = df_actions.values.tolist()
df_actions_list = [i.strip('][').split(', ') for i in df_actions_list]

# flatten it
flat_list = [item for sublist in df_actions_list for item in sublist]

actions_dict = {}
for i in flat_list:
    if i in actions_dict:
        actions_dict[i] += 1
    else:
        actions_dict[i] = 1

count_tens = 0
keys = list(actions_dict.keys())
for i in range(len(keys)):
    if actions_dict[keys[i]] <= 10:
        count_tens += 1
        actions_dict.pop(keys[i])
print(count_tens)

print(len(actions_dict.keys()))

with open("actions.json", "w") as outfile:
    json.dump(actions_dict, outfile)'''

'''df = pd.read_csv('./dataset-with-workflow-7.csv')

# get actions
df_runs = df['run_cmds']

# convert to list of lists
df_actions_list = df_runs.values.tolist()
df_actions_list = [i.strip('][').split(', ') for i in df_actions_list]

# flatten it
flat_list = [item for sublist in df_actions_list for item in sublist]
tokenized_list = [item.split() for item in flat_list]
flat_tokenized_list = [item for sublist in tokenized_list for item in sublist]

#print(flat_tokenized_list[10])

runs_dict = {}
for i in flat_tokenized_list:
    if i in runs_dict:
        runs_dict[i] += 1
    else:
        runs_dict[i] = 1

count_tens = 0
keys = list(runs_dict.keys())
for i in range(len(keys)):
    if runs_dict[keys[i]] <= 10:
        count_tens += 1
        runs_dict.pop(keys[i])
print(count_tens)

print(len(runs_dict.keys()))

with open("runs.json", "w") as outfile:
    json.dump(runs_dict, outfile)'''

df = pd.read_csv('./dataset-with-workflow-7.csv')

df['any_test'] = np.nan
df['testing_frameworks'] = np.nan
df['any_linter'] = np.nan
df['any_coverage'] = np.nan
df['any_doc_framework'] = np.nan
df['any_security_checkers'] = np.nan

repos = df['name'].tolist()

any_test_regex = re.compile('.*test.*')
testing_frameworks_regex = re.compile('.*(pytest|brownie|tox|unittest|doctest|nox|nbmake).*')
any_linter_regex = re.compile(
    '.*(ruff|flake8|isort|lint|black|eslint|pyupgrade|pylint|yapf|pycodestyle|mypy|anchore/scan-action).*')
any_coverage_regex = re.compile('.*(coverage|codecov|coveralls).*')
any_doc_regex = re.compile('.*(codespell|pandoc|freeze|graphviz|mkdocs|sphinx|markdown).*')
any_security_regex = re.compile('.*(safety|bandit|codeql).*')

for repo in repos:
    cur_action = df.loc[df['name'] == repo, 'actions'].tolist()[0]
    cur_run = df.loc[df['name'] == repo, 'run_cmds'].tolist()[0]
    #print(cur_action)
    #print(cur_run)

    if any_test_regex.match(cur_action) or any_test_regex.match(cur_run):
        df.loc[df['name'] == repo, 'any_test'] = 1
    else:
        df.loc[df['name'] == repo, 'any_test'] = 0

    if testing_frameworks_regex.match(cur_action) or testing_frameworks_regex.match(cur_run):
        df.loc[df['name'] == repo, 'testing_frameworks'] = 1
    else:
        df.loc[df['name'] == repo, 'testing_frameworks'] = 0

    if any_linter_regex.match(cur_action) or any_linter_regex.match(cur_run):
        df.loc[df['name'] == repo, 'any_linter'] = 1
    else:
        df.loc[df['name'] == repo, 'any_linter'] = 0

    if any_coverage_regex.match(cur_action) or any_coverage_regex.match(cur_run):
        df.loc[df['name'] == repo, 'any_coverage'] = 1
    else:
        df.loc[df['name'] == repo, 'any_coverage'] = 0

    if any_doc_regex.match(cur_action) or any_doc_regex.match(cur_run):
        df.loc[df['name'] == repo, 'any_doc_framework'] = 1
    else:
        df.loc[df['name'] == repo, 'any_doc_framework'] = 0

    if any_security_regex.match(cur_action) or any_security_regex.match(cur_run):
        df.loc[df['name'] == repo, 'any_security_checkers'] = 1
    else:
        df.loc[df['name'] == repo, 'any_security_checkers'] = 0

df.to_csv('./dataset-with-workflow-8.csv', index=False)
