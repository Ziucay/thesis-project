import pandas as pd
import numpy as np
import os
import re
import os.path
import yaml


def get_metrics(df, regex, repo):
    workflows_path = []
    for root, dirs, files in os.walk('./temp'):
        for file in files:
            if regex.match(file):
                workflows_path.append(os.path.join(root, file))

    workflows_path.sort()
    if len(workflows_path) != 0:
        jobs_count = 0
        steps_count = 0
        uses_count = 0
        run_count = 0
        run_commands = []
        uses_commands = []
        for workflow in workflows_path:
            with open(workflow, 'r') as stream:
                data_loaded = yaml.safe_load(stream)
                jobs = data_loaded['jobs']
                jobs_count += len(jobs)
                #print(f'Number of jobs: {len(jobs)}')
                for job in jobs:
                    current_job_steps = jobs[job]['steps']
                    # print(current_job_steps)
                    steps_count += len(current_job_steps)
                    #print(f'Amount of steps in a job: {len(current_job_steps)}')
                    for step in current_job_steps:
                        # print(step)
                        if 'uses' in step:
                            uses_count += 1
                            run_commands.append(step['uses'])
                            #print(step['uses'])
                        elif 'run' in step:
                            run_count += 1
                            uses_commands.append(step['run'])
                            #print(step['run'])
        df.loc[df['name'] == repo, 'jobs_number'] = jobs_count
        df.loc[df['name'] == repo, 'steps_number'] = steps_count
        df.loc[df['name'] == repo, 'actions_number'] = uses_count
        df.loc[df['name'] == repo, 'jobs_avg_steps_number'] = steps_count / jobs_count
        df.loc[df['name'] == repo, 'jobs_avg_actions_number'] = uses_count / jobs_count
        df.loc[df['name'] == repo, 'jobs_avg_run_number'] = run_count / jobs_count
        df.loc[df['name'] == repo, 'workflow_avg_jobs_number'] = jobs_count / len(workflows_path)
        print(f'Len of cmds is {len(run_commands)}')
        print(f'Len of cmds is {len(uses_commands)}')
        df.loc[df['name'] == repo, 'run_commands'] = str(run_commands)
        df.loc[df['name'] == repo, 'uses_actions'] = str(uses_commands)


def get_files_by(repo_name):
    #print(f'https://github.com/{repo_name}/tree/master/.github/workflows/')
    main_result = os.system(f'ghget -o temp https://github.com/{repo_name}/tree/main/.github/workflows/')

    if main_result == 0:
        print('Ok main')
        return

    master_result = os.system(f'ghget -o temp https://github.com/{repo_name}/tree/master/.github/workflows/')

    if master_result == 0:
        print('Ok master')
        return

    dev_result = os.system(f'ghget -o temp https://github.com/{repo_name}/tree/dev/.github/workflows/')


def pipeline():
    df = pd.read_csv('./dataset-with-workflow-2.csv')
    #df['jobs_number'] = np.nan
    #df['steps_number'] = np.nan
    #df['actions_number'] = np.nan
    #df['jobs_avg_steps_number'] = np.nan
    #df['jobs_avg_actions_number'] = np.nan
    #df['jobs_avg_run_number'] = np.nan
    #df['workflow_avg_jobs_number'] = np.nan
    df['run_commands'] = np.nan
    df['uses_actions'] = np.nan
    df['run_commands'] = df['run_commands'].astype(object)
    df['uses_actions'] = df['uses_actions'].astype(object)

    repos = df['name'].tolist()

    limit = 4000
    pointer = 0

    regex = re.compile('.*\.(yml|yaml)')

    for repo in repos:
        if pointer == limit:
            break
        print(f'repo: {repo}')
        #if not pd.isnull(df.loc[df['name'] == repo, 'jobs_number']).any():
        #    continue
        pointer += 1
        try:
            print('Start repo clone')

            get_files_by(repo)

            print('Start collecting metrics')

            # Attention! it actually changes supplied df
            get_metrics(df=df, regex=regex, repo=repo)

            print('remove repo')
            os.system('rm -rf ./temp')

        except Exception as e:
            print(str(e))
            print(f"Exception occured at repo {repo}")
            df.to_csv(f'./datasets/dataset-workflow-error-{pointer}.csv')
            os.system('rm -rf ./temp')
            continue

    df.to_csv('./dataset-with-workflow-3.csv')


if __name__ == '__main__':
    pipeline()
