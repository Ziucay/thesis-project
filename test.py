import pandas as pd
import os
import os.path
import yaml


def get_metrics(regex):
    workflows_path = []
    for root, dirs, files in os.walk('./temp'):
        for file in files:
            if regex.match(file):
                workflows_path.append(os.path.join(root, file))

    workflows_path.sort()
    if len(workflows_path) != 0:
        for workflow in workflows_path:
            with open(workflow, 'r') as stream:
                data_loaded = yaml.safe_load(stream)
                jobs = data_loaded['jobs']
                print(f'Number of jobs: {len(jobs)}')
                for job in jobs:
                    current_job_steps = jobs[job]['steps']
                    #print(current_job_steps)
                    print(f'Amount of steps in a job: {len(current_job_steps)}')
                    for step in current_job_steps:
                        #print(step)
                        if 'uses' in step:
                            print(step['uses'])
                        elif 'run' in step:
                            print(step['run'])
            pass
        pass
    else:
        pass


if __name__ == '__main__':
    #get_metrics(regex=re.compile('.*\.(yml|yaml)'))
    df = pd.read_csv('./dataset-with-workflow-1.csv')

    print(df.head())
