import pandas as pd
import os
import re
import os.path
import json

def workflows_metrics(*, df, regex, repo):
    requirements_path = []
    for root, dirs, files in os.walk('./temp'):
        for file in files:
            if regex.match(file):
                print(file)
                requirements_path.append(os.path.join(root, file))
    requirements_path.sort()
    if len(requirements_path) != 0:
        #TODO: Find out how to parse yml
        # Then make all needed strings to be in one array
        # Put it in one column for now
        # Append all strings in one big file
        least_path = requirements_path[0]
        os.system(f'safety check -r {least_path} --output json > ./requirements_results.json')
        with open('./requirements_results.json', 'r') as results:
            data = json.load(results)

            #df.loc[df['name'] == repo, 'affected_scanned_ratio'] = affected_scanned_ratio

            #df.loc[df['name'] == repo, 'vulnerabilities_affected_ratio'] = vulnerabilities_affected_ratio
    else:
        pass
        #TODO: nullify all metrics if error or not found
        #df.loc[df['name'] == repo, 'affected_scanned_ratio'] = 0
        #df.loc[df['name'] == repo, 'vulnerabilities_affected_ratio'] = 0



def pipeline():
    df = pd.read_csv('./cleaned-dataset-3.csv')

    repos = df['name'].tolist()

    limit = 4000
    pointer = 0

    regex = re.compile('.*.yml')

    for repo in repos:
        if pointer == limit:
            break
        print(f'repo: {repo}')
        if not pd.isnull(df.loc[df['name'] == repo, 'affected_scanned_ratio']).any():
            continue
        pointer += 1
        try:
            print('Start repo clone')
        # Repo.clone_from(f'https://github.com/{repo}.git', './temp')
            os.system(f'git clone https://github.com/{repo}.git ./temp')

            print('Start repo walk')
        # Attention! it actually changes supplied df
            workflows_metrics(df=df, regex=regex, repo=repo)

        # Get maintainability metrics
            print('remove repo')
            os.system('rm -rf ./temp')

        except:
            print(f"Exception occured at repo {repo}")
            df.to_csv(f'./datasets/dataset-error-{pointer}.csv')
            os.system('rm -rf ./temp')
            continue

    df.to_csv('./complete-cleaned-dataset.csv')


if __name__ == '__main__':
    pipeline()
