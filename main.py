import pandas as pd
import os
import re
import os.path
import json
import dataset_statistics


def requirements_metrics(*, df, regex, repo):
    requirements_path = []
    for root, dirs, files in os.walk('./temp'):
        for file in files:
            if regex.match(file):
                print(file)
                requirements_path.append(os.path.join(root, file))
    requirements_path.sort()
    if len(requirements_path) != 0:
        least_path = requirements_path[0]
        os.system(f'safety check -r {least_path} --output json > ./requirements_results.json')
        with open('./requirements_results.json', 'r') as results:
            data = json.load(results)

            scanned_packages = []
            for i in data['scanned_packages']:
                scanned_packages.append(i)

            affected_packages = []
            for i in data['affected_packages']:
                affected_packages.append(i)

            vulnerabilities = []
            for i in data['vulnerabilities']:
                vulnerabilities.append(i['vulnerability_id'])
            if len(scanned_packages) != 0:
                affected_scanned_ratio = len(affected_packages) / len(scanned_packages)
            else:
                affected_scanned_ratio = 0

            if len(affected_packages) != 0:
                vulnerabilities_affected_ratio = len(vulnerabilities) / len(affected_packages)
            else:
                vulnerabilities_affected_ratio = 0

            df.loc[df['name'] == repo, 'affected_scanned_ratio'] = affected_scanned_ratio

            df.loc[df['name'] == repo, 'vulnerabilities_affected_ratio'] = vulnerabilities_affected_ratio
    else:
        df.loc[df['name'] == repo, 'affected_scanned_ratio'] = 0

        df.loc[df['name'] == repo, 'vulnerabilities_affected_ratio'] = 0


def cyclomatic_complexity(*, df, repo):
    # Attempt to get cyclomatic complexity

    with open('./cc.json', 'r') as results:
        cc = []
        data = json.load(results)
        for file in data:
            for func in data[file]:
                if 'complexity' in func:
                    cc.append(func['complexity'])
        if len(cc) == 0:
            df.loc[df['name'] == repo, 'cc_mean'] = 0

            df.loc[df['name'] == repo, 'cc_variance'] = 0
        else:
            mean = sum(cc) / len(cc)

            df.loc[df['name'] == repo, 'cc_mean'] = mean

            df.loc[df['name'] == repo, 'cc_stdev'] = statistics.stdev(cc)


def maintainability_index(*, df, repo):
    # Attempt to get maintainability index

    with open('./mi.json', 'r') as results:
        mi = []
        data = json.load(results)
        for file in data:
            if 'mi' in data[file]:
                mi.append(data[file]['mi'])
        if len(mi) == 0:
            df.loc[df['name'] == repo, 'cc_mean'] = 0

            df.loc[df['name'] == repo, 'cc_variance'] = 0
        else:
            mean = sum(mi) / len(mi)

            df.loc[df['name'] == repo, 'mi_mean'] = mean

            df.loc[df['name'] == repo, 'mi_stdev'] = statistics.stdev(mi)


def raw_metrics(*, df, repo):
    # Attempt to get maintainability index
    with open('./raw.json', 'r') as results:
        loc = []
        lloc = []
        sloc = []
        comments = []
        blank = []
        data = json.load(results)
        for file in data:
            if 'loc' in data[file]:
                loc.append(data[file]['loc'])
                lloc.append(data[file]['lloc'])
                sloc.append(data[file]['sloc'])
                comments.append(data[file]['comments'])
                blank.append(data[file]['blank'])
        if len(loc) == 0:
            df.loc[df['name'] == repo, 'mean_loc'] = 0
            df.loc[df['name'] == repo, 'mean_lloc'] = 0
            df.loc[df['name'] == repo, 'mean_sloc'] = 0
            df.loc[df['name'] == repo, 'mean_comments'] = 0
            df.loc[df['name'] == repo, 'mean_blank'] = 0

            df.loc[df['name'] == repo, 'stdev_loc'] = 0
            df.loc[df['name'] == repo, 'stdev_lloc'] = 0
            df.loc[df['name'] == repo, 'stdev_sloc'] = 0
            df.loc[df['name'] == repo, 'stdev_comments'] = 0
            df.loc[df['name'] == repo, 'stdev_blank'] = 0
        else:

            df.loc[df['name'] == repo, 'mean_loc'] = statistics.mean(loc)
            df.loc[df['name'] == repo, 'mean_lloc'] = statistics.mean(lloc)
            df.loc[df['name'] == repo, 'mean_sloc'] = statistics.mean(sloc)
            df.loc[df['name'] == repo, 'mean_comments'] = statistics.mean(comments)
            df.loc[df['name'] == repo, 'mean_blank'] = statistics.mean(blank)

            df.loc[df['name'] == repo, 'stdev_loc'] = statistics.stdev(loc)
            df.loc[df['name'] == repo, 'stdev_lloc'] = statistics.stdev(lloc)
            df.loc[df['name'] == repo, 'stdev_sloc'] = statistics.stdev(sloc)
            df.loc[df['name'] == repo, 'stdev_comments'] = statistics.stdev(comments)
            df.loc[df['name'] == repo, 'stdev_blank'] = statistics.stdev(blank)


def pipeline():
    # token = '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'
    # g = Github(token)
    df = pd.read_csv('Second_datasets_with_workflow_3561.csv')

    repos = df['name'].tolist()

    limit = 4000
    pointer = 0

    regex = re.compile('.*requirements.*\.txt')

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
            requirements_metrics(df=df, regex=regex, repo=repo)

        # Get maintainability metrics
            os.system('radon cc --total-average -j -O \'./cc.json\' \'./temp\'')
            os.system('radon mi -j -O \'./mi.json\' \'./temp\'')
            os.system('radon raw --summary -j -O \'./raw.json\' \'./temp\'')

            cyclomatic_complexity(df=df, repo=repo)
            maintainability_index(df=df, repo=repo)
            raw_metrics(df=df, repo=repo)

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
