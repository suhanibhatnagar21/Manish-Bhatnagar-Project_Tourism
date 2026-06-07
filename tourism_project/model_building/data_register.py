import os
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get('HF_TOKEN')
login(token=HF_TOKEN)

api     = HfApi()
repo_id = 'Suhani2128/tourism-dataset'

api.create_repo(
    repo_id   = repo_id,
    repo_type = 'dataset',
    exist_ok  = True,
    token     = HF_TOKEN
)
print(f'Repository created / verified: {repo_id}')

api.upload_file(
    path_or_fileobj = 'tourism_project/data/tourism.csv',
    path_in_repo    = 'tourism.csv',
    repo_id         = repo_id,
    repo_type       = 'dataset',
    token           = HF_TOKEN
)
print(f'Dataset registered at: https://huggingface.co/datasets/{repo_id}')
