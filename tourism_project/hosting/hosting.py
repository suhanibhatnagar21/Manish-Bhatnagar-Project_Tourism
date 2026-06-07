import os
from huggingface_hub import HfApi, login

# ── Authenticate ──────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get('HF_TOKEN')
login(token=HF_TOKEN)

api     = HfApi()
repo_id = 'Suhani2128/wellness-tourism-prediction'

# ── Create the Hugging Face Space (Docker SDK) ────────────────────────────
api.create_repo(
    repo_id   = repo_id,
    repo_type = 'space',
    space_sdk = 'docker',
    exist_ok  = True,
    token     = HF_TOKEN
)
print(f'Space ready: https://huggingface.co/spaces/{repo_id}')

# ── Push all deployment files to the Hugging Face Space ──────────────────
deploy_dir = 'tourism_project/deployment'
for fname in ['app.py', 'requirements.txt', 'Dockerfile']:
    fpath = os.path.join(deploy_dir, fname)
    api.upload_file(
        path_or_fileobj = fpath,
        path_in_repo    = fname,
        repo_id         = repo_id,
        repo_type       = 'space',
        token           = HF_TOKEN
    )
    print(f'  Uploaded {fname}')

# ── Add HF_TOKEN as a space secret so the app can load the model ──────────
try:
    api.add_space_secret(
        repo_id = repo_id,
        key     = 'HF_TOKEN',
        value   = HF_TOKEN,
        token   = HF_TOKEN
    )
    print('HF_TOKEN secret added to the space.')
except Exception as e:
    print(f'Secret note (add manually if needed): {e}')

print(f'\nApp deployed: https://huggingface.co/spaces/{repo_id}')
