from huggingface_hub import list_repo_files

repo_id = "open-world-agents/vpt-owamcap"
files = list_repo_files(repo_id, repo_type="dataset")
tar_files = [f for f in files if f.endswith(".tar")]
print(f"Found {len(tar_files)} tar files.")
print("First 5 files:", tar_files[:5])
