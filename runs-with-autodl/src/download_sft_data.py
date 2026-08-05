import os, json, urllib.request, time
GITEE_BASE = "https://gitee.com/hf-datasets"
GITEE_TASKS = {
    "HuggingFaceTB/smol-smoltalk": [
        ("data/train-00000-of-00004.parquet","default","train",0),
        ("data/train-00001-of-00004.parquet","default","train",1),
        ("data/train-00002-of-00004.parquet","default","train",2),
        ("data/train-00003-of-00004.parquet","default","train",3),
        ("data/test-00000-of-00001.parquet","default","test",0),
    ],
    "cais/mmlu": [
        ("all/auxiliary_train-00000-of-00001.parquet","all","auxiliary_train",0),
        ("all/test-00000-of-00001.parquet","all","test",0),
    ],
    "openai/gsm8k": [
        ("main/train-00000-of-00001.parquet","main","train",0),
        ("main/test-00000-of-00001.parquet","main","test",0),
    ],
}
base = os.path.expanduser("/root/.cache/nanochat")
slugs = {"HuggingFaceTB/smol-smoltalk":"smol-smoltalk",
         "cais/mmlu":"cais-mmlu",
         "openai/gsm8k":"openai-gsm8k"}
for repo, files in GITEE_TASKS.items():
    slug = repo.replace("/", "--")
    gitee_repo = slugs[repo]
    for hf_path, subset, split, idx in files:
        out_dir = os.path.join(base, "task_data", slug, subset, split)
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{idx:05d}.parquet"
        url = f"{GITEE_BASE}/{gitee_repo}/raw/main/{hf_path}"
        print(f"  [{fname}] {url[:70]}...", end=" ", flush=True)
        t0 = time.time()
        with urllib.request.urlopen(url, timeout=600) as resp:
            data = resp.read()
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(data)
        print(f"OK {len(data)/1024/1024:.1f}MB ({time.time()-t0:.0f}s)")
        man_path = os.path.join(out_dir, "manifest.json")
        existing = json.load(open(man_path)) if os.path.exists(man_path) else []
        if fname not in existing:
            existing.append(fname)
            json.dump(sorted(existing), open(man_path, "w"))
print("SFT data download done")
