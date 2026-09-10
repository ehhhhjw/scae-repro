# dataset/

数据集放这里。复现包不携带数据（仓库快照几十 GB）。

    python3 ../setup/prepare_dataset.py --from-hf --skip-clone   # 只生成 CSV
    python3 ../setup/prepare_dataset.py --from-hf --clone        # 含仓库快照

生成后应有：

    dataset/instance_summary.csv     500 行，字段 instance_id/base_commit/problem_statement/patch_file
    dataset/projects/<instance_id>/  每实例一个 git 快照，HEAD == base_commit

也可以用环境变量指向别处，不必放这里：

    export SCAE_CSV=/path/to/instance_summary.csv
    export SCAE_PROJECTS=/path/to/projects

⚠️ 分发的快照常常叠着 gold patch（已修复状态）= 泄题。跑
`python3 ../setup/prepare_dataset.py --verify-only` 体检。见 ../docs/PITFALLS.md
