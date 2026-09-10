# dataset/

Put the dataset here. The reproduction package carries no data (repository snapshots are tens of GB).

    python3 ../setup/prepare_dataset.py --from-hf --skip-clone   # CSV only
    python3 ../setup/prepare_dataset.py --from-hf --clone        # including repository snapshots

After generation you should have:

    dataset/instance_summary.csv     500 rows, fields instance_id/base_commit/problem_statement/patch_file
    dataset/projects/<instance_id>/  one git snapshot per instance, HEAD == base_commit

You can also point elsewhere via environment variables instead of putting files here:

    export SCAE_CSV=/path/to/instance_summary.csv
    export SCAE_PROJECTS=/path/to/projects

⚠️ Distributed snapshots often have the gold patch already applied (fixed state) = answer leakage. Run
`python3 ../setup/prepare_dataset.py --verify-only` to check. See ../docs/PITFALLS.md
