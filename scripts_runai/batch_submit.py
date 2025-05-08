import yaml
import subprocess
import os
import argparse

root_dir = "/nfs/home/tbarfoot/SACROS/scripts_runai/"


def get_parser():
    parser = argparse.ArgumentParser(
        description="Submit batch jobs defined in a YAML file."
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Relative path to the YAML file, e.g., 'acdc17/eval.yaml'.",
    )
    return parser


def submit_job(job):
    job_name = job["job_name"]
    bundle = job["bundle"]
    mode = job["mode"]
    seed = job["seed"]
    gpu_memory = job.get("gpu_memory", "32G")
    cpu_memory = job.get("cpu_memory", "64G")  # Default to 64G if not specified

    print(f"Submitting job: {job_name}")
    subprocess.run(
        [
            "bash",
            os.path.join(root_dir, "runai_submit.sh"),
            bundle,
            mode,
            str(seed),
            gpu_memory,
            cpu_memory,
            job_name,
        ]
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    yaml_file_path = os.path.join(root_dir, "batch_jobs", args.yaml_file)

    with open(yaml_file_path, "r") as file:
        jobs = yaml.safe_load(file)["jobs"]
        for job in jobs:
            submit_job(job)
