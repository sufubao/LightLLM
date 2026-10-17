"""Passive NVML sampling for the running request experiments; invoke via exp."""
import csv
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1])
with (root / "nvml.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "gpu", "uuid", "name", "driver", "used_mib", "total_mib", "utilization"])
    while not (root / "stop_nvml").exists():
        sample_time = time.time()
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=2,3",
                "--query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        for row in csv.reader(result.stdout.splitlines()):
            writer.writerow([sample_time] + [cell.strip() for cell in row])
        f.flush()
        time.sleep(0.5)
