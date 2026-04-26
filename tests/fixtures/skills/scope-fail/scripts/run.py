"""Privileged script — should be flagged by SKILL-SCOPE-03 (read-only claim)."""
import os
import subprocess


def main():
    subprocess.run(["echo", "boot"])
    os.system("touch /tmp/footprint")


if __name__ == "__main__":
    main()
