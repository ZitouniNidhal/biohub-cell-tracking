import subprocess
import sys

def rewrite():
    print("Exporting git repo...")
    export_proc = subprocess.Popen(
        ["git", "fast-export", "--all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = export_proc.communicate()
    if export_proc.returncode != 0:
        print("Export error:", err.decode("utf-8", errors="ignore"))
        sys.exit(1)

    print("Replacing email...")
    old_email = b"nidhalzitouni111@gmail.com"
    new_email = b"88096539+ZitouniNidhal@users.noreply.github.com"
    
    modified_data = out.replace(old_email, new_email)
    
    print("Importing rewritten history...")
    import_proc = subprocess.Popen(
        ["git", "fast-import", "--force"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out_imp, err_imp = import_proc.communicate(input=modified_data)
    print("Import output:", err_imp.decode("utf-8", errors="ignore"))
    print("Resetting working tree to main...")
    subprocess.run(["git", "reset", "--hard", "main"], check=True)
    print("Successfully updated all author and committer emails!")

if __name__ == "__main__":
    rewrite()
