$max = 30
for ($i = 1; $i -le $max; $i++) {
    $status = git status | Out-String
    if ($status -like "*You are currently rebasing*") {
        $unmerged = git diff --name-only --diff-filter=U | Out-String
        if ($unmerged.Trim()) {
            foreach ($file in $unmerged.Split("`n")) {
                $file = $file.Trim()
                if ($file) {
                    Write-Host "Resolving conflict in $file"
                    git checkout --ours $file
                    git add $file
                }
            }
        }
        git -c core.editor="powershell -Command exit" rebase --continue
    } else {
        Write-Host "Rebase finished or not active!"
        break
    }
}
