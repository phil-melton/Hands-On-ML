# Windows commit watchdog for opencv-expert GPU jobs in WSL.
# Every 5 s: if free commit (RAM + page file) drops below 3 GB, kill the training/export processes in WSL
# before Windows starts failing allocations (the 2026-10-05 23:06 crash). Logs a line per minute and the minimum seen.
param([double]$ThresholdGB = 3.0, [int]$MaxHours = 4)
$log = Join-Path $PSScriptRoot "commit_watchdog.log"
$min = 1e9; $start = Get-Date; $n = 0
"$(Get-Date -f s) watchdog start, threshold $ThresholdGB GB" | Add-Content $log
while (((Get-Date) - $start).TotalHours -lt $MaxHours) {
    $os = Get-CimInstance Win32_OperatingSystem
    $free = $os.FreeVirtualMemory / 1MB
    if ($free -lt $min) { $min = $free }
    if ($free -lt $ThresholdGB) {
        "$(Get-Date -f s) commit free {0:N2} GB < $ThresholdGB GB: killing WSL training processes" -f $free | Add-Content $log
        # Kill the chain scripts first so nothing starts the next step, then the Python jobs.
        wsl -d Ubuntu-24.04 -e pkill -f "train/hydrotest.sh|resume_.*\.sh"
        wsl -d Ubuntu-24.04 -e pkill -f "python -m train\."
        "$(Get-Date -f s) killed; min commit free {0:N2} GB" -f $min | Add-Content $log
        Write-Output "KILLED at commit free $([math]::Round($free,2)) GB"
        exit 2
    }
    if ($n % 12 -eq 0) {
        "$(Get-Date -f s) commit free {0:N1} GB (min {1:N1}), RAM free {2:N1} GB" -f $free, $min, ($os.FreePhysicalMemory / 1MB) | Add-Content $log
    }
    $n++
    Start-Sleep -Seconds 5
}
"$(Get-Date -f s) watchdog end, min commit free {0:N2} GB" -f $min | Add-Content $log
Write-Output ("watchdog ended normally; min commit free {0:N2} GB" -f $min)
