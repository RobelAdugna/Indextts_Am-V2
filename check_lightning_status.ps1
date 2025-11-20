# Lightning.AI Training Status Checker
# Run from your local Windows machine

$session = "s_01kabdte02etx540f8hgx2dt1s@ssh.lightning.ai"
$project = "~/Indextts_Am-V2"

Write-Host "=== Training Process Status ===" -ForegroundColor Cyan
ssh $session "ps aux | grep train_gpt_v2 | grep -v grep | wc -l"

Write-Host "`n=== GPU Status ===" -ForegroundColor Cyan
ssh $session "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv"

Write-Host "`n=== Training Output Directory ===" -ForegroundColor Cyan
ssh $session "cd $project && ls -lh training_output/ 2>/dev/null | head -15"

Write-Host "`n=== Latest Checkpoint ===" -ForegroundColor Cyan
ssh $session "cd $project && ls -lth training_output/*.pth 2>/dev/null | head -3"

Write-Host "`n=== Recent Training Logs (last 20 lines) ===" -ForegroundColor Cyan
ssh $session "cd $project && tail -20 nohup.out 2>/dev/null || tail -20 training_output/training.log 2>/dev/null || echo 'No logs found'"

Write-Host "`n=== TensorBoard Events ===" -ForegroundColor Cyan
ssh $session "cd $project && find training_output -name 'events.out.tfevents.*' -type f 2>/dev/null | head -3"
