Write-Host "=== HEALTH ==="
curl http://127.0.0.1:8000/health

Write-Host "`n=== MODEL INFO ==="
curl http://127.0.0.1:8000/model-info

Write-Host "`n=== PREDICT ==="
$body = @{
    sk_id_curr = 123456
    features = @{
        EXT_SOURCE_1 = 0.52
        EXT_SOURCE_2 = 0.71
        EXT_SOURCE_3 = 0.41
    }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/predict" `
    -ContentType "application/json" `
    -Body $body