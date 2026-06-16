$body = @{pair_code="555215"; client_id="test"} | ConvertTo-Json
$r = Invoke-RestMethod -Uri "http://127.0.0.1:8006/pair" -Method POST -ContentType "application/json" -Body $body
$r | ConvertTo-Json -Depth 3
