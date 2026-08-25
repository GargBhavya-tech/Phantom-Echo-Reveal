# PHANTOM-ECHO REVEAL — Firewall Fix
# Run this script as Administrator (right-click → Run with PowerShell as Admin)
# OR: open PowerShell as Admin and run: .\fix_firewall.ps1

$ruleName = "Phantom-Echo-8000"

# Remove old rule if it exists
netsh advfirewall firewall delete rule name=$ruleName 2>$null

# Add inbound rule for port 8000 on all profiles (domain, private, public)
$result = netsh advfirewall firewall add rule `
    name=$ruleName `
    dir=in `
    action=allow `
    protocol=TCP `
    localport=8000 `
    profile=any

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Firewall rule added successfully!" -ForegroundColor Green
    Write-Host "   Port 8000 is now open for inbound connections." -ForegroundColor Green
    Write-Host ""
    Write-Host "Now restart the server:" -ForegroundColor Cyan
    Write-Host "   python -m src.main --mode realtime" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Your phone URL: http://172.16.129.4:8000/m" -ForegroundColor Yellow
    Write-Host "(Make sure phone is on the same WiFi!)" -ForegroundColor Yellow
} else {
    Write-Host "❌ Failed. Make sure you are running as Administrator." -ForegroundColor Red
}

Read-Host "Press Enter to close"
