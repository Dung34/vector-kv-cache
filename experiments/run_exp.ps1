param (
    [Parameter(Mandatory=$true)]
    [string]$Script,

    [Parameter(Mandatory=$true)]
    [string]$Nonce,

    [Parameter(Mandatory=$true)]
    [string]$Tag,

    [Parameter(Mandatory=$true)]
    [string]$Py
)

$ErrorActionPreference = "Continue"

if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

$logPath = "logs/$Tag.log"
$numPath = "logs/$Tag.numbered.txt"

# Chạy script và ghi toàn bộ stdout + stderr ra file log
& $Py $Script --nonce $Nonce 2>&1 | Tee-Object -FilePath $logPath

$exitCode = $LASTEXITCODE
if ($null -eq $exitCode) {
    $exitCode = 0
}

# Ghi dòng [EXIT] vào cuối file log
$exitLine = "[EXIT] code=$exitCode"
Add-Content -Path $logPath -Value $exitLine
Write-Host $exitLine

# Đọc file log và tạo file numbered.txt: định dạng "  12| <nội dung>"
$rawLines = Get-Content -Path $logPath -Encoding UTF8
$numberedLines = @()
$lineNum = 1
foreach ($line in $rawLines) {
    $numberedLines += ("{0,4}| {1}" -f $lineNum, $line)
    $lineNum++
}
$numberedLines | Set-Content -Path $numPath -Encoding UTF8

# Tính sha256 của file log
$hash = (Get-FileHash -Path $logPath -Algorithm SHA256).Hash.ToLower()
$hashLine = "[HASH] $hash"
Write-Host $hashLine
