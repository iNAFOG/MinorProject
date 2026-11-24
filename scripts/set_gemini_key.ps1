# Run this yourself in PowerShell. The key is never printed or written to a project file.
$gameGeminiSecret = Read-Host 'Paste your Gemini API key into this hidden prompt' -AsSecureString
$gameGeminiPointer = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($gameGeminiSecret)
try {
    $gameGeminiValue = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($gameGeminiPointer)
    if ([string]::IsNullOrWhiteSpace($gameGeminiValue)) { throw 'The key was empty.' }
    [Environment]::SetEnvironmentVariable('GEMINI_API_KEY', $gameGeminiValue, 'User')
    $env:GEMINI_API_KEY = $gameGeminiValue
} finally {
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($gameGeminiPointer)
    Remove-Variable gameGeminiValue -ErrorAction SilentlyContinue
    $gameGeminiSecret.Dispose()
}
Write-Output 'GEMINI_API_KEY is set in your Windows user environment. Its value was not displayed.'
