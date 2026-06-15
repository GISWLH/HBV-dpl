# Archive Notes

`oneearth_critical.tar.gz` is kept locally but is not committed directly because
GitHub LFS Free/Pro has a 2 GB per-file limit.

For GitHub publishing, the archive is split into parts under:

```text
version2/data/archive/oneearth_critical_parts/
```

Reconstruct on Windows PowerShell:

```powershell
$out = "version2\data\archive\oneearth_critical.tar.gz"
Remove-Item $out -ErrorAction SilentlyContinue
Get-ChildItem "version2\data\archive\oneearth_critical_parts\oneearth_critical.tar.gz.part*" |
  Sort-Object Name |
  ForEach-Object {
    $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
    $stream = [System.IO.File]::Open($out, "Append")
    try {
      $stream.Write($bytes, 0, $bytes.Length)
    } finally {
      $stream.Close()
    }
  }
```
