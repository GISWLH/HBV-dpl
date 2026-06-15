# Input Data Notes

`f1223_1023_daymet_20240826.npy` is kept locally but is not committed directly
because it is larger than GitHub LFS per-file limits.

For GitHub publishing, the file is split into parts under:

```text
version2/data/inputs/f1223_1023_daymet_20240826_parts/
```

Reconstruct on Windows PowerShell:

```powershell
$out = "version2\data\inputs\f1223_1023_daymet_20240826.npy"
Remove-Item $out -ErrorAction SilentlyContinue
Get-ChildItem "version2\data\inputs\f1223_1023_daymet_20240826_parts\f1223_1023_daymet_20240826.npy.part*" |
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

