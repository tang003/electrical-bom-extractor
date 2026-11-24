"""Build the optional headless DWG reader. No AutoCAD installation is required.

Prefer `dotnet publish tools/cad_converter -c Release -o tools/cad_converter/publish`.
For this Windows workstation's runtime-only installation, --bootstrap downloads
pinned NuGet packages and invokes Microsoft's compiler using the installed .NET 8.
Nothing is downloaded while serving an upload.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "tools" / "cad_converter"


def package(name: str, version: str, cache: Path) -> Path:
    dest = cache / f"{name}.{version}.nupkg"
    if not dest.exists():
        url = f"https://api.nuget.org/v3-flatcontainer/{name}/{version}/{name}.{version}.nupkg"
        with urllib.request.urlopen(url, timeout=90) as response:
            content = response.read()
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if not any(p.endswith(".nuspec") for p in archive.namelist()):
                raise RuntimeError("Invalid NuGet package")
        dest.write_bytes(content)
    return dest


def build(bootstrap=False):
    dotnet = shutil.which("dotnet")
    if not dotnet:
        raise RuntimeError("Install the .NET 8 SDK or runtime on this machine first.")
    output = PROJECT / "publish"
    output.mkdir(parents=True, exist_ok=True)
    sdk = subprocess.run([dotnet, "--list-sdks"], capture_output=True, text=True, check=True)
    if sdk.stdout.strip() and not bootstrap:
        subprocess.run([dotnet, "publish", str(PROJECT), "-c", "Release", "-o", str(output), "--self-contained", "false"], check=True)
        return
    if not bootstrap:
        raise RuntimeError("No .NET SDK found. Use --bootstrap with the installed .NET 8 runtime, or install the SDK.")
    runtimes = subprocess.run([dotnet, "--list-runtimes"], capture_output=True, text=True, check=True).stdout
    candidates = re.findall(r"Microsoft\.NETCore\.App (8\.\d+\.\d+) \[(.+)\]", runtimes)
    if not candidates:
        raise RuntimeError("The bootstrap build requires an installed .NET 8 runtime.")
    version, folder = max(candidates, key=lambda p: tuple(map(int, p[0].split("."))))
    runtime = Path(folder) / version
    cache = ROOT / "tmp" / "cad_build_packages"
    cache.mkdir(parents=True, exist_ok=True)
    acad = package("acadsharp", "3.8.35", cache)
    compiler = package("microsoft.net.compilers.toolset", "4.12.0", cache)
    compiler_dir = cache / "compiler"
    with zipfile.ZipFile(compiler) as archive:
        prefix = "tasks/netcore/bincore/"
        for entry in archive.namelist():
            if entry.startswith(prefix) and not entry.endswith("/"):
                relative = Path(entry[len(prefix):])
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("Invalid archive entry")
                target = compiler_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(entry))
    with zipfile.ZipFile(acad) as archive:
        (output / "ACadSharp.dll").write_bytes(archive.read("lib/net8.0/ACadSharp.dll"))
    references = ["System.Runtime.dll", "System.Console.dll", "System.Private.CoreLib.dll",
                  "System.Collections.dll", "System.Linq.dll", "System.Text.Json.dll",
                  "System.Text.Encoding.CodePages.dll", "System.Memory.dll", "netstandard.dll"]
    command = [dotnet, str(compiler_dir / "csc.dll"), "-nologo", "-nullable:enable", "-target:exe",
               "-out:" + str(output / "ElectricalBom.CadConverter.dll"), "-r:" + str(output / "ACadSharp.dll")]
    command.extend("-r:" + str(runtime / ref) for ref in references)
    subprocess.run(command + [str(PROJECT / "Program.cs")], check=True)
    (output / "ElectricalBom.CadConverter.runtimeconfig.json").write_text(json.dumps({
        "runtimeOptions": {"tfm": "net8.0", "framework": {"name": "Microsoft.NETCore.App", "version": "8.0.0"}}
    }), encoding="utf-8")
    (output / "build_manifest.json").write_text(json.dumps({
        "provider": "ACadSharp", "version": "3.8.35", "license": "MIT", "target": "net8.0",
        "packages": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (acad, compiler)},
    }, indent=2), encoding="utf-8")
    (output / "THIRD_PARTY.txt").write_text(
        "ACadSharp 3.8.35 - MIT - Copyright (c) 2026 Albert Domenech\n"
        "https://github.com/DomCR/ACadSharp/blob/master/LICENSE\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files "
        "(the Software), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, "
        "publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, "
        "subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial "
        "portions of the Software. THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO "
        "THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT "
        "HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF "
        "OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n"
        "Microsoft compiler is a build dependency only and is not part of the runtime output.\n", encoding="utf-8")
    print(f"CAD converter ready: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true")
    build(parser.parse_args().bootstrap)
