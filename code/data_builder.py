from zipfile import ZipFile

with ZipFile("./data/maestro-v3.0.0.zip") as zf:
    for file in zf.namelist():
        if not file.endswith(".midi"):
            print(file)
