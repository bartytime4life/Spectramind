import json, zipfile, pathlib
sub = pathlib.Path("outputs/submission.csv")
bundle = pathlib.Path("outputs/submission_bundle.zip")
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    if sub.exists(): z.write(sub, arcname=sub.name)
json.dump({"bundle": str(bundle)}, open("outputs/submission_meta.json","w"), indent=2)
print(f"📦 Bundle ready: {bundle}")
