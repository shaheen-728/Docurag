import json
import shutil
from pathlib import Path

path = Path("results/day3/review.json")
backup = path.with_name("review.before-labels.json")

labels = {
    "q1": [
        "342bf1e0-adc5-4f2e-bd7a-cd02e724730e",
        "4a473e98-65bc-4c28-a62e-5b913eae37dc",
        "75be9de4-8b8c-4a82-9373-191e81e2a002",
        "7e21f426-70c6-4178-9394-ff67d07cd453",
        "a2216103-e076-47ef-9783-4e11cbfd5bf0",
        "e7160089-32e9-46f8-8932-654ef487b9c8",
        "fb2d90c0-3878-4949-99c3-f7b471856845",
    ],
    "q2": [
        "cf746f2c-da0c-4945-ba38-38089fb1b04a",
        "d4953b52-4e76-456c-bf47-78c7f44c9c87",
    ],
    "q3": [
        "a46be6ad-8826-4a23-a08a-02d856c74093",
    ],
    "q4": [
        "398c8520-2522-4679-9f2d-e13c3ab6ad5e",
        "d8b30fdf-2836-43d1-b831-9217bdc2db01",
        "f5d7855e-f98c-486b-be07-21898f930da1",
    ],
    "q5": [
        "00e9c2e9-473c-4007-8e91-5b75b9588c0d",
        "6d79ac60-73d6-4b7b-b3ef-bdfd5161e17c",
        "80c24fc6-3633-4d6b-b630-58fa0d7a3ed4",
        "9fd17e53-319f-4107-8af0-1f36dca2d6e2",
        "ad471785-e2f2-4cd4-aab2-405f5bbf2878",
    ],
    "q6": [
        "006264fc-e80d-40b6-9a6c-c38b5106bfbf",
        "1af85c6b-78d0-4083-9238-34a961687877",
        "cf30252b-d41c-49de-a707-ff3f4de17fb7",
        "e260ba72-ea19-4b6b-b8ef-74308a7ac02b",
    ],
    "q7": [
        "baf4d503-c8c9-4da0-95d1-0918afdba8b1",
        "bdb69a78-2d2e-4281-a6fe-e933eae6df15",
    ],
    "q8": [
        "09db29f0-6c9e-4749-b783-369ef692faba",
        "9d1cc7b9-40cc-469c-98fc-b7e4e17fd6e8",
    ],
    "q9": [
        "2422da55-fd80-4c20-ad6d-15dc9987f926",
        "6de4be0c-9553-4c96-8a77-addad44ddcd2",
        "84ee7df9-8690-4395-9b69-b756ff17c9aa",
        "8919c354-6912-4eb2-804d-55b04bf9da92",
    ],
    "q10": [
        "2ed274c8-4ff0-47f0-b2bd-ecace896f7b9",
        "4ded39ea-6432-4f17-88a5-8cc8b7641848",
        "51e747ff-bc2a-45cb-b91c-58a2b591f140",
        "599430be-36cc-4563-bff1-9c3c22a4bff7",
        "bf460690-6833-450c-a729-6abb37afb2a0",
        "e596cb0f-cb2d-4551-a02f-bbe9bf9e3c21",
    ],
}

notes = {
    "q1": (
        "ConfigMap references, mounted files and environment-variable "
        "examples provide configuration mechanisms. Pod conditions "
        "does not answer the question."
    ),
    "q2": (
        "Explicitly states that immutable ConfigMap data cannot be "
        "mutated. General ConfigMap descriptions are insufficient."
    ),
    "q3": (
        "Provides kubectl rollout undo for Deployments, including "
        "--to-revision. Merely stating rollback is possible is insufficient."
    ),
    "q4": (
        "Identifies StatefulSets and directly describes stable identities "
        "and persistent storage."
    ),
    "q5": (
        "Explains DaemonSet per-node placement, eligible-node selection "
        "or a concrete logging-agent deployment."
    ),
    "q6": (
        "Explains backoffLimit, per-index retry limits or a concrete "
        "failure-policy configuration. General failure consequences "
        "alone are insufficient."
    ),
    "q7": (
        "Explains concurrencyPolicy: Forbid and skipping a scheduled "
        "run while the previous run remains active."
    ),
    "q8": (
        "Provides default-deny ingress examples. These examples select "
        "all Pods; a subset needs an appropriate podSelector. The "
        "deny-all example also restricts egress."
    ),
    "q9": (
        "Distinguishes storage supplied by a PV from a PVC request "
        "and explains their binding or consumption relationship."
    ),
    "q10": (
        "Explicitly explains that regular init containers complete "
        "before application containers start. Sidecars are different."
    ),
}

review = json.loads(path.read_text(encoding="utf-8"))

# Validate every proposed ID before modifying the review file.
for question_id, chunk_ids in labels.items():
    available = {
        candidate["chunk_id"] for candidate in review[question_id]["candidates"]
    }

    missing = set(chunk_ids) - available

    if missing:
        raise ValueError(f"{question_id}: IDs missing from candidates: {missing}")

if not backup.exists():
    shutil.copy2(path, backup)

for question_id, chunk_ids in labels.items():
    entry = review[question_id]

    entry["relevant_chunk_ids"] = chunk_ids
    entry["notes"] = notes[question_id]
    entry["label_source"] = "assistant-proposed; awaiting human verification"
    entry["reviewed"] = False

path.write_text(
    json.dumps(review, indent=2),
    encoding="utf-8",
)

print("Suggested labels saved. Human verification is still required.")
print("Backup:", backup)
