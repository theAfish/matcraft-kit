---
name: mat-modelling-kit
description: A modular Python framework for building and analyzing atomic structures with CLI tools for materials modelling.
when_to_use: Use when you need to build, modify, inspect, or analyze atomic structures.
entry: mmkit -h
---

# Materials Modeling Toolkit (`mmkit`)

Use this skill to streamline and automate materials modeling workflows. The framework is divided into two primary command categories:

### Operations (Build & Modify)

Tasks dedicated to creating and manipulating atomic structures.

* **Command:** `mmkit operate -h`
* **Capabilities:** Building bulk crystals, generating surface slabs with precise termination control, and applying structural modifications.

### Observations (Inspect & Analyze)

Tasks dedicated to analyzing structural properties and verifying geometry.

* **Command:** `mmkit observe -h`
* **Capabilities:** Calculating structural metrics, inspecting coordination environments, and running validation checks.

---

> **Critical Workflow Rule:** Always run a sanity check after any structural modification using:
> `mmkit observe basic_check [<structure_file>]`
> *Never rely solely on intuition—always verify structural integrity programmatically.*