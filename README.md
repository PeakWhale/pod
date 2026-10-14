<p><img src="pod.svg" width="64" height="64" alt="PeakWhale™ Pod logo"></p>

# PeakWhale™ Pod
### Group any spreadsheet of regions, customers, or accounts into clear segments with plain-English profiles.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-K--Means-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-HTTP%20API-teal)
![MCP](https://img.shields.io/badge/MCP-Server-informational)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)
[![CI](https://github.com/PeakWhale/pod/actions/workflows/ci.yml/badge.svg)](https://github.com/PeakWhale/pod/actions/workflows/ci.yml)

**PeakWhale™ Pod** is a domain-agnostic cohort and regional clustering engine. Give it any CSV table of territories, stores, customer accounts, or operational cohorts. Pod standardizes the numeric signals, picks the cleanest number of segments using silhouette scoring (or uses the `k` you choose), labels each segment in plain English, and flags outlier rows that sit far from their group center.

> [!NOTE]
> All bundled example datasets (`regional-markets.csv` and `customer-accounts.csv`) are 100% synthetic. The original 2020 regional clustering notebook is preserved in [`archive/`](archive/) for historical provenance.

## Quick start

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/PeakWhale/pod.git
cd pod
uv sync
```

Run a built-in example from the command line:

```bash
uv run pod cluster --example regional-markets
uv run pod cluster --example customer-accounts -k 3 --out-csv segmented-accounts.csv
```

Cluster your own CSV file:

```bash
uv run pod cluster path/to/your-data.csv --id-column account_id
```

## HTTP API and MCP server

Start the local FastAPI server (interactive OpenAPI docs at `http://127.0.0.1:8000/docs`):

```bash
uv run pod serve
```

Or run the Model Context Protocol (MCP) server for Claude Desktop, Cursor, or any MCP client:

```bash
uv run pod-mcp
```

## Tests

```bash
uv run pytest
```

## What it enables businesses to do

* **Territory and market planning:** Group sales territories, regions, or store locations by demand, growth, and coverage so leadership can allocate field teams with objective data.
* **Customer and account segmentation:** Turn raw billing and usage tables into clear customer tiers with plain-English traits instead of opaque cluster numbers.
* **Outlier and anomaly spotting:** Automatically flag accounts or regions that sit far from their assigned group center so analysts know where to look closer.

## Connect and customize

Want help adapting **PeakWhale™ Pod** to your own data, workflows, or infrastructure?

* **Book 30 minutes:** [Schedule a working session with Addy](https://adrkv.github.io/meet.html?audience=teams)
* **Portfolio and whitepapers:** [adrkv.github.io](https://adrkv.github.io)
* **LinkedIn:** [Addy Kaveti](https://www.linkedin.com/in/adrkv/) · [PeakWhale™](https://www.linkedin.com/company/peakwhale%E2%84%A2/)

## Part of the PeakWhale™ fleet

PeakWhale™ is an open-source initiative by [Addy Kaveti](https://adrkv.github.io) making practical AI and machine learning accessible to every team:

* **[PeakWhale™ Helm](https://github.com/PeakWhale/helm)**: local-first research assistant over your own SQL databases and documents, with source citations.
* **[PeakWhale™ Orca](https://github.com/PeakWhale/orca)**: self-hosted payment fraud check that scores transactions and explains every decision in plain English.
* **[PeakWhale™ Harbor](https://github.com/PeakWhale/harbor)**: turns any CSV spreadsheet into a prediction web app and JSON API with honest prediction ranges.
* **[PeakWhale™ Trident](https://github.com/PeakWhale/trident)**: fail-safe triage engine pairing AI with editable YAML rule packs so unclear cases always go to a person.
* **[PeakWhale™ Berth](https://github.com/PeakWhale/berth)**: EV depot fleet charging planner that schedules every vehicle to leave ready at the lowest power cost.
* **[PeakWhale™ Pod](https://github.com/PeakWhale/pod)**: domain-agnostic K-means cohort and regional clustering engine with plain-English segment profiles.
* **[PeakWhale™ Pilot](https://github.com/PeakWhale/pilot)**: outcome forecaster and early-warning engine that flags at-risk records and explains the top drivers.

## Legal, trademark, and liability notice

* **Open-source license:** Licensed under the **Apache License 2.0** (© 2020-2026 Aditya Rao (Addy) Kaveti). See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). If you distribute, modify, or build on this software, you must retain the [`NOTICE`](NOTICE) file as required by Section 4(d) of the Apache License 2.0.
* **Trademark reservation:** **PeakWhale™** and the PeakWhale project marks and logos are trademarks of Aditya Rao (Addy) Kaveti. Under Section 6 of the Apache License 2.0, this license grants no permission to use the PeakWhale™ name, marks, or logos to endorse or promote derivative products or services without prior written permission.
* **No warranty or professional advice:** This software is provided **"AS IS"**, without warranty or liability of any kind (see Sections 7 and 8 of the Apache License 2.0). Output from this tool does not constitute financial, medical, legal, compliance, or safety advice. You are solely responsible for independent validation, security controls, human oversight, and regulatory compliance before using this software in production.
* **Independent personal project and synthetic data:** Created and maintained by Aditya Rao (Addy) Kaveti in an independent personal capacity. All bundled demo datasets are synthetic or public benchmark examples and contain no proprietary or employer material.
