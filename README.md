# SUMMER-INTERN

🚀 Continual Learning Research Implementation (Summer Internship)

A comprehensive implementation and comparative analysis of state-of-the-art Continual Learning (CL) techniques to address catastrophic forgetting in dynamic environments.

📌 Overview

This project explores Continual Learning (CL) — a paradigm where models learn sequential tasks while retaining previously acquired knowledge. Unlike traditional training, CL systems must adapt to new data distributions without retraining from scratch.

The project implements and evaluates three cutting-edge research approaches:

LingoCL (CVPR 2024) – Language-guided supervision

PCL (AAAI 2021) – Per-class holistic learning

RCLP (WACV 2025) – Multi-label CL for medical imaging

The goal is to analyze how different strategies mitigate catastrophic forgetting and improve knowledge transfer across tasks.

🎯 Problem Statement

Traditional deep learning models fail in dynamic environments due to:

❌ Catastrophic forgetting

❌ Lack of semantic knowledge transfer

❌ Inability to handle evolving data distributions

This project investigates how modern CL techniques overcome these limitations.

🧠 Key Concepts

Continual Learning (CL)

Catastrophic Forgetting

Class / Domain / Task Incremental Learning

Representation Drift

Knowledge Transfer

As described in your report, CL enables models to learn continuously like humans by accumulating and retaining knowledge over time .

⚙️ Implemented Approaches
1️⃣ LingoCL – Language-Guided Continual Learning

Uses pretrained language models (PLMs) to generate semantic class representations

Replaces one-hot labels with semantic embeddings

Classifier is frozen, improving stability

📈 Key Insight:

Reduces representation drift

Improves cross-task knowledge transfer

2️⃣ Per-Class Continual Learning (PCL)

Learns one class at a time using one-class learning

Avoids interference between tasks

Uses holistic feature representation

📈 Key Insight:

Prevents catastrophic forgetting by avoiding parameter overwriting

3️⃣ Multi-Label CL (RCLP – Medical Domain)

Handles multi-label classification with domain shifts

Introduces Replay + Label Propagation

Designed for real-world medical scenarios (X-ray classification)

📈 Key Insight:

Combines new classes + new domains (NIC scenario)

Improves robustness in dynamic environments

🛠️ Tech Stack

Language: Python

Frameworks: PyTorch / TensorFlow

Libraries: NumPy, Pandas, Matplotlib

Datasets:

CIFAR-100

ImageNet-100

DomainNet

OfficeHome

📊 Experimental Results

Experiments were conducted across multiple datasets and settings.

🔹 Key Observations (from your results)

LingoCL consistently improves accuracy across baselines

Significant reduction in forgetting rate

Performance gains across:

CIFAR100

ImageNet100

DomainNet

📌 Example (from your report, page ~15):

GEM baseline: 58.4 → 64.8 with LingoCL

Forgetting reduced significantly

🔄 Workflow

Dataset Preparation

Task-wise Incremental Training

Model Implementation (LingoCL / PCL / RCLP)

Evaluation Metrics:

Accuracy

Forgetting Rate

Comparative Analysis

📈 Key Contributions

✅ Implemented 3 research-level CL methods from scratch

✅ Performed cross-method comparative analysis

✅ Validated improvements using real experimental results

✅ Demonstrated practical application of CL in multiple domains

🧠 Key Learnings

Importance of semantic information in learning systems

Trade-offs between stability vs plasticity

Real-world challenges in incremental AI systems

Deep understanding of modern CL research

🔮 Future Work

Integrate Transformer-based models (CLIP / LLMs)

Apply CL to real-time streaming data

Deploy system using MLOps pipelines (Docker + AWS)

Extend to reinforcement learning environments

▶️ How to Run
git clone https://github.com/Hayagreevan-K/SUMMER-INTERN.git
cd SUMMER-INTERN
pip install -r requirements.txt
python main.py
📧 Contact

Hayagreevan K

GitHub: https://github.com/Hayagreevan-K
