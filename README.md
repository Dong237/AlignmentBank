# 🏛️ AlignmentBank
**AlignmentBank** is an ongoing repository of paper notes and selective code reproductions, focused on foundational LLM techniques, reasoning research, and, above all, alignment. It is a “bank” of that aims to facilitate alignment-related knowledge flow with zero interest rate 😎.

The mission is to foster a continuously evolving, high-quality archive of both theoretical and empirical developments in LLM alignment. The broader vision is to scale this initiative into a community-oriented resource in the future. 

The repository is actively maintained with weekly updates.

## 📂 Areas Covered (To be expanded)

| Theme                | Subtopics |
|----------------------|-----------|
| Attention Mechanisms | MHA, MQA, SWA, GQA |
| LLM Architectures    | Dense, MoE |
| LLM Family           | GPT, LLaMA, Qwen, DeepSeek, etc. |
| LLM Alignment        | Algorithms, LLM Reasoning etc. |
| xxxxx                |        xxxxx                 |



## 📚 Catalog

1. [Attention Mechanism](#attention-mechanism)

2. [LLM Architecture Implementations](#llm-architecture-implementations)

3. [LLM Family](#llm-family)
    - [GPTs](#gpts)
    - [Llamas](#llamas)
    -  [Qwens](#qwens)
    -  [DeepSeek](#deepseek)
    -  [Mistral](#mistral)
    -  [AI2](#ai2)
    -  [Google](#google)

4. [LLM Alignment](#llm-alignment)

    - [Algorithms](#algorithms)

        - [Supervised Fine-Tuning](#supervised-fine-tuning)

        - [Reinforcement Learning](#reinforcement-learning)

    - [LLM Reasoning](#llm-reasoning)

        - [Verifiers](#verifiers---reasoning-with-verification)

        - [Proposer](#proposer-reasoning-by-changing-distribution)

        - [Planning](#planning)




## 🧠 Attention Mechanism

- Scaled dot-product self-attention / Multi-Head Attention (MHA): [Attnetion is all you need](https://vaulted-hardware-c41.notion.site/Attention-Is-All-You-Need-Vaswani-et-al-2017-1b92f406e62680bea2b9c43f2513e12e?pvs=73)

- Multi-Query Attention (MQA): [Fast Transformer Decoding: One Write-Head is All You Need](https://vaulted-hardware-c41.notion.site/Fast-Transformer-Decoding-One-Write-Head-is-All-You-Need-Noam-Shazeer-2019-1bf2f406e626803a8776f0fdf1366394)

- Sliding Window Attention (SWA): [Longformer: The Long-Document Transformer](https://vaulted-hardware-c41.notion.site/Longformer-The-Long-Document-Transformer-Beltagy-et-al-2020-1bc2f406e62680f19a63dffe4ef285ea)

- Grouped-Query Attention: [Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://vaulted-hardware-c41.notion.site/GQA-Training-Generalized-Multi-Query-Transformer-Models-from-Multi-Head-Checkpoints-Ainslie-et-al--1be2f406e62680109cc9f88c614e26f8)

## 🏗️ LLM Architecture Implementations

- [Reproducing dense models with GPT-2 as an example](./code/architectures/Dense/) (largely inspired by [Andrej Karpathy's video](https://www.youtube.com/watch?v=l8pRSuU81PU&t=11056s))

- Reproducing MoE models

## 🧬 LLM Family

### GPTs



- GPT: [Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)](https://vaulted-hardware-c41.notion.site/Improving-Language-Understanding-by-Generative-Pre-Training-Radford-et-al-2018-1ba2f406e626803a9f5ad52d0d74ced6)

- GPT-2: [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://vaulted-hardware-c41.notion.site/Language-Models-are-Unsupervised-Multitask-Learners-Radford-et-al-2019-1b72f406e62680aea5d3e1f73b592989)

- GPT-3: [Language Models are Few-Shot Learners (Brown et al., 2020)](https://vaulted-hardware-c41.notion.site/Language-Models-are-Few-Shot-Learners-Brown-et-al-2020-1bf2f406e626808b8dbbd5d7f8441ef7)

- InstructGPT: [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://vaulted-hardware-c41.notion.site/Training-language-models-to-follow-instructions-with-human-feedback-Ouyang-et-al-2022-1c32f406e62680deb5c3dc4a23732871?pvs=73)

### Llamas

- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)](https://vaulted-hardware-c41.notion.site/LLaMA-Open-and-Efficient-Foundation-Language-Models-Touvron-et-al-2023-1b22f406e62680148765caa6f083092e)

- [Llama 2: Open Foundation and Fine-Tuned Chat Models (GenAI, Meta, 2023.07)](https://vaulted-hardware-c41.notion.site/Llama-2-Open-Foundation-and-Fine-Tuned-Chat-Models-GenAI-Meta-2023-07-1c02f406e62680b9889de379c8856812)

- Llama 3

- Llama 4

### Qwens

- [Qwen Technical Report](https://vaulted-hardware-c41.notion.site/QWEN-TECHNICAL-REPORT-1d32f406e626803382f1c59c2af6bd6a)

- [Qwen2 Technical Report](https://vaulted-hardware-c41.notion.site/QWEN2-TECHNICAL-REPORT-1dd2f406e62680c2920ef15b4d69df42)

- [Qwen2.5 Technical Report](https://vaulted-hardware-c41.notion.site/Qwen2-5-Technical-Report-1e32f406e626807bacdeefe12af3bdda)

- Qwen3

### DeepSeek

- [DeepSeek LLM Scaling Open-Source Language Models with Longtermism](https://vaulted-hardware-c41.notion.site/DeepSeek-LLM-Scaling-Open-Source-Language-Models-with-Longtermism-1d92f406e626805eb969e97a6a6e4533)

- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://vaulted-hardware-c41.notion.site/DeepSeekMoE-Towards-Ultimate-Expert-Specialization-in-Mixture-of-Experts-Language-Models-1d92f406e62680b190b8cd57bb02f4e7)

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://vaulted-hardware-c41.notion.site/DeepSeek-V2-A-Strong-Economical-and-Efficient-Mixture-of-Experts-Language-Model-1ec2f406e626802c904affa50c50ed5d)

- DeepSeek-V3

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://vaulted-hardware-c41.notion.site/DeepSeek-R1-Incentivizing-Reasoning-Capability-in-LLMs-via-Reinforcement-Learning-1912f406e626807792ebd88dfcb55a8c)

### Mistral

- [Mistral 7B](https://vaulted-hardware-c41.notion.site/Mistral-7-3-B-Jiang-et-al-2023-09-1d12f406e6268061b000eb9d9be248af)

- [Mixtral of Experts](https://vaulted-hardware-c41.notion.site/Mixtral-of-Experts-Jiang-et-al-2023-12-1dd2f406e626805ba28ec590c0c93008)

### AI2

- OLMo

- 2OLMo 2 Furious

### Google

- Gemma

- Gemma 2

## 🎯 LLM Alignment

### Algorithms

#### Supervised Fine-Tuning

- [Injecting New Knowledge into Large Language Models via Supervised Fine-Tuning](https://vaulted-hardware-c41.notion.site/Injecting-New-Knowledge-into-Large-Language-Models-via-Supervised-Fine-Tuning-Mecklenburg-et-al-2-12a2f406e62680c598a6c09c3e6e2f67)

#### Reinforcement Learning

- PPO: [Proximal Policy Optimization Algorithms](https://vaulted-hardware-c41.notion.site/Proximal-Policy-Optimization-Algorithms-1932f406e6268043bec7d20d000f0c75)

- DPO: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://vaulted-hardware-c41.notion.site/Direct-Preference-Optimization-Your-Language-Model-is-Secretly-a-Reward-Model-1822f406e6268002a5dae77f801c146c?pvs=73)

- GRPO:

- DAPO:

- VAPO:

### LLM Reasoning

- Verifiers - Reasoning with verification:
    
    - [Training Verifiers to Solve Math Word Problems (Cobbe et al., 2021)](https://vaulted-hardware-c41.notion.site/Training-Verifiers-to-Solve-Math-Word-Problems-Cobbe-et-al-2021-1272f406e62680379431e2395cb52792)

    - [Solving math word problems with process and outcome-based feedback (Ursato et al., 2022)](https://vaulted-hardware-c41.notion.site/Solving-math-word-problems-with-process-and-outcome-based-feedback-Ursato-et-al-2022-1312f406e62680fd85e0d01cb3be5c22)

    - [Let’s Verify Step by Step (Lightman et al., 2023)](https://vaulted-hardware-c41.notion.site/Let-s-Verify-Step-by-Step-Lightman-et-al-2023-1292f406e62680589dafe09f813d6028)

    - [Improve Mathematical Reasoning in Language Models by Automated Process Supervision (Luo et al., 2024)](https://vaulted-hardware-c41.notion.site/Improve-Mathematical-Reasoning-in-Language-Models-by-Automated-Process-Supervision-Luo-et-al-2024-1352f406e62680a5a615dc556083fcbb)

    - [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://vaulted-hardware-c41.notion.site/Generative-Verifiers-Reward-Modeling-as-Next-Token-Prediction-1282f406e626803d8aeedf7175405080)

    - []()

- Proposer: reasoning by changing distribution

    - [REFT: Reasoning with REinforced Fine-Tuning (Luong el al., 2024)](https://vaulted-hardware-c41.notion.site/REFT-Reasoning-with-REinforced-Fine-Tuning-Luong-el-al-2024-8f1365269f12432aa912f1ae16a22162)

    - [Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning (Xie et al.)](https://vaulted-hardware-c41.notion.site/Logic-RL-Unleashing-LLM-Reasoning-with-Rule-Based-Reinforcement-Learning-Xie-et-al-1a52f406e62680a588f8e9221ac4bb07)

    - []()


- Planning

    - [Reasoning with Language Model is Planning with World Model (Hao et al., 2023)](https://vaulted-hardware-c41.notion.site/Reasoning-with-Language-Model-is-Planning-with-World-Model-Hao-et-al-2023-17a2f406e62680aebb5ecf1aed1e7ea0)

    - [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://vaulted-hardware-c41.notion.site/Tree-of-Thoughts-Deliberate-Problem-Solving-with-Large-Language-Models-1742f406e62680de8870fa04dc0ce4b4)

    - []()


