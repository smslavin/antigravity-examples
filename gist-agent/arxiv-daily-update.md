# arXiv Daily Update - 2025-12-09

Could not generate overall trends.

## Top Papers

### [The Adoption and Usage of AI Agents: Early Evidence from Perplexity](http://arxiv.org/abs/2512.07828v1)
**Published:** 2025-12-08

**Abstract:** This paper presents the first large-scale field study of the adoption, usage intensity, and use cases of general-purpose AI agents operating in open-world web environments. Our analysis centers on Comet, an AI-powered browser developed by Perplexity, and its integrated agent, Comet Assistant. Drawing on hundreds of millions of anonymized user interactions, we address three fundamental questions: Who is using AI agents? How intensively are they using them? And what are they using them for? Our findings reveal substantial heterogeneity in adoption and usage across user segments. Earlier adopters, users in countries with higher GDP per capita and educational attainment, and individuals working in digital or knowledge-intensive sectors -- such as digital technology, academia, finance, marketing, and entrepreneurship -- are more likely to adopt or actively use the agent. To systematically characterize the substance of agent usage, we introduce a hierarchical agentic taxonomy that organizes use cases across three levels: topic, subtopic, and task. The two largest topics, Productivity & Workflow and Learning & Research, account for 57% of all agentic queries, while the two largest subtopics, Courses and Shopping for Goods, make up 22%. The top 10 out of 90 tasks represent 55% of queries. Personal use constitutes 55% of queries, while professional and educational contexts comprise 30% and 16%, respectively. In the short term, use cases exhibit strong stickiness, but over time users tend to shift toward more cognitively oriented topics. The diffusion of increasingly capable AI agents carries important implications for researchers, businesses, policymakers, and educators, inviting new lines of inquiry into this rapidly emerging class of AI capabilities.

**Analysis:**
Analysis failed.

---

### [Understanding Privacy Risks in Code Models Through Training Dynamics: A Causal Approach](http://arxiv.org/abs/2512.07814v1)
**Published:** 2025-12-08

**Abstract:** Large language models for code (LLM4Code) have greatly improved developer productivity but also raise privacy concerns due to their reliance on open-source repositories containing abundant personally identifiable information (PII). Prior work shows that commercial models can reproduce sensitive PII, yet existing studies largely treat PII as a single category and overlook the heterogeneous risks among different types. We investigate whether distinct PII types vary in their likelihood of being learned and leaked by LLM4Code, and whether this relationship is causal. Our methodology includes building a dataset with diverse PII types, fine-tuning representative models of different scales, computing training dynamics on real PII data, and formulating a structural causal model to estimate the causal effect of learnability on leakage. Results show that leakage risks differ substantially across PII types and correlate with their training dynamics: easy-to-learn instances such as IP addresses exhibit higher leakage, while harder types such as keys and passwords leak less frequently. Ambiguous types show mixed behaviors. This work provides the first causal evidence that leakage risks are type-dependent and offers guidance for developing type-aware and learnability-aware defenses for LLM4Code.

**Analysis:**
Analysis failed.

---

### [Collaborative Causal Sensemaking: Closing the Complementarity Gap in Human-AI Decision Support](http://arxiv.org/abs/2512.07801v1)
**Published:** 2025-12-08

**Abstract:** LLM-based agents are rapidly being plugged into expert decision-support, yet in messy, high-stakes settings they rarely make the team smarter: human-AI teams often underperform the best individual, experts oscillate between verification loops and over-reliance, and the promised complementarity does not materialise. We argue this is not just a matter of accuracy, but a fundamental gap in how we conceive AI assistance: expert decisions are made through collaborative cognitive processes where mental models, goals, and constraints are continually co-constructed, tested, and revised between human and AI. We propose Collaborative Causal Sensemaking (CCS) as a research agenda and organizing framework for decision-support agents: systems designed as partners in cognitive work, maintaining evolving models of how particular experts reason, helping articulate and revise goals, co-constructing and stress-testing causal hypotheses, and learning from the outcomes of joint decisions so that both human and agent improve over time. We sketch challenges around training ecologies that make collaborative thinking instrumentally valuable, representations and interaction protocols for co-authored models, and evaluation centred on trust and complementarity. These directions can reframe MAS research around agents that participate in collaborative sensemaking and act as AI teammates that think with their human partners.

**Analysis:**
Analysis failed.

---

### [LLM Use for Mental Health: Crowdsourcing Users' Sentiment-based Perspectives and Values from Social Discussions](http://arxiv.org/abs/2512.07797v1)
**Published:** 2025-12-08

**Abstract:** Large language models (LLMs) chatbots like ChatGPT are increasingly used for mental health support. They offer accessible, therapeutic support but also raise concerns about misinformation, over-reliance, and risks in high-stakes contexts of mental health. We crowdsource large-scale users' posts from six major social media platforms to examine how people discuss their interactions with LLM chatbots across different mental health conditions. Through an LLM-assisted pipeline grounded in Value-Sensitive Design (VSD), we mapped the relationships across user-reported sentiments, mental health conditions, perspectives, and values. Our results reveal that the use of LLM chatbots is condition-specific. Users with neurodivergent conditions (e.g., ADHD, ASD) report strong positive sentiments and instrumental or appraisal support, whereas higher-risk disorders (e.g., schizophrenia, bipolar disorder) show more negative sentiments. We further uncover how user perspectives co-occur with underlying values, such as identity, autonomy, and privacy. Finally, we discuss shifting from "one-size-fits-all" chatbot design toward condition-specific, value-sensitive LLM design.

**Analysis:**
Analysis failed.

---

### [Large Causal Models from Large Language Models](http://arxiv.org/abs/2512.07796v1)
**Published:** 2025-12-08

**Abstract:** We introduce a new paradigm for building large causal models (LCMs) that exploits the enormous potential latent in today's large language models (LLMs). We describe our ongoing experiments with an implemented system called DEMOCRITUS (Decentralized Extraction of Manifold Ontologies of Causal Relations Integrating Topos Universal Slices) aimed at building, organizing, and visualizing LCMs that span disparate domains extracted from carefully targeted textual queries to LLMs. DEMOCRITUS is methodologically distinct from traditional narrow domain and hypothesis centered causal inference that builds causal models from experiments that produce numerical data. A high-quality LLM is used to propose topics, generate causal questions, and extract plausible causal statements from a diverse range of domains. The technical challenge is then to take these isolated, fragmented, potentially ambiguous and possibly conflicting causal claims, and weave them into a coherent whole, converting them into relational causal triples and embedding them into a LCM. Addressing this technical challenge required inventing new categorical machine learning methods, which we can only briefly summarize in this paper, as it is focused more on the systems side of building DEMOCRITUS. We describe the implementation pipeline for DEMOCRITUS comprising of six modules, examine its computational cost profile to determine where the current bottlenecks in scaling the system to larger models. We describe the results of using DEMOCRITUS over a wide range of domains, spanning archaeology, biology, climate change, economics, medicine and technology. We discuss the limitations of the current DEMOCRITUS system, and outline directions for extending its capabilities.

**Analysis:**
Analysis failed.

---

### [ReasonBENCH: Benchmarking the (In)Stability of LLM Reasoning](http://arxiv.org/abs/2512.07795v1)
**Published:** 2025-12-08

**Abstract:** Large language models (LLMs) are increasingly deployed in settings where reasoning, such as multi-step problem solving and chain-of-thought, is essential. Yet, current evaluation practices overwhelmingly report single-run accuracy while ignoring the intrinsic uncertainty that naturally arises from stochastic decoding. This omission creates a blind spot because practitioners cannot reliably assess whether a method's reported performance is stable, reproducible, or cost-consistent. We introduce ReasonBENCH, the first benchmark designed to quantify the underlying instability in LLM reasoning. ReasonBENCH provides (i) a modular evaluation library that standardizes reasoning frameworks, models, and tasks, (ii) a multi-run protocol that reports statistically reliable metrics for both quality and cost, and (iii) a public leaderboard to encourage variance-aware reporting. Across tasks from different domains, we find that the vast majority of reasoning strategies and models exhibit high instability. Notably, even strategies with similar average performance can display confidence intervals up to four times wider, and the top-performing methods often incur higher and less stable costs. Such instability compromises reproducibility across runs and, consequently, the reliability of reported performance. To better understand these dynamics, we further analyze the impact of prompts, model families, and scale on the trade-off between solve rate and stability. Our results highlight reproducibility as a critical dimension for reliable LLM reasoning and provide a foundation for future reasoning methods and uncertainty quantification techniques. ReasonBENCH is publicly available at https://github.com/au-clan/ReasonBench .

**Analysis:**
Analysis failed.

---

### [Automating High Energy Physics Data Analysis with LLM-Powered Agents](http://arxiv.org/abs/2512.07785v1)
**Published:** 2025-12-08

**Abstract:** We present a proof-of-principle study demonstrating the use of large language model (LLM) agents to automate a representative high energy physics (HEP) analysis. Using the Higgs boson diphoton cross-section measurement as a case study with ATLAS Open Data, we design a hybrid system that combines an LLM-based supervisor-coder agent with the Snakemake workflow manager. In this architecture, the workflow manager enforces reproducibility and determinism, while the agent autonomously generates, executes, and iteratively corrects analysis code in response to user instructions. We define quantitative evaluation metrics including success rate, error distribution, costs per specific task, and average number of API calls, to assess agent performance across multi-stage workflows. To characterize variability across architectures, we benchmark a representative selection of state-of-the-art LLMs spanning the Gemini and GPT-5 series, the Claude family, and leading open-weight models. While the workflow manager ensures deterministic execution of all analysis steps, the final outputs still show stochastic variation. Although we set the temperature to zero, other sampling parameters (e.g., top-p, top-k) remained at their defaults, and some reasoning-oriented models internally adjust these settings. Consequently, the models do not produce fully deterministic results. This study establishes the first LLM-agent-driven automated data-analysis framework in HEP, enabling systematic benchmarking of model capabilities, stability, and limitations in real-world scientific computing environments. The baseline code used in this work is available at https://huggingface.co/HWresearch/LLM4HEP. This work was accepted as a poster at the Machine Learning and the Physical Sciences (ML4PS) workshop at NeurIPS 2025. The initial submission was made on August 30, 2025.

**Analysis:**
Analysis failed.

---

### [Mary, the Cheeseburger-Eating Vegetarian: Do LLMs Recognize Incoherence in Narratives?](http://arxiv.org/abs/2512.07777v1)
**Published:** 2025-12-08

**Abstract:** Leveraging a dataset of paired narratives, we investigate the extent to which large language models (LLMs) can reliably separate incoherent and coherent stories. A probing study finds that LLMs' internal representations can reliably identify incoherent narratives. However, LLMs generate responses to rating questions that fail to satisfactorily separate the coherent and incoherent narratives across several prompt variations, hinting at a gap in LLM's understanding of storytelling. The reasoning LLMs tested do not eliminate these deficits, indicating that thought strings may not be able to fully address the discrepancy between model internal state and behavior. Additionally, we find that LLMs appear to be more sensitive to incoherence resulting from an event that violates the setting (e.g., a rainy day in the desert) than to incoherence arising from a character violating an established trait (e.g., Mary, a vegetarian, later orders a cheeseburger), suggesting that LLMs may rely more on prototypical world knowledge than building meaning-based narrative coherence. The consistent asymmetry found in our results suggests that LLMs do not have a complete grasp on narrative coherence.

**Analysis:**
Analysis failed.

---

### [RL-MTJail: Reinforcement Learning for Automated Black-Box Multi-Turn Jailbreaking of Large Language Models](http://arxiv.org/abs/2512.07761v1)
**Published:** 2025-12-08

**Abstract:** Large language models are vulnerable to jailbreak attacks, threatening their safe deployment in real-world applications. This paper studies black-box multi-turn jailbreaks, aiming to train attacker LLMs to elicit harmful content from black-box models through a sequence of prompt-output interactions. Existing approaches typically rely on single turn optimization, which is insufficient for learning long-term attack strategies. To bridge this gap, we formulate the problem as a multi-turn reinforcement learning task, directly optimizing the harmfulness of the final-turn output as the outcome reward. To mitigate sparse supervision and promote long-term attack strategies, we propose two heuristic process rewards: (1) controlling the harmfulness of intermediate outputs to prevent triggering the black-box model's rejection mechanisms, and (2) maintaining the semantic relevance of intermediate outputs to avoid drifting into irrelevant content. Experimental results on multiple benchmarks show consistently improved attack success rates across multiple models, highlighting the effectiveness of our approach. The code is available at https://github.com/xxiqiao/RL-MTJail. Warning: This paper contains examples of harmful content.

**Analysis:**
Analysis failed.

---

### [SpatialDreamer: Incentivizing Spatial Reasoning via Active Mental Imagery](http://arxiv.org/abs/2512.07733v1)
**Published:** 2025-12-08

**Abstract:** Despite advancements in Multi-modal Large Language Models (MLLMs) for scene understanding, their performance on complex spatial reasoning tasks requiring mental simulation remains significantly limited. Current methods often rely on passive observation of spatial data, failing to internalize an active mental imagery process. To bridge this gap, we propose SpatialDreamer, a reinforcement learning framework that enables spatial reasoning through a closedloop process of active exploration, visual imagination via a world model, and evidence-grounded reasoning. To address the lack of fine-grained reward supervision in longhorizontal reasoning tasks, we propose Geometric Policy Optimization (GeoPO), which introduces tree-structured sampling and step-level reward estimation with geometric consistency constraints. Extensive experiments demonstrate that SpatialDreamer delivers highly competitive results across multiple challenging benchmarks, signifying a critical advancement in human-like active spatial mental simulation for MLLMs.

**Analysis:**
Analysis failed.

---

