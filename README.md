# AI Learning Roadmap

## 1. Preamble

This roadmap is designed for absolute beginners with **no prior experience** in computer science or mathematics.  It provides a step‑by‑step pathway through foundational topics into the most modern areas of artificial intelligence.  Each section explains why the knowledge is important and lists **free** resources.  Topics build on one another, so you can follow the tracks sequentially or dip into specific areas once you have the necessary prerequisites.  To keep the journey approachable, jargon is explained in plain language and resources are marked with a difficulty tag:

- **Easy** – gentle introductions or beginner‑friendly materials.
- **Medium** – assumes some basic familiarity with the prerequisites.
- **Hard** – in‑depth or mathematically rigorous content.

The diagrams shown throughout (see `/images` folder) depict each track as a subway‑style path so that prerequisites and dependencies are easy to visualise.  Feel free to print them or keep them nearby as you progress.

## 2. Legend

Each resource includes a difficulty tag and format indicator:

- **Video** – course, lecture series or recorded talk.
- **Text** – book, article or blog post.
- **Interactive** – coding exercises or labs you can try in your browser.

Tags are subjective and based on learner feedback.  If you find a resource too advanced, revisit it later after completing earlier topics.

## 3. Global Fundamentals

A strong foundation is the key to unlocking advanced AI topics.  In this section you will build intuition about how computers work and how to reason mathematically about data and algorithms.  Think of it as your **north–south line** on the subway map: everything else branches off of it.  The diagram below summarises the core foundational subjects; each stop corresponds to a topic in the tables that follow.  You can complete these in any order, but subjects like algebra and programming basics will make later topics easier.

![Fundamental CS Topics](images/new-track-fundamentals-cs.png)
![Fundamental Math Topics](images/new-track-fundamentals-math.png)

### 3.1 Computer Science Fundamentals

These subjects underpin all modern computing.  Take your time here; a strong grasp of these ideas will pay dividends later.  By learning how machines represent and manipulate information, you will be prepared to optimise code, debug issues and design systems that scale.  Each topic builds on the previous one: you start by writing simple programs, then understand how algorithms manage memory and time, and eventually explore how operating systems and networks coordinate complex tasks.

| Topic | Why it matters | Free resources (format) | Difficulty |
| --- | --- | --- | --- |
| **Programming Basics** | Learn to think like a programmer, write simple scripts and understand variables, control flow and data types. | *CS50: Introduction to Computer Science* (edX, Video) – a comprehensive entry‑level course; *Python for Everybody* (Coursera, Video) – gentle introduction using Python; *Automate the Boring Stuff with Python* (Text) – practical book that teaches Python through simple projects. | Easy |
| **Data Structures & Algorithms** | Efficient data structures and algorithms are crucial for solving problems and writing scalable code. | *Algorithms, Part I & II* (Coursera, Video) – covers sorting, searching, graphs and string algorithms; *MIT 6.006 Introduction to Algorithms* (OCW, Video) – lecture series with problem sets; *Data Structures and Algorithms in Python* (Interactive) – online notes and exercises. | Medium |
| **Computer Architecture & Operating Systems** | Understanding how computers actually work demystifies performance bottlenecks and hardware constraints.  Operating systems manage memory, processes and resources. | *Nand2Tetris* (official site, Video & Labs) – build a computer from logic gates up to a working OS; *Computer Systems: A Programmer’s Perspective* (Text) – recommended reading; *Operating Systems: Three Easy Pieces* (Text) – free online book covering OS fundamentals. | Medium |
| **Networking & Web** | Almost every modern application relies on networks.  Learn how data moves across the internet and how web protocols work. | *Stanford CS144: Introduction to Computer Networking* (Video) – explains protocols from sockets up to HTTP/HTTPS; *Mozilla Developer Network HTTP documentation* (Text) – practical guide to web requests and headers. | Medium |
| **Databases & SQL** | Structured data lives in relational databases.  Learning SQL enables you to query and manage data effectively. | *SQLBolt* (Interactive) – hands‑on lessons with immediate feedback; *Khan Academy: Intro to SQL* (Video) – beginner‑friendly series; *Mode Analytics SQL Tutorial* (Text & Interactive) – teaches analysis‑oriented SQL. | Easy |
| **Version Control** | Git is the industry standard for tracking code and collaborating with others. | *Git‑SCM Pro Git Book* (Text) – official free book; *Learn Git Branching* (Interactive) – visual practice environment; *GitHub Learning Lab* (Interactive) – hands‑on projects. | Easy |
| **Software Engineering Practices** | Writing production‑quality code involves design patterns, testing and documentation. | *Google’s Software Engineering Guide* (Text) – guidelines from Google engineers; *The Missing Semester of Your CS Education* (MIT, Video) – covers the shell, debugging, build systems and more; *Refactoring Guru* (Text) – free explanations of design patterns. | Medium |

### 3.2 Mathematics Fundamentals

Mathematics provides the language of modern machine learning and data science.  Build intuition before diving into heavy proofs.  Many newcomers fear the math behind AI, but you do not need to be a mathematician to succeed.  Instead, focus on the concepts: what does a derivative represent?  How do matrices transform space?  Why do probabilities add up to one?  With intuition, you can later tackle more abstract work.

| Topic | Why it matters | Free resources | Difficulty |
| --- | --- | --- | --- |
| **Algebra & Pre‑Calculus** | Refresh essential algebraic skills (equations, functions, exponents) which are used throughout calculus and linear algebra. | *Khan Academy: Algebra* (Video); *Paul’s Online Math Notes* (Text) – concise notes and exercises; *Pre‑calculus on MIT OpenCourseWare* (Video). | Easy |
| **Discrete Mathematics** | Basis of logic, combinatorics and graph theory – critical for algorithms and complexity analysis. | *MIT 6.042J Mathematics for Computer Science* (OCW, Video & Notes) – covers logic, proofs, sets and counting; *Neso Academy Discrete Mathematics* (YouTube) – lecture series; *Discrete Mathematics: An Open Introduction* (Text). | Medium |
| **Calculus** | Differential and integral calculus allow you to optimise functions and understand change – essential for gradient‑based learning algorithms. | *Khan Academy Calculus 1 & 2* (Video) – self‑paced; *MIT 18.01 Single‑Variable Calculus* (OCW, Video & Notes); *Paul’s Calculus Notes* (Text). | Medium |
| **Linear Algebra** | Vectors, matrices and linear transformations form the backbone of deep learning. | *MIT 18.06 Linear Algebra* (OCW, Video & Notes); *Essence of Linear Algebra* by 3Blue1Brown (YouTube) – intuitive visual explanations; *Linear Algebra Done Right* (Text) – rigorous textbook. | Medium |
| **Probability & Statistics** | Understanding uncertainty, distributions and statistical inference is crucial for data analysis and model evaluation. | *Khan Academy: Statistics and Probability* (Video); *Harvard Stat 110* (Video) – comprehensive course on probability; *OpenIntro Statistics* (Text) – free textbook; *Think Stats* by Allen Downey (Text) – statistics for programmers. | Medium |

## 4. Track‑by‑Track Roadmap

### 4.1 Data Engineering Track

![Data Engineering Track](images/new-track-data-engineering.png)

Data engineers build and maintain the pipelines that collect, process and store data.  They ensure that raw logs and transactional tables are transformed into clean, structured datasets that analysts and models can use.  This track begins with basic scripting and SQL, then gradually introduces the infrastructure used in industry: containers that package code, message brokers that handle streaming events, orchestrators that schedule jobs and warehouses that store terabytes of data.  By the end, you will have built a mini data platform end‑to‑end.

- **Python/Linux/SQL (Easy)** – refresh your Python skills (see Programming Basics), learn command‑line basics via *The Missing Semester of Your CS Education*, and practice SQL using *SQLBolt* or *Mode Analytics SQL Tutorial*.
- **Containers & Cloud (Medium)** – understand containers with *Docker Beginners Guide* (official docs) and explore basic cloud storage using *AWS S3 Getting Started* or *Google Cloud Storage Quickstarts*.
- **Ingestion & Streaming (Medium)** – learn message queues and streaming via *Apache Kafka 101* (Confluent free course) and practise building ingestion pipelines; explore cloud messaging services like Google Pub/Sub.
- **Workflow Orchestration (Medium)** – study orchestrators like *Apache Airflow* through the official tutorial and try the hands‑on *Prefect* or *Dagster* tutorials.
- **Warehouse & Analytics Engineering (Medium)** – understand columnar warehouses and star schemas with *Google BigQuery Fundamentals* (Coursera audit) or *Redshift Getting Started*; learn *dbt* with the free *dbt Fundamentals* course and practise modelling tables.
- **Batch & Monitoring (Medium)** – delve into distributed computing via *Apache Spark Free Intro* (Databricks community edition) or *Flink Getting Started* (official docs); learn about data quality using *Great Expectations* tutorials and monitoring concepts from *Monte Carlo blog posts*.
- **Capstone (Hard)** – put it all together with *Data Engineering Zoomcamp* (DataTalks.Club) – a project‑oriented nine‑week course covering Docker, Terraform, GCP, Airflow, Kafka, Spark and BigQuery.

### 4.2 Business Intelligence (BI) Track

![Business Intelligence Track](images/new-track-business-intelligence.png)

Business Intelligence focuses on turning data into actionable insights via dashboards and reports.  Whereas data science asks “why does this happen?”, BI often asks “what happened and how can we measure it?”.  Practitioners need strong communication skills, domain understanding and a sense of design to build dashboards that decision makers can trust.  You will learn how to design schemas that support reporting, select appropriate charts, tell stories with numbers and empower non‑technical colleagues to explore data on their own.

- **Spreadsheets & SQL (Easy)** – master spreadsheet basics using *Excel for Everyone* (Coursera audit) or Microsoft’s *Excel Training* site; practise formulas, pivot tables and charts.  Combine with SQL practice from the earlier fundamentals.
- **Data Modeling (Medium)** – learn to design star and snowflake schemas.  Read articles such as *Star Schema: The Complete Reference* overview (free blog summarising Kimball methodology) and practise modelling exercises using sample datasets.
- **BI Tools (Easy/Medium)** – choose a tool and follow its official free training:  *Power BI Guided Learning* (Microsoft), *Tableau Free Training Videos* (Tableau), or *Google Looker Studio* tutorials (Google).  Install the free versions and build sample dashboards.
- **Dashboards & Storytelling (Medium)** – watch *Storytelling with Data* webinars and read the blog to learn about visual perception, chart design and narrative flow.  Practise designing dashboards that highlight key metrics rather than clutter.
- **Self‑Service BI (Medium)** – explore open‑source tools like *Metabase* or *Apache Superset* through their docs and community tutorials; learn how business users can explore data on their own.
- **Advanced DAX & Reporting (Hard)** – if you choose Power BI, deepen your knowledge of Data Analysis Expressions (DAX) using *DAX Guide* (SQLBI) and the *Definitive Guide to DAX* series on YouTube; practise building complex measures and calculated tables.

### 4.3 Data Science Track

![Data Science Track](images/new-track-data-science.png)

Data scientists extract insights from data through exploration, analysis and modelling.  They act as detectives: cleaning messy datasets, asking the right questions, applying statistical tests and building models to make predictions.  This track emphasises statistics and practical data handling, guiding you through the entire lifecycle of a project—from loading CSV files to communicating findings.  You’ll practice on open datasets, learn how to visualise patterns and understand the assumptions behind common models.

- **Python (NumPy/Pandas) (Easy)** – read the *Python Data Science Handbook* by Jake VanderPlas (free online) and work through exercises in Jupyter.  Follow the official *NumPy* and *Pandas* tutorials.
- **Data Wrangling (Medium)** – practise cleaning messy datasets using *Pandas Cookbook* (GitHub) and Kaggle’s *Data Cleaning* micro‑course; learn to handle missing values, outliers and text formats.
- **Exploration & Visualization (Medium)** – learn exploratory data analysis using *Kaggle Exploratory Data Analysis* micro‑course, the *Matplotlib* and *Seaborn* documentation, and *Plotly Express* tutorials for interactive charts.
- **Statistics & Probability (Medium)** – revisit the probability & statistics resources above; supplement with *StatQuest with Josh Starmer* (YouTube) for intuitive explanations of statistical concepts.
- **Hypothesis Testing & A/B Testing (Medium)** – take the *A/B Testing* course from Google’s Data Analytics Professional Certificate (audit for free) or read blog posts explaining t‑tests and p‑values; practise designing experiments in a notebook.
- **Time Series (Medium)** – follow the *Forecasting: Principles and Practice* (free online book) and Kaggle’s *Time Series* micro‑course.  Learn ARIMA, exponential smoothing and Prophet models.
- **Feature Engineering & Causal Inference (Hard)** – take Kaggle’s *Feature Engineering* micro‑course; read *Causal Inference for the Brave and True* (free book) to understand confounding variables and potential outcomes; explore uplift modelling.
- **Portfolio Project (Hard)** – build a complete data science project by choosing an open dataset (Kaggle, UCI or governmental) and walking through question formulation, cleaning, exploration, modelling and communication.

### 4.4 Machine Learning Track

![Machine Learning Track](images/new-track-machine-learning.png)

Machine learning algorithms automatically learn patterns from data.  They power everything from spam filters to recommendation engines.  This track introduces supervised learning, where the goal is to predict labels, followed by unsupervised learning for discovering hidden structure, ensemble methods that combine many weak learners and deep learning for end‑to‑end function approximation.  You will also learn how to evaluate models properly and how to deploy them into production with robust pipelines and monitoring.

- **Supervised Learning (Easy/Medium)** – take *Machine Learning* by Andrew Ng (Coursera audit) for an overview of linear regression, logistic regression, neural nets and SVMs; practise implementing these algorithms using *scikit‑learn* tutorials; watch *StatQuest ML playlist* for intuitive explanations.
- **Evaluation & Metrics (Easy)** – study confusion matrices, precision/recall, ROC curves and cross‑validation through *scikit‑learn Evaluation Metrics* documentation; practise measuring overfitting and underfitting.
- **Unsupervised & Dimensionality Reduction (Medium)** – learn clustering (k‑means, hierarchical), mixture models and dimensionality reduction (PCA, t‑SNE) via the *Coursera Unsupervised Learning* course (DeepLearning.AI) or *Elements of Statistical Learning* (free PDF) chapters; visualise high‑dimensional data using t‑SNE/UMAP tutorials.
- **Advanced Models & Ensembles (Medium/Hard)** – explore ensemble methods like Random Forests, Gradient Boosting, XGBoost and LightGBM through their open documentation and tutorials; watch *StatQuest* videos on decision trees and random forests.
- **Deep Learning Basics (Medium)** – take *Neural Networks and Deep Learning* (Coursera, part of the Deep Learning Specialization) or *fast.ai Practical Deep Learning for Coders* (free course) to understand backpropagation, activation functions and CNN/RNN basics; read *Neural Networks and Deep Learning* by Michael Nielsen (free book).
- **Deployment & MLOps (Medium/Hard)** – learn how to package and deploy models using *TensorFlow Serving* or *TorchServe* tutorials; explore model tracking with *MLflow* (Docs) and orchestrate training pipelines with *Kubeflow* or *Prefect*; take *Made With ML MLOps Course* (free) or watch *Full Stack Deep Learning* lecture series.

### 4.5 Natural Language Processing (NLP) Track

![NLP Track](images/new-track-nlp.png)

Natural Language Processing (NLP) focuses on machines understanding and generating human language.  It powers chatbots, translators and search engines.  You will start with classic techniques like tokenisation and bag‑of‑words, progress to distributed representations such as word embeddings, and then explore neural sequence models.  Modern NLP relies heavily on transformers and pre‑trained large language models; this track introduces these architectures and shows how to fine‑tune them for your own tasks.

- **Text Processing & Tokenization (Easy)** – read *Natural Language Processing with Python* (NLTK Book, free) to learn tokenization, stop‑word removal and stemming; practise with NLTK and spaCy tutorials; watch *CS224N Lecture 2* on text preprocessing.
- **Word & Sentence Embeddings (Medium)** – understand distributed representations via *Word2Vec Tutorial* (TensorFlow blog) and *GloVe Paper and Demo*; learn sentence embeddings with *SentenceTransformers* (HuggingFace docs); read *Embeddings* chapter in the *Hugging Face Course*.
- **Sequence Models (Medium)** – take *Sequence Models* by Andrew Ng (Coursera) to learn RNNs, GRUs and LSTMs; watch *StatQuest* explanation of RNNs; implement simple language models using PyTorch or TensorFlow.
- **Transformers & Attention (Medium/Hard)** – read *The Illustrated Transformer* (Jay Alammar blog) for a gentle introduction to attention; follow the *Hugging Face Transformers Course* to build transformer models; watch lectures from *Stanford CS224N* on transformers.
- **Transfer Learning & Fine‑Tuning (Medium)** – follow the *Hugging Face Fine‑Tuning Tutorial* to adapt pre‑trained BERT/GPT models to specific tasks; practise on text classification and question answering tasks; explore parameter‑efficient methods like adapters and LoRA (Low‑Rank Adaptation).
- **Large Language Models & Prompting (Medium/Hard)** – take the *ChatGPT Prompt Engineering for Developers* course (DeepLearning.AI) to learn best practices; read the *OpenAI Cookbook* for examples using the OpenAI API; explore *LangChain* tutorials to create chatbots that call tools and search engines.

### 4.6 Computer Vision (CV) Track

![Computer Vision Track](images/new-track-computer-vision.png)

Computer vision enables machines to interpret images and video.  Cameras and sensors generate billions of pixels per second, and CV algorithms extract meaningful information from them.  You will begin by learning how images are represented as matrices and how simple filters can enhance or detect edges.  Then you will delve into convolutional neural networks for recognition and detection, eventually exploring cutting‑edge architectures like vision transformers.  Applications range from medical imaging to self‑driving cars.

- **Image Processing (Easy)** – learn basics of image manipulation using the *OpenCV‑Python Tutorials* (Docs) and *Pillow*; practise colour spaces, filtering and edge detection.
- **Convolutional Neural Networks (CNNs) (Medium)** – study *CS231n: Convolutional Neural Networks for Visual Recognition* (Stanford, Video & Notes) or the *Deep Learning Specialization: Convolutional Neural Networks* (Coursera audit); implement simple CNNs using PyTorch or TensorFlow.
- **Object Detection & Segmentation (Medium/Hard)** – follow the *Coursera: Computer Vision Specialization* (DeepLearning.AI) for detection, localisation and segmentation; explore frameworks like YOLOv5, Detectron2 or Ultralytics (docs and tutorials) to build real‑time detectors; learn about semantic and instance segmentation with U‑Net and Mask R‑CNN.
- **Transfer Learning & Pretrained Models (Medium)** – read the *PyTorch Transfer Learning Tutorial* and use pre‑trained networks (ResNet, EfficientNet) as feature extractors; practise fine‑tuning on your own image datasets.
- **Vision Transformers (Hard)** – read the *An Image Is Worth 16×16 Words* paper and Jay Alammar’s illustration; implement ViT using the *Hugging Face Transformers* library.
- **Multimodal Vision (Hard)** – explore models that combine vision and language such as CLIP (OpenAI blog and GitHub), DALL·E mini or Stable Diffusion; learn how image captioning and vision‑language pre‑training work.

### 4.7 Multimodal, RAG, Agents & vLLM Track

![Multimodal, RAG, Agents & vLLM Track](images/new-track-multimodal-rag-agents-vllm.png)

Modern AI increasingly integrates multiple data types and leverages large language models with external tools.  Multimodal models combine text, images, audio and video to produce rich understanding; retrieval‑augmented generation (RAG) supplements language models with external knowledge bases; autonomous agents use LLMs to decide which action to take next; and vLLM provides efficient inference for serving large models.  This track introduces these cutting‑edge domains and offers practical tutorials so you can build your own chatbots, multimodal search engines and agents.

- **Multimodal Basics & Fusion (Medium)** – read the *CMU Multimodal Machine Learning* course notes to understand challenges like representation, alignment and fusion; watch lecture videos to see examples of audio‑visual models; explore *Multimodal Machine Learning (MMML) tutorial* slides and notes.
- **Vision‑Language Models (Medium/Hard)** – study CLIP via OpenAI’s blog post and code examples; learn about image captioning with *Show and Tell* or *Show, Attend and Tell* papers; experiment with HuggingFace implementations of CLIP, BLIP, LLaVA or Flamingo for vision–language tasks.
- **Cross‑Modal Retrieval (Medium)** – build systems that retrieve images from text or vice versa.  Follow the *HuggingFace cross‑modal retrieval tutorial* and Weaviate’s *Multimodal Search Guide* (docs).  Understand how contrastive loss aligns embeddings across modalities.
- **RAG & Vector Databases (Medium)** – learn the concept of Retrieval Augmented Generation via LangChain’s *RAG documentation* and *Build a RAG App* tutorials.  Experiment with vector databases like Pinecone, Weaviate, FAISS or Chroma using their free tiers and tutorials.
- **Agents & Tool Use (Medium)** – explore *LangChain Agents* tutorials to see how language models can decide which tool to call; study open‑source projects like *BabyAGI* and *AutoGPT* on GitHub to understand task decomposition and memory; practise building simple agents that search the web, perform calculations and summarise results.
- **vLLM & High‑Performance Inference (Hard)** – read the *vLLM documentation* for installation and examples; learn how paged attention and token streaming improve throughput; integrate vLLM with LangChain or FastAPI to serve models cost‑effectively.  Compare with other inference engines like TensorRT‑LLM or HuggingFace’s Text Generation Inference.

## 5. References & Next Steps

This roadmap collates a wealth of **free** resources across the AI landscape.  Remember to pace yourself, revisit challenging topics and build projects that interest you.  As you progress, consider contributing to open‑source projects, joining online communities (such as Kaggle, HuggingFace forums or the DataTalks.Club Slack) and sharing your knowledge through blogs or talks.  Continuous learning and curiosity are key to mastering modern AI.

Enjoy the journey!

