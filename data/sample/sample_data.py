"""
sample_data.py - Generates synthetic resume and job description data for testing
"""

SAMPLE_RESUMES = [
    {
        "id": "R001",
        "name": "Aarav Sharma",
        "text": """
        Aarav Sharma
        Email: aarav.sharma@email.com | Phone: +91-9876543210
        
        PROFESSIONAL SUMMARY
        Senior Data Scientist with 6 years of experience in machine learning, deep learning,
        and data analytics. Proficient in Python, TensorFlow, PyTorch, and cloud platforms.
        
        SKILLS
        Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, Scikit-learn,
        NLP, Computer Vision, SQL, AWS, Docker, Kubernetes, Spark, Hadoop,
        Data Visualization, Tableau, Power BI, Statistics, A/B Testing
        
        EXPERIENCE
        Senior Data Scientist | TechCorp India | 2020 - Present (4 years)
        - Built recommendation engine improving CTR by 35%
        - Developed NLP pipeline for sentiment analysis
        - Led team of 5 data scientists
        
        Data Scientist | StartupXYZ | 2018 - 2020 (2 years)
        - Implemented predictive models for churn analysis
        - Built automated ML pipelines using Airflow
        
        EDUCATION
        M.Tech Computer Science | IIT Bombay | 2018
        B.Tech Computer Science | NIT Trichy | 2016
        
        CERTIFICATIONS
        AWS Certified Machine Learning Specialty
        Google Professional Data Engineer
        """,
        "role": "Data Scientist"
    },
    {
        "id": "R002",
        "name": "Priya Patel",
        "text": """
        Priya Patel
        Email: priya.patel@email.com
        
        SUMMARY
        Full Stack Developer with 4 years experience building scalable web applications.
        Strong expertise in React, Node.js, and cloud infrastructure.
        
        TECHNICAL SKILLS
        JavaScript, TypeScript, React, Node.js, Express, MongoDB, PostgreSQL,
        Redis, Docker, AWS, CI/CD, Git, REST APIs, GraphQL, HTML, CSS,
        Webpack, Jest, Agile, Microservices
        
        WORK EXPERIENCE
        Full Stack Developer | WebSolutions Pvt Ltd | 2021 - Present (3 years)
        - Developed e-commerce platform serving 100k+ users
        - Implemented real-time chat using WebSockets
        - Reduced page load time by 40%
        
        Junior Developer | CodeBase Inc | 2020 - 2021 (1 year)
        - Built RESTful APIs for mobile applications
        - Maintained legacy PHP applications
        
        EDUCATION
        B.E. Information Technology | Pune University | 2020
        """,
        "role": "Full Stack Developer"
    },
    {
        "id": "R003",
        "name": "Rohit Verma",
        "text": """
        ROHIT VERMA - DATA ENGINEER
        rohit.verma@gmail.com | LinkedIn: linkedin.com/in/rohitverma
        
        PROFILE
        Data Engineer with 5 years of experience designing and maintaining data pipelines,
        ETL processes, and data warehousing solutions.
        
        SKILLS
        Python, Spark, Hadoop, Kafka, Airflow, SQL, PostgreSQL, MongoDB,
        AWS Glue, Redshift, S3, EMR, Databricks, dbt, Snowflake,
        Data Modeling, ETL, Data Warehousing, Docker, Linux
        
        EXPERIENCE
        Senior Data Engineer | DataFlow Technologies | 2021 - Present (3 years)
        - Designed real-time data pipeline processing 10M events/day using Kafka
        - Migrated on-premise data warehouse to Snowflake (50% cost reduction)
        - Built automated data quality checks
        
        Data Engineer | Analytics Corp | 2019 - 2021 (2 years)
        - Developed ETL pipelines using Apache Spark
        - Created data marts for business intelligence
        
        EDUCATION
        B.Tech Computer Science | Delhi University | 2019
        """,
        "role": "Data Engineer"
    },
    {
        "id": "R004",
        "name": "Sneha Gupta",
        "text": """
        Sneha Gupta
        sneha.gupta@outlook.com | +91-8765432109
        
        OBJECTIVE
        Aspiring Data Analyst with 2 years of experience in data analysis,
        visualization, and reporting. Seeking to leverage analytical skills.
        
        SKILLS
        Python, R, SQL, Excel, Tableau, Power BI, Pandas, NumPy,
        Matplotlib, Seaborn, Statistics, Data Cleaning, Data Visualization,
        Business Intelligence, Google Analytics
        
        WORK HISTORY
        Data Analyst | BizInsights | 2022 - Present (2 years)
        - Created dashboards tracking KPIs for 10+ business units
        - Performed A/B test analysis for marketing campaigns
        - Automated weekly reports saving 10 hours/week
        
        Data Analyst Intern | MarketResearch Co | 2021 - 2022 (1 year)
        - Conducted market research and competitor analysis
        - Built Excel models for financial forecasting
        
        EDUCATION
        MBA Business Analytics | Symbiosis | 2021
        B.Com | Mumbai University | 2019
        """,
        "role": "Data Analyst"
    },
    {
        "id": "R005",
        "name": "Vikram Singh",
        "text": """
        Vikram Singh | vikram.singh@email.com
        
        ABOUT ME
        Machine Learning Engineer with 7 years of experience deploying ML models
        at scale. Expert in MLOps, model serving, and production ML systems.
        
        CORE COMPETENCIES
        Python, Machine Learning, MLOps, Kubernetes, Docker, TensorFlow, PyTorch,
        Scikit-learn, MLflow, Kubeflow, FastAPI, Redis, Kafka, AWS SageMaker,
        Feature Engineering, Model Monitoring, A/B Testing, Statistical Modeling
        
        PROFESSIONAL EXPERIENCE
        ML Engineer Lead | AI Platform Co | 2020 - Present (4 years)
        - Built ML platform serving 50+ models in production
        - Reduced model deployment time from weeks to hours
        - Implemented automated retraining pipelines
        
        ML Engineer | DeepTech | 2017 - 2020 (3 years)
        - Developed computer vision models for defect detection
        - Optimized model inference (5x speed improvement)
        
        EDUCATION
        M.S. Machine Learning | Carnegie Mellon University | 2017
        B.Tech ECE | IIT Delhi | 2015
        """,
        "role": "ML Engineer"
    },
    {
        "id": "R006",
        "name": "Anjali Nair",
        "text": """
        Anjali Nair
        anjali.nair@email.com
        
        PROFESSIONAL SUMMARY
        Business Intelligence Developer with 3 years experience creating data-driven
        insights and reports for executive stakeholders.
        
        TECHNICAL SKILLS
        SQL, Tableau, Power BI, Python, Excel, SAP BusinessObjects,
        Data Warehousing, ETL, SSRS, SSAS, QlikView, Data Modeling
        
        EXPERIENCE
        BI Developer | Enterprise Solutions | 2021 - Present (3 years)
        - Developed 50+ Tableau dashboards for C-suite
        - Optimized SQL queries reducing report time by 60%
        - Implemented row-level security in Power BI
        
        Junior BI Developer | Reports Inc | 2020 - 2021 (1 year)
        - Maintained existing SSRS reports
        - Created ad-hoc analysis for sales team
        
        EDUCATION
        B.Tech Information Systems | VIT | 2020
        """,
        "role": "Data Analyst"
    },
    {
        "id": "R007",
        "name": "Karan Mehta",
        "text": """
        KARAN MEHTA
        karan.mehta@gmail.com | GitHub: github.com/karanml
        
        SUMMARY
        NLP Research Engineer with 5 years of experience in natural language processing,
        large language models, and conversational AI systems.
        
        SKILLS
        Python, NLP, Transformers, BERT, GPT, LLMs, HuggingFace, PyTorch,
        TensorFlow, Spacy, NLTK, Information Extraction, Text Classification,
        Named Entity Recognition, Question Answering, Machine Translation,
        Vector Databases, FAISS, Langchain, RAG
        
        EXPERIENCE
        NLP Engineer | AI Research Lab | 2021 - Present (3 years)
        - Fine-tuned BERT models for domain-specific classification
        - Built RAG system for enterprise knowledge base
        - Published 3 research papers in ACL/EMNLP
        
        ML Engineer | Chatbot Company | 2019 - 2021 (2 years)
        - Developed intent recognition system (95% accuracy)
        - Built multilingual NLP pipeline supporting 8 languages
        
        EDUCATION
        M.Tech AI | IISc Bangalore | 2019
        B.E. CSE | BITS Pilani | 2017
        """,
        "role": "Data Scientist"
    },
    {
        "id": "R008",
        "name": "Divya Krishnan",
        "text": """
        Divya Krishnan | divya.k@email.com
        
        PROFILE
        Cloud Data Architect with 8 years of experience designing enterprise-scale
        data platforms and cloud migration strategies.
        
        EXPERTISE
        AWS, Azure, GCP, Terraform, Kubernetes, Docker, Spark, Databricks,
        Snowflake, Redshift, BigQuery, Kafka, Airflow, Python, Scala,
        Data Architecture, Data Governance, Security, Cost Optimization
        
        CAREER HISTORY
        Principal Data Architect | CloudFirst | 2019 - Present (5 years)
        - Architected multi-cloud data platform for Fortune 500 client
        - Led data lake modernization (AWS + Databricks)
        - Defined data governance frameworks
        
        Senior Data Engineer | CloudTech | 2016 - 2019 (3 years)
        - Built real-time streaming platform on AWS Kinesis
        - Migrated 50TB legacy data warehouse to Redshift
        
        EDUCATION
        M.S. Computer Science | University of Texas | 2016
        B.Tech CSE | Anna University | 2014
        """,
        "role": "Data Engineer"
    }
]

SAMPLE_JOB_DESCRIPTIONS = {
    "data_scientist": """
    Job Title: Senior Data Scientist
    Company: TechInnovate India
    Location: Bangalore / Remote
    
    About the Role:
    We are looking for a Senior Data Scientist to join our AI team and work on
    cutting-edge machine learning solutions that impact millions of users.
    
    Required Skills:
    - 4+ years of experience in Data Science or Machine Learning
    - Strong proficiency in Python and ML frameworks (TensorFlow, PyTorch, Scikit-learn)
    - Experience with NLP and Natural Language Processing
    - Knowledge of deep learning architectures
    - Proficiency in SQL and data manipulation (Pandas, NumPy)
    - Experience with cloud platforms (AWS, GCP, or Azure)
    - Strong understanding of Statistics and Mathematics
    
    Preferred Skills:
    - Experience with MLOps and model deployment
    - Knowledge of distributed computing (Spark)
    - Publications or contributions to open source
    - Experience with computer vision
    
    Responsibilities:
    - Design and implement machine learning models
    - Collaborate with engineering teams for model deployment
    - Conduct exploratory data analysis and feature engineering
    - Present insights to business stakeholders
    - Mentor junior team members
    
    Education: M.Tech/M.S. in Computer Science, Statistics, or related field
    Experience: 4-7 years
    """,
    
    "data_engineer": """
    Job Title: Data Engineer
    Company: DataStream Technologies
    Location: Hyderabad / Remote
    
    Role Overview:
    Join our data platform team to build and maintain scalable data infrastructure
    that powers business intelligence and analytics.
    
    Must-Have Skills:
    - 3+ years in Data Engineering
    - Strong Python and SQL skills
    - Experience with Apache Spark and distributed computing
    - Knowledge of data pipeline tools (Airflow, Kafka)
    - Experience with cloud data warehouses (Snowflake, Redshift, BigQuery)
    - Familiarity with containerization (Docker, Kubernetes)
    - ETL/ELT pipeline development
    
    Nice to Have:
    - Experience with dbt
    - Knowledge of data modeling and dimensional modeling
    - Databricks experience
    - Scala programming
    
    Key Responsibilities:
    - Design and build robust data pipelines
    - Maintain data quality and reliability
    - Optimize query performance
    - Collaborate with data scientists and analysts
    - Implement data governance best practices
    
    Education: B.Tech/B.E. in Computer Science or related field
    Experience: 3-5 years
    """
}
