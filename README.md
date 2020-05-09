# Tempus Genomic Machine Learning Proposal
Architecture proposal for a self-service bioinformatics machine learning platform in AWS

## Assumptions
-Illumina machines are not capable of streaming. A private, dedicated bridge is created via AWS Direct Connect between Tempus' Chicago Lab and Tempus' AWS VPC. Illumina machines will write data to Tempus Data Lake's S3 Raw bucket and an Apache Airflow DAG will kickoff when the final file for a given sequencing job. Airflow will listen for the sequencing job's terminating file (RTAComplete.xml) using S3KeySensor.

![Basic Sequencing Pipeline](seq_pipeline.jpg)
