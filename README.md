# E-Commerce-Chat-Assistant

## Overview
The E-Commerce Help Assistant is a comprehensive application designed to assist users with Bluetooth headset products by providing detailed reviews, summaries, and responses to specific queries. The project integrates various technologies such as NLP, Machine Learning, and Web Scraping to offer a robust and user-friendly experience.

## Features:
1. Review Extraction: Extract reviews from Amazon product pages using Selenium.
2. Data Storage: Store extracted reviews in a structured format using Pandas and save them in an Excel file.
3. Text Summarization: Summarize the longest reviews using TF-IDF and NLTK.
4. Vector Embedding: Generate embeddings for document retrieval using NVIDIAEmbeddings.
5. Question Answering: Answer user queries based on the provided context using a pre-trained GPT-2 model.
6. Web Application: Flask API to serve review summaries and query responses.
7. Streamlit Interface: User-friendly interface for interacting with the assistant.

## Prerequisites:
Before running the application, ensure you have the following software installed:
- Python 3.9+
- Selenium WebDriver for Edge

## Getting Started

To get a local copy of the project up and running, follow these steps:

## Prerequisites

Ensure you have Git installed on your system. You can download it from [here](https://git-scm.com/).

## Installation
### Cloning the Repository

1. **Open your terminal (MacOS/Linux) or command prompt (Windows).**
2. **Navigate to the directory where you want to clone the repository.** Use the `cd` command to change directories. For example:
    ```bash
    cd path/to/your/directory
    ```
3. **Clone the repository** using the following command:
    ```bash
    git clone https://github.com/sanjai-rs7/E-Commerce-Chat-Assistant.git
    ```
4. **Navigate into the cloned directory**:
    ```bash
    cd E-Commerce-Chat-Assistant
    ```

You now have a local copy of the project and can begin working with it.

### Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
### Install the required packages:
```bash
pip install -r requirements.txt
```

### Configuration
Set up your environment variables by creating a .env file in the root directory with the following content:
```bash
NVIDIA_API_KEY=your_nvidia_api_key
ASTRA_DB_API_ENDPOINT=your_astra_db_endpoint
ASTRA_DB_APPLICATION_TOKEN=your_astra_db_application_token
ASTRA_DB_KEYSPACE=your_astra_db_keyspace
```

## Executing:
### Run the Flask API:
**If you want to run GPT2 and Detailed Summary of the review, do the following steps.**
```bash
python app.py
```
**API Endpoints:**
1. /get_longest_review: POST request to get the longest review and its summary.
2. /get_response: POST request to get a response to a specific query.

**Run the Streamlit application:**
```bash
streamlit run streamlit_app.py
```

**If you want to chat with LLama3 using NVIDIA Inferencing, run the command below.**
```bash
streamlit run chatbot.py
```
