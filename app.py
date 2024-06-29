from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import pandas as pd
from langchain_core.documents import Document
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import string
import math
import re



df = pd.read_excel(r"your_path//data_533.xlsx")


app = Flask(__name__)
MODEL_PATH = r"your_path//gpt2"



gpt2_model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def dataconverter():
    product_data = pd.read_excel(r"your_path//data_533.xlsx")
    data = product_data

    product_list = []
    for index, row in data.iterrows():
        obj = {
            'product_title': row['product_title'], 
            'Review_title': row['Review title'],  
            'Rating': row['Rating'],
            'Review_text': row['Review text']    
        }
        product_list.append(obj)

    docs = []
    for entry in product_list:
        metadata = {
            "product_title": entry['product_title'],
            "Review_title": entry['Review_title'], 
            "Rating": entry['Rating']
        }
        doc = Document(page_content=entry['Review_text'], metadata=metadata)
        docs.append(doc)
    return docs



class SentenceTransformerWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, text):
        return self.model.encode([text])[0]

    def embed_documents(self, texts):
        return self.model.encode(texts)


def ingestdata(status):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding_instance = SentenceTransformerWrapper(model)


    vstore = AstraDBVectorStore(
        embedding=embedding_instance,
        collection_name="db_533",
        api_endpoint=ASTRA_DB_API_ENDPOINT,  
        token=ASTRA_DB_APPLICATION_TOKEN,    
        namespace=ASTRA_DB_KEYSPACE          
    )
    
    if status == "None":
        print("Ingesting data...")
        docs = dataconverter()
        inserted_ids = vstore.add_documents(docs)
    else:
        return vstore
    
    return vstore, inserted_ids



def getVectorStore():
    vstore = ingestdata("Done")
    return vstore


def similarity_search(query):
    vstore = getVectorStore()
    results = vstore.similarity_search(query) # query --> Number
    sentences_with_products = []
    
    for res in results:
        sentence = res.page_content
        metadata = res.metadata
        product_title = metadata.get('product_title', 'Unknown Product')
        review_title = metadata.get('Review_title', 'Unknown Review Title')
        rating = metadata.get('Rating', 'Unknown Rating')
        
        sentences_with_products.append({
            "sentence": sentence,
            "product_title": product_title,
            "review_title": review_title,
            "rating": rating
        })
    
    return sentences_with_products





def get_response(query):
    model_name = "gpt2"
    save_directory = r"your_path\\gpt2"
    model = GPT2LMHeadModel.from_pretrained(save_directory)
    tokenizer = GPT2Tokenizer.from_pretrained(save_directory)

    results = similarity_search(query)
    context = "\n".join(
        f"Product: {result['product_title']}\n"
        f"Title: {result['review_title']}\n"
        f"Rating: {result['rating']}\n"
        f"Review: {result['sentence']}\n"
        for result in results
    )
    
    max_length = tokenizer.model_max_length
    context = context[:max_length]
    
    question = "Please provide a summary of the general opinion about the Bluetooth headsets based on the reviews."

    PRODUCT_BOT_TEMPLATE = """
    You are an expert chatbot specialized in Bluetooth headsets. Your role is to provide accurate and helpful information regarding various Bluetooth headset products based on customer reviews and product descriptions. Your responses should be relevant to the product context, concise, and informative.

    Guidelines for your responses:
    1. Just make a small summary from the information available.
    2. If you dont know the answer just make up something realted to headsets.
    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    """

    input_text = PRODUCT_BOT_TEMPLATE.format(context=context, question=question)
    
    # print("Input text before encoding:", input_text)  # Debugging

    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # print("Encoded input_ids:", input_ids)  # Debugging
    # print(len(input_ids[0]))

    output = model.generate(
        input_ids,
        max_length=512,
        max_new_tokens=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    # print("Output from model.generate():", output)  # Debugging
    # print(len(output))

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    answer_start = generated_text.find("YOUR ANSWER:") + len("YOUR ANSWER:")
    answer = generated_text[answer_start:].strip()

    return answer

# Example usage:
# print(get_response("Tell me about Bluetooth headsets"))


def compute_tf(word_dict, words):
    tf_dict = {}
    n = len(words)
    for word, count in word_dict.items():
        tf_dict[word] = count / n
    return tf_dict

def compute_idf(doc_list):
    idf_dict = defaultdict(int)
    N = len(doc_list)
    for doc in doc_list:
        for word in set(doc):
            idf_dict[word] += 1
    
    for word, count in idf_dict.items():
        idf_dict[word] = math.log(N / float(count))
    return idf_dict

def compute_tfidf(tf, idf):
    tfidf = {}
    for word, tf_val in tf.items():
        tfidf[word] = tf_val * idf[word]
    return tfidf


def get_longest_review(product_name):
    # Escape special characters in the product name
    escaped_product_name = re.escape(product_name)
    # Filter reviews for the given product name
    product_reviews = df[df['product_title'].str.contains(escaped_product_name, case=False)]
    if product_reviews.empty:
        return None
    # Find the longest review text
    longest_review = product_reviews.loc[product_reviews['Review text'].str.len().idxmax()]
    return longest_review

# Example usage
# longest_review = get_longest_review(product_name)

# if longest_review is not None:
#     review_text = longest_review["Review text"]
#     print("Longest review text:", review_text)
# else:
#     print("No reviews found for the given product name.")



def summarize_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize the text into words
    words = [word_tokenize(sentence.lower()) for sentence in sentences]
    
    # Remove stop words and punctuation from the words
    stop_words = set(stopwords.words('english'))
    words = [[word for word in word_list if word not in stop_words and word not in string.punctuation] for word_list in words]
    
    # Compute TF for each sentence
    tf_scores = []
    for word_list in words:
        word_dict = defaultdict(int)
        for word in word_list:
            word_dict[word] += 1
        tf = compute_tf(word_dict, word_list)
        tf_scores.append(tf)
    
    # Compute IDF for the document
    idf = compute_idf(words)
    
    # Compute TF-IDF for each sentence
    tfidf_scores = []
    for tf in tf_scores:
        tfidf = compute_tfidf(tf, idf)
        tfidf_scores.append(tfidf)
    
    # Compute sentence scores based on TF-IDF
    sentence_scores = defaultdict(int)
    for i, tfidf in enumerate(tfidf_scores):
        sentence_scores[sentences[i]] = sum(tfidf.values())
    
    # Get the top 30% of sentences based on their scores
    import heapq
    summary_sentences = heapq.nlargest(int(len(sentences) * 0.3), sentence_scores, key=sentence_scores.get)
    
    # Join the summary sentences
    summary = ' '.join(summary_sentences)
    
    return summary



@app.route('/get_longest_review', methods=['POST'])
def get_longest_review_route():
    product_name = request.json.get('product_name')
    if not product_name:
        return jsonify({'error': 'Product name is required'}), 400
    
    longest_review = get_longest_review(product_name)
    if longest_review is None:
        return jsonify({'error': 'No reviews found for the given product name'}), 404
    
    review_text = longest_review['Review text']
    summary = summarize_text(review_text)
    
    return jsonify({'longest_review': review_text, 'summary': summary})




@app.route('/get_response', methods=['POST'])
def get_response_route():
    query = request.json.get('query')
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    response = get_response(query)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  