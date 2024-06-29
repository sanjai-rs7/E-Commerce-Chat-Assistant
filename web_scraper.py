from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
import time
import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver_path = r"path\\msedgedriver.exe"

options = webdriver.EdgeOptions()
service = EdgeService(driver_path)
driver = webdriver.Edge(service=service, options=options)

import requests
from bs4 import BeautifulSoup
import pandas as pd


def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    # Additional cleaning steps (e.g., remove specific unwanted phrases)
    # text = re.sub(r'Specific phrase to remove', '', text)
    return text


def get_reviews(url, num_reviews=10):
    # Set up Edge options
    options = webdriver.EdgeOptions()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    
    # Provide the path to the Edge WebDriver executable
    driver_path = r"C:\Users\Lenovo\Downloads\msedgedriver.exe"
    service = EdgeService(driver_path)
    
    # Initialize WebDriver
    driver = webdriver.Edge(service=service, options=options)
    
    driver.get(url)
    
    # Extract the product title
    product_title = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "a[data-hook='product-link']"))
    ).text.strip()
    
    reviews = []

    review_divs = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[data-hook='review']"))
    )

    for review in review_divs[:num_reviews]:
        review_dict = {}
        title_text = review.find_element(By.CSS_SELECTOR, "a[data-hook='review-title']").text.strip()
        review_dict["product_title"] = product_title  # Add the product title to each review
        review_dict["Review title"] = re.sub(r'^\d+.\d+ out of 5 stars\n', '', title_text).strip()  # Remove the rating part
        review_dict["Author"] = review.find_element(By.CSS_SELECTOR, "span.a-profile-name").text.strip()
        review_dict["Date"] = review.find_element(By.CSS_SELECTOR, "span[data-hook='review-date']").text.strip()
        rating_element = review.find_element(By.CSS_SELECTOR, "i[data-hook='review-star-rating'] span")
        review_dict["Rating"] = float(rating_element.get_attribute("textContent").split()[0])
        # review_dict["rating"] = float(rating_text.split()[0])  # Extract the numeric rating
        review_dict["Review text"] = review.find_element(By.CSS_SELECTOR, "span[data-hook='review-body']").text.strip()
        reviews.append(review_dict)

    driver.quit()
    return reviews




# List of URLs of the Amazon product pages
links = [
    "https://www.amazon.in/Zebronics-Zeb-Thunder-PRO-Headphone-Supporting/product-reviews/B097JPDQR8/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
    "https://www.amazon.in/Boult-Audio-Wireless-Playtime-Bluetooth/product-reviews/B0BQN3NW8C/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
    "https://www.amazon.in/Boult-Wireless-Earbuds-Playtime-Breathing/product-reviews/B0CVLB99YR/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
    "https://www.amazon.in/OnePlus-Bluetooth-Adaptive-Cancellation-Charging/product-reviews/B0CRH561RC/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
    "https://www.amazon.in/Bose-Earbuds-OpenAudio-Technology-Wireless/product-reviews/B0CPFV77W4/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
    "https://www.amazon.in/amazon-basics-Earphones-Powerful-Isolation/product-reviews/B0CKL64MBT/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
    "https://www.amazon.in/Refurbished-Bluetooth-Assistance-Cancellation-Water-Resistance/product-reviews/B0D4JZ46S4/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
    "https://www.amazon.in/INSTAPLAY-INSTABUDS-Headphones-Lightweight-Sweat-Resistant/product-reviews/B09Z2976HG/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
    "https://www.amazon.in/ZEBRONICS-Zeb-Bro-Earphones-Drivers-Compatible/product-reviews/B09R24HMNW/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
    "https://www.amazon.in/Launched-Noise-Headphones-Playtime-Latency/product-reviews/B0B1PXM75C/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
]

all_reviews = []

for link in links:
    try:
        reviews = get_reviews(link, num_reviews=10)
        all_reviews.extend(reviews)
    except Exception as e:
        print(f"Error fetching reviews from {link}: {e}")

# Create a DataFrame
df = pd.DataFrame(all_reviews)
print(df)


df.to_excel(r'your_path\\data\\review_83.xlsx', index=True)




from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

def convert_to_pdf():

    file_path = 'your_path\\data\\data\\data_533.pdf'
    pdf = SimpleDocTemplate(file_path, pagesize=letter)

    # Sample style sheet
    styles = getSampleStyleSheet()

    # List to hold the elements for the PDF
    elements = []

    # Iterate through the DataFrame and create paragraphs
    for index, row in df.iterrows():
        product_title = row['product_title']
        review_title = row['Review title']
        rating = row['Rating']
        review_text = row['Review text']
        
        # Create the paragraph text
        paragraph_text = f"""
        <b>Product Title:</b> {product_title}<br/>
        <b>Review Title:</b> {review_title}<br/>
        <b>Rating:</b> {rating}<br/>
        <b>Review Text:</b> {review_text}<br/><br/>
        """
        
        # Create a Paragraph object
        paragraph = Paragraph(paragraph_text, styles['Normal'])
        
        # Add the paragraph to the elements list
        elements.append(paragraph)
        elements.append(Spacer(1, 12))  # Add space between paragraphs

    # Build the PDF
    pdf.build(elements)

    print(f'DataFrame saved to {file_path}')