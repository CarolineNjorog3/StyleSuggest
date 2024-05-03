import os
import urllib.request
from bs4 import BeautifulSoup
import requests

def scrape_images(base_url, max_pages=5):
    try:
        print(f'Scraping images from URL: {base_url}')

        # Counter to keep track of downloaded images
        count = 0

        for page in range(1, max_pages + 1):
            # Construct the page URL with the query parameter
            page_url = f'{base_url}/page/{page}/?v=518f4a738816'

            # Send a GET request to the URL
            response = requests.get(page_url)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                print(f'Fetching images from page {page}')
                # Parse the HTML content of the webpage
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find the product list container
                product_list = soup.find('div', class_="nasa-archive-product-content nasa-after-clear margin-bottom-40")

                if product_list:
                    # Find all image tags within the product list container
                    images = product_list.find_all('img')

                    # Create a directory to store the images if it doesn't exist
                    os.makedirs('el_afrique', exist_ok=True)

                    # Download the images
                    for image in images:
                        # Check if the src attribute exists for the image
                        if 'src' in image.attrs:
                            # Get the source URL of the image
                            img_url = image['src']

                            # Check if the URL is a relative URL
                            if img_url.startswith('//'):
                                img_url = 'https:' + img_url

                            # Download the image
                            img_name = f'image_{count}.jpg'
                            img_path = os.path.join('el_afrique', img_name)
                            urllib.request.urlretrieve(img_url, img_path)
                            print(f'Downloaded: {img_url}')

                            # Increment the counter
                            count += 1
                else:
                    print('Product list container not found on the page.')
            else:
                print(f'Failed to fetch images from page {page}. Status Code: {response.status_code}')

    except Exception as e:
        print(f'Error occurred: {e}')

# Base URL of the webpage to scrape
base_url = 'https://elafrique.com/product-category/womens-fashion'

# Call the function to scrape images from all pages
scrape_images(base_url, max_pages=5)  # Change max_pages to the number of pages you want to scrape
