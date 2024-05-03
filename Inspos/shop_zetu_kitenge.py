import requests
import os
import re
from bs4 import BeautifulSoup

def valid_filename(s):
    """Sanitize a string to be used as a filename."""
    filename = s.split("/")[-1].split("?")[0]  # Extract filename from URL and remove query parameters
    # Remove variations like _widthx and _400x from the filename
    filename = re.sub(r'(_widthx|_400x)', '', filename)
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()

def download_image(url, folder="ShopZetu-Kitenge"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Extract a valid filename from the URL
    filename = valid_filename(url)
    if not filename.endswith((".jpg", ".jpeg", ".png", ".gif")):  # Check for common image extensions
        filename += '.jpg'  # Default extension if missing

    response = requests.get(url)
    filepath = os.path.join(folder, filename)
    
    with open(filepath, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filepath}")

def remove_widthx_files(folder):
    """Remove files with _widthx.jpg pattern from the given folder."""
    for filename in os.listdir(folder):
        if filename.endswith("_widthx.jpg"):
            filepath = os.path.join(folder, filename)
            os.remove(filepath)

def scrape_images(base_url, pages=20):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    
    for page in range(1, pages + 1):
        url = f"{base_url}?page={page}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            divs = soup.find_all('div', class_='grid grid--uniform')
            
            if not divs:
                print("No more images found or end of pages.")
                break
            
            for div in divs:
                images = div.find_all('img')
                for img in images:
                    src = img.get('src') or img.get('data-src')
                    if src:
                        if src.startswith("//"):
                            src = "https:" + src
                        elif src.startswith("/"):
                            src = "https://shopzetu.com" + src
                        download_image(src)
        else:
            print(f"Failed to retrieve page {page}. Status code: {response.status_code}")

if __name__ == "__main__":
    base_url = 'https://shopzetu.com/search?options%5Bprefix%5D=last&page=2&q=ankara&type=product'
    scrape_images(base_url, pages=5)  # Scrape the first 5 pages

    # After scraping, remove files with _widthx.jpg pattern
    remove_widthx_files("ShopZetu-Kitenge")
