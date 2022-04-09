
#### SOURCE --- 
# https://github.com/kathi-roll/Python/blob/master/Mask%20Recognition/scrapeImages.py
# https://github.com/bellamkondaprakash/Lab_white_coat_detection/blob/441048c9aa650d592c9f822223ae2ee71075de9f/google_image_scrap.py
# https://github.com/sr1jan/videoAutoProduction/blob/e6988c1d2debd511b63968a55e29c4d2373c5ab6/pythonScripts/extractImages.py
# https://github.com/kyleleeners/PythonImageGui/blob/f0e093e62a7098a28ff0265128d12cc95cc65d90/colour_scraper.py
# https://github.com/sivamsinghsh/Machine-Learning-Textextraction/blob/b1bade8cf8db639dbd1ef3f5c2fced859a95157a/google_search.py
# https://github.com/miguelgfierro/pybase/blob/de8e4f11ed5c655e748178e65195c7e70a9c98af/url_base/google_image_search.py

import argparse
import json
import itertools
import logging
import re
import os
import uuid
import sys
from urllib.request import urlopen, Request

from bs4 import BeautifulSoup


def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('[%(asctime)s %(levelname)s %(module)s]: %(message)s'))
    logger.addHandler(handler)
    Filehandler = logging.FileHandler("G:\log.txt") #Path to your LOG FILE.
    Filehandler.setFormatter(
        logging.Formatter('[%(asctime)s %(levelname)s %(module)s]: %(message)s'))
    logger.addHandler(Filehandler)
    
    return logger

logger = configure_logging()

REQUEST_HEADER = {
    'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}


def get_soup(url, header):
    response = urlopen(Request(url, headers=header))
    print("-get_soup---",response)
    soup_1 = BeautifulSoup(response, 'html.parser')
    print("---soup_1-----",soup_1)
    # write Soup to Text File 
    file = open("soup_text.txt", "w")
    file.write(str(soup_1))
    file.close()


    return soup_1

def get_query_url(query):
    print("----get_query_url---",query)

    #quey_url_goog_cats = "https://www.google.com/search?q=cats&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiZ8vC4l9L2AhXP4jgGHRkwBS8Q_AUoAXoECAQQAw&biw=1299&bih=669&dpr=1"
    quey_url_goog_cats =  "https://www.google.com/search?q=cats&tbm=isch&ved=2ahUKEwjykJ779tbzAhXhgnIEHSVQBksQ2-cCegQIABAA&oq=cats&gs_lcp=CgNpbWcQAzIHCAAQsQMQQzIHCAAQsQMQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzoHCCMQ7wMQJ1C_31NYvOJTYPbjU2gCcAB4AIABa4gBzQSSAQMzLjOYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=7vZuYfLhOeGFytMPpaCZ2AQ&bih=817&biw=1707&rlz=1C1CHBF_enCA918CA918"
    #url = "https://www.google.com/search?q=cats&tbm=isch&ved=2ahUKEwjykJ779tbzAhXhgnIEHSVQBksQ2-cCegQIABAA&oq=cats&gs_lcp=CgNpbWcQAzIHCAAQsQMQQzIHCAAQsQMQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzoHCCMQ7wMQJ1C_31NYvOJTYPbjU2gCcAB4AIABa4gBzQSSAQMzLjOYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=7vZuYfLhOeGFytMPpaCZ2AQ&bih=817&biw=1707&rlz=1C1CHBF_enCA918CA918"
    #return "https://www.google.co.in/search?q=%s&source=lnms&tbm=isch" % query

    return quey_url_goog_cats


def extract_images_from_soup(soup):
    image_elements = soup.find_all("div", {"class": "rg_meta"})
    print("--extract_images_from_soup---image_elements--",image_elements)

    metadata_dicts = (json.loads(e.text) for e in image_elements)
    link_type_records = ((d["ou"], d["ity"]) for d in metadata_dicts)
    return link_type_records

def extract_images(query, num_images):
    url = get_query_url(query)
    logger.info("Souping")
    soup = get_soup(url, REQUEST_HEADER)


    logger.info("Extracting image urls")
    link_type_records = extract_images_from_soup(soup)
    return itertools.islice(link_type_records, num_images)

def get_raw_image(url):
    req = Request(url, headers=REQUEST_HEADER)
    resp = urlopen(req)
    return resp.read()

def save_image(raw_image, image_type, save_directory):
    extension = image_type if image_type else 'jpg'
    file_name = uuid.uuid4().hex + "." + extension
    save_path = os.path.join(save_directory, file_name)
    with open(save_path, 'wb') as image_file:
        image_file.write(raw_image)

def download_images_to_dir(images, save_directory, num_images):
    for i, (url, image_type) in enumerate(images):
        try:
            logger.info("Making request (%d/%d): %s", i, num_images, url)
            raw_image = get_raw_image(url)
            save_image(raw_image, image_type, save_directory)
        except Exception as e:
            logger.exception(e)

def run(query, save_directory, num_images=100):
    query = '+'.join(query.split())
    logger.info("Extracting image links")
    images = extract_images(query, num_images)
    logger.info("Downloading images")
    download_images_to_dir(images, save_directory, num_images)
    logger.info("Finished")

def main():
    parser = argparse.ArgumentParser(description='Scrape Google images')
    parser.add_argument('-s', '--search', default='bananas', type=str, help='search term')
    parser.add_argument('-n', '--num_images', default=100, type=int, help='num images to save')
    parser.add_argument('-d', '--directory', default='G:\__main__\tf_files\flower_photos', type=str, help='save directory')
    args = parser.parse_args()
    run(args.search, args.directory, args.num_images)

if __name__ == '__main__':
    main()
