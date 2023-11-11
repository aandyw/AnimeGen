from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
import os

dataset_loc = 'dataset/all_data.csv'
threads = 12


def get_urls(csv_file, num=60000):
    data = pd.read_csv(csv_file)
    urls = data['sample_url']
    return urls[:num]


def downloader(url, i):
    with open("dataset/faces/{}.jpg".format(i), 'wb') as handler:
        data = requests.get(url).content
        handler.write(data)
    print('downloaded {}.jpg from {}'.format(i, url))


def main(urls):
    print('[INFO] Downloading {} images'.format(len(urls)))
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for i, url in enumerate(urls):
            executor.submit(downloader, url, i)

    print('[INFO] Finished downloading {} images'.format(len(urls)))


if __name__ == '__main__':
    try:
        os.makedirs('dataset/faces')
    except FileExistsError:
        pass
    urls = get_urls(dataset_loc)
    main(urls)
