import concurrent.futures
import concurrent.futures
import time
from urllib.error import HTTPError
from urllib.request import urlretrieve

from tqdm import tqdm


def find_in_log_n(start, end, func, bias=0.3):
    if end - start <= 1:
        return start
    mid = int(start * (1 - bias) + end * bias)
    if start == mid:
        mid += 1
    if func(mid):
        return find_in_log_n(mid, end, func)
    else:
        return find_in_log_n(start, mid, func)


def fetch_tile_content(tile_url, retry=15):
    for i in range(retry):
        try:
            image_path = urlretrieve(tile_url)[0]
            with open(image_path, "rb") as file:
                return file.read()
        except Exception as e:
            print("e", end="|")
            time.sleep(2 ** (0.3 * (i + 1)))
            if i == retry - 1:
                if input("continue") == "y":
                    return fetch_tile_content(tile_url, retry)
                raise e
    raise HTTPError("Not able for fetch image tile", code=500, msg="", hdrs={}, fp=None)


def download_urls_in_thread(url_and_index_list):
    def download(args):
        url, index = args
        file_content = fetch_tile_content(url)
        return file_content, index

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        for tile, i in tqdm(executor.map(download, url_and_index_list), total=len(url_and_index_list)):
            yield tile, i


if __name__ == '__main__':
    import math

    print(math.log2(1000 * 1000))
    print(find_in_log_n(0, 100, lambda x: x <= 76))
