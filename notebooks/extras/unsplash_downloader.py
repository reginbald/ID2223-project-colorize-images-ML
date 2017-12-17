import requests

pages = 183

for page in range(179, pages+1):
    print('Urls for page ', page)
    url = 'https://api.unsplash.com/search/photos?query=face&page={}&per_page=1000'.format(page)
    client_id = 'Client-ID 6b6499076f3829c60f06b80bcf34aeb202749530d341e647d83f1705c8552fde'
    headers = {'Authorization': client_id}

    r = requests.get(url, headers=headers)
    r = r.json()

    file = open('face_image_urls.txt', 'a')
    for res in r['results']:
        image_url = res['urls']['raw']
        print(res['urls']['raw'] + '\n')
        file.write(image_url + '\n')
    file.close()
