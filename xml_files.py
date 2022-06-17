from bs4 import BeautifulSoup


def get_xml_thingies(filepath):
    file = open(filepath, 'r')
    contents = file.read()
    soup = BeautifulSoup(contents, 'lxml')

    n_calibrated = None
    if len(soup.find_all('images')) > 0:
        n_calibrated = soup.find_all("images")[1]['calibrated']

    GSD = None
    if len(soup.find_all('gsd')) > 0:
        GSD = soup.find_all('gsd')[0]['cm']

    optim = None
    if len(soup.find_all('cameraoptimization')) > 0:
        optim = soup.find_all('cameraoptimization')[0]['relativedifference']

    D2_BBA = None
    if len(soup.find_all('trackhistogram')) > 0:
        D2_BBA = soup.find_all('trackhistogram')[0]['observed2dpoints']

    D3_BBA = None
    if len(soup.find_all('trackhistogram')) > 0:
        D3_BBA = soup.find_all('trackhistogram')[0]['numberof3dpoints']

    keypoints_img = None
    if len(soup.find_all('distribution')) > 0:
        keypoints_img = soup.find_all('distribution')[0]['median']

    matches_img = None
    if len(soup.find_all('distribution')) > 0:
        matches_img = soup.find_all('distribution')[1]['median']

    mre = None
    if len(soup.find_all('atps')) > 0:
        mre = soup.find_all('atps')[0]['meanprojectionerror']

    return (n_calibrated, GSD, optim, D2_BBA, D3_BBA, keypoints_img, matches_img, mre)

def rec_print(dyct,t):
    for k, v in dyct.items():
        if type(v) is dict:
            print('- '*t+str(k))
            t=t+1
            rec_print(v,t)
            t=t-1
        else:
            print('- '*t+str(k) +' : '+ str(v))