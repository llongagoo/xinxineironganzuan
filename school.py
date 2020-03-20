import requests
from bs4 import BeautifulSoup
if __name__ == '__main__':
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:70.0) Gecko/20100101 Firefox/70.0'
    }   #伪装ip防止被封
    url = 'http://gaokao.xdf.cn/202001/11019357.html'   #爬取网页的网址
    fp = open('./中国大学排行.txt','w',encoding='utf-8')
    response = requests.get(url=url,headers=headers)
    response.encoding = 'utf-8' #定义content-type防止乱码
    response = response.text
    soup = BeautifulSoup(response,'lxml')
    schoole_list = soup.select('.air_con.f-f0 tr')
    for li in schoole_list:
        detail = li.text
        school_detail = (' '.join(detail.split())+'\n')
        print(school_detail)
        fp.write(school_detail) #遍历输出并储存