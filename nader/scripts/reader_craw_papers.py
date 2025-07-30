import requests
from bs4 import BeautifulSoup
import json
import os
import argparse

def download_file(url, filename):
    response = requests.get(url,stream=True)
    response.raise_for_status()
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        print("下载失败，状态码：", response.status_code)

def fetch_cvpr_abstract(url):
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'html.parser')
    abstract = soup.find('div',id='abstract')
    return abstract.text.strip()

def fetch_cvpr_papers(year,out_dir):
    anno_path = out_dir+'/annotations.json'
    paper_dir = out_dir+'/papers'
    abstract_dir = out_dir+'/abstracts'
    os.makedirs(paper_dir,exist_ok=True)
    os.makedirs(abstract_dir,exist_ok=True)
    url = f"https://openaccess.thecvf.com/CVPR{year}?day=all"
    base_url = 'https://openaccess.thecvf.com'
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        dl = soup.find_all('dl')[0]
        dts = dl.find_all('dt')
        dds = dl.find_all('dd')
        with open(anno_path,'w') as f:
            for i,(dt,dd1,dd2) in enumerate(zip(dts,dds[1::2],dds[2::2])):
                print(f"{i}/{len(dts)}")
                title = dt.text.strip()
                abstract = fetch_cvpr_abstract(base_url+dt.find('a')['href'])
                authors = [a.strip() for a in dd1.text.strip().split(',')]
                links = dd2.find_all('a')
                paper_link = base_url+links[0]['href']
                anno = {'id':i+1,'title':title,'authors':authors,'paper_link':paper_link}
                f.write(json.dumps(anno)+'\n')
                download_file(paper_link,os.path.join(paper_dir,f'paper{i+1}.pdf'))
                with open(os.path.join(abstract_dir,f"abstract{i+1}.txt"),'w') as f2:
                    f2.write(abstract)
    else:
        print("Failed to retrieve data")



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvpr-year',default=2023)
    parser.add_argument('--out-dir',default='papers')
    args = parser.parse_args()
    out_dir = os.path.join(args.out_dir,'cvpr-'+args.cvpr_year)
    fetch_cvpr_papers(args.cvpr_year,out_dir)