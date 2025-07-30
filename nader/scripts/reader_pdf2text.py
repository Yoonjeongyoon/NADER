import pdfplumber
import statistics
import tiktoken
import fitz
import re
import os
import argparse

def get_line_font_size(page,target_line):
    blocks = page.get_text('dict')['blocks']
    for block in blocks:
        if 'lines' in block:
            for line in block['lines']:
                txts,sizes = [],[]
                for span in line['spans']:
                    txts.append(span['text'])
                    sizes.append(span['size'])
                if ''.join(txts)==target_line:
                    return txts,sizes
    return [],[]

def extract_text_from_dual_column_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        flag = False
        section = 1
        doc = fitz.open(pdf_path) # Use fitz to get the text boxes, assuming two columns layout
        for page in pdf.pages:
            page_num = page.page_number
            page_fitz = doc[page_num-1]  # fitz is 0-based index
            lines = page_fitz.get_text().split('\n')
            for i,line in enumerate(lines):
                if line=='Abstract':
                    flag = True
                    text+='Abstract\n'
                    continue
                elif line in ['References','Appendix'] or line.startswith('Acknowledgement'):
                    return text
                if flag:
                    # title
                    matches = re.findall(rf'^{section}[\s]*\.[\s]*[A-Z][^.]*', line)
                    if len(matches)>0:
                        assert len(matches)==1
                        _,sizes = get_line_font_size(page_fitz,line)
                        if len(set(sizes))==1 and sizes[0]>11:
                            text=text+'\n\n'+line
                            if not line.endswith('-'):
                                text+='\n'
                            section+=1
                            continue
                    # subtitle
                    matches = re.findall(r'^\d\.\d[\s|\.]+[A-Z][^.]*', line)
                    if len(matches)>0:
                        assert len(matches)==1
                        text=text+'\n'+line
                        if not line.endswith('-'):
                            text+='\n'
                        continue
                    if (len(line)<40 and (not (line.endswith('.') or line.endswith(':')) or text.endswith('\n'))) or line.startswith('arXiv:'):
                        continue
                    # figure caption
                    matches = re.findall(r'^Figure [\d]+\.', line)
                    if len(matches)>0:
                        assert len(matches)==1
                        # continue
                        if not text.endswith('\n'):
                            text+='\n'
                    # table caption
                    matches = re.findall(r'^Table [\d]+\.', line)
                    if len(matches)>0:
                        assert len(matches)==1
                        # continue
                        if not text.endswith('\n'):
                            text+='\n'
                    if line.endswith('-'):
                        line=line[:-1]
                        text+=line
                        continue
                    text+=line
                    if line.endswith('.') and not line.endswith('e.g.') or line.endswith(':'):
                        text+='\n'
                    else:
                        text+=' '
    return text

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf-dir', default='papers/cvpr-2023/papers')
    parser.add_argument('--txt-dir',default='papers/cvpr-2023/txts')
    parser.add_argument('--token_num_path',default='papers/cvpr-2023/txt_token_nums.txt')
    args = parser.parse_args()

    os.makedirs(args.txt_dir,exist_ok=True)
    token_nums = []
    with open(args.token_num_path,'w') as f:
        for file in os.listdir(args.pdf_dir):
            pdf_path = os.path.join(args.pdf_dir,file)
            txt_path = os.path.join(args.txt_dir,file.replace('.pdf','.txt'))
            try:
                text_content = extract_text_from_dual_column_pdf(pdf_path)
            except Exception as e:
                print(f"error:{file},{str(e)}")
                f.write(f"{file.replace('.pdf','')} error\n")
                f.flush()
                continue
            with open(txt_path,'w') as f2:
                f2.write(text_content)
            token_num = num_tokens_from_string(text_content,'gpt-4')
            token_nums.append(token_num)
            f.write(f"{file.replace('.pdf','')} {token_num}\n")
            f.flush()
    print(f"mean token num:{statistics.mean(token_nums)}")
