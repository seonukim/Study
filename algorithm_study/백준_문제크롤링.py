# 백준 문제 긁어오기
from urllib.request import urlopen
import bs4
​
# q_num=1000
q_num=input("문제번호는?")
url = f"https://www.acmicpc.net/problem/{q_num}"
source = urlopen(url).read()
source_bs4 = bs4.BeautifulSoup(source,"html.parser")
name = __file__.split("\\")[-1]
# folder=__file__[:-len(name)-1].split("\\")[-1]
path = __file__[:-len(name)] + "descript.txt"
print(path)
with open(path, "w", encoding = "utf-8") as file:
    
# print(source)
    title = source_bs4.find('title').string
    file.write("\n")
    file.write(f"--url--\n")
    file.write(f"{url}\n")
    file.write("\n")
    file.write("--title--\n")
    file.write(f"{title}\n")
    file.write("\n")
    file.write("--problem_description--\n")
    text_0 = source_bs4.find('div', id = "problem_description").find_all('p')
    for t in text_0:
        file.write(f"{t.string}\n")
        file.write("\n")
​
    file.write("--problem_input--\n")
    text_0 = source_bs4.find('div', id = "problem_input").find_all('p')
    for t in text_0:
        file.write(f"{t.string}\n")
        file.write("\n")
​
    file.write("--problem_output--\n")
    text_0 = source_bs4.find('div', id = "problem_output").find_all('p')
    for t in text_0:
        file.write(f"{t.string}\n")
        file.write("\n")