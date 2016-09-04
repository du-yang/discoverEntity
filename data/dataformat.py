with open('news_lines_splited.txt') as f:
    for i,line in enumerate(f):
        if i>5000:
            break
        print(line.strip().split())