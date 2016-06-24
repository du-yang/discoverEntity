with open('news_lines.txt') as f:
    for i,line in enumerate(f):
        if i>50:
            break
        print(line.strip())