def txt_to_dic(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n')
    return lines

def extract_num(lines):
    for line in lines:
        if ('train' in line):
            if ('acc' in line):
                des = 'tr_acc.txt'
                line = line[10:]
            elif ('loss' in line):
                des = 'tr_loss.txt'
                line = line[12:]
        elif ('test' in line):
            if ('acc' in line):
                des = 'ts_acc.txt'
                line = line[9:]
            elif ('loss' in line):
                des = 'ts_loss.txt'
                line = line[11:]
        else:
            continue
        with open(des, 'a') as f:
            f.write(line+'\n')



if __name__ == '__main__':
    lines = txt_to_dic('training.txt')
    extract_num(lines)
