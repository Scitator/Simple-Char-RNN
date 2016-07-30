with open('shakespeare_original.txt','r') as infile, open('shakespeare_input.txt', 'w') as outfile:
    lines = []
    for line in infile:
        line = line.strip()
        if not line:
            continue
        line.replace( '\s+', ' ')
        lines.append(line)
        lines.append(' ')
    outfile.write(''.join(lines))
