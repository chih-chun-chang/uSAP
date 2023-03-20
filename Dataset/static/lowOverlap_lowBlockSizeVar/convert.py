f = open("static_lowOverlap_lowBlockSizeVar_300_nodes.tsv", "w")
with open("static_lowOverlap_lowBlockSizeVar_1000_nodes.tsv", "r") as f2:
    for line in f2:
        numbers_str = line.split()
        numbers_int = [int(x) for x in numbers_str]
        if numbers_int[1] > 300: continue
        else: f.write(line)
f.close()

