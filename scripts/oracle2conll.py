
import sys

head = [ -1 for i in range(400)]
rel = [ "" for i in range(400)]
pos = []
words = []
cnt = 0
stk = []
current = 0
for line in open(sys.argv[1]):
	line = line.strip()
	cnt += 1
	if line == "":
		assert len(stk) == 1

		for i in range(current-1):
			if head[i] == -1:
				break
			seq = []
			seq.append(str(i+1))
			seq.append(words[i])
			seq.append("_")
			seq.append(pos[i])
			seq.append(pos[i])	
			seq.append("_")
			if head[i] == current-1:
				seq.append("0")
			else:
				seq.append(str(head[i]+1))
			seq.append(rel[i])
			seq.append("_")
			seq.append("_")
			print "\t".join(seq)
		print	
		stk = []
		current = 0
		cnt = 0
		head = [ -1 for i in range(400)]
		rel = [ "" for i in range(400)]
	else:
		if cnt == 1:
			pos = line.split()
			continue
		if cnt == 2:
			words = line.split()
		if cnt >= 5:
			if line == "SHIFT":
				stk.append(current)
				current += 1
			elif "RIGHT-ARC" in line:
				assert len(stk) >= 2
				head[stk[-1]] = stk[-2]
				rel[stk[-1]] = line[10:-1]
				stk = stk[:-1]
			elif "LEFT-ARC" in line:
				assert len(stk) >= 2
				head[stk[-2]] = stk[-1]
				rel[stk[-2]] = line[9:-1]
				h = stk[-1]
				stk = stk[:-2]
				stk.append(h)
		
			
				
