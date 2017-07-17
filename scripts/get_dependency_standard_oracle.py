import sys
import dependency_dict

# tokens is a list of tokens, so no need to split it again
def unkify(tokens, words_dict):
    final = []
    for token in tokens:
        # only process the train singletons and unknown words
        if len(token.rstrip()) == 0:
            final.append('UNK')
        elif not(token.rstrip() in words_dict):
            numCaps = 0
            hasDigit = False
            hasDash = False
            hasLower = False
            for char in token.rstrip():
                if char.isdigit():
                    hasDigit = True
                elif char == '-':
                    hasDash = True
                elif char.isalpha():
                    if char.islower():
                        hasLower = True
                    elif char.isupper():
                        numCaps += 1
            result = 'UNK'
            lower = token.rstrip().lower()
            ch0 = token.rstrip()[0]
            if ch0.isupper():
                if numCaps == 1:
                    result = result + '-INITC'    
                    if lower in words_dict:
                        result = result + '-KNOWNLC'
                else:
                    result = result + '-CAPS'
            elif not(ch0.isalpha()) and numCaps > 0:
                result = result + '-CAPS'
            elif hasLower:
                result = result + '-LC'
            if hasDigit:
                result = result + '-NUM'
            if hasDash:
                result = result + '-DASH' 
            if lower[-1] == 's' and len(lower) >= 3:
                ch2 = lower[-2]
                if not(ch2 == 's') and not(ch2 == 'i') and not(ch2 == 'u'):
                    result = result + '-s'
            elif len(lower) >= 5 and not(hasDash) and not(hasDigit and numCaps > 0):
                if lower[-2:] == 'ed':
                    result = result + '-ed'
                elif lower[-3:] == 'ing':
                    result = result + '-ing'
                elif lower[-3:] == 'ion':
                    result = result + '-ion'
                elif lower[-2:] == 'er':
                    result = result + '-er'            
                elif lower[-3:] == 'est':
                    result = result + '-est'
                elif lower[-2:] == 'ly':
                    result = result + '-ly'
                elif lower[-3:] == 'ity':
                    result = result + '-ity'
                elif lower[-1] == 'y':
                    result = result + '-y'
                elif lower[-2:] == 'al':
                    result = result + '-al'
            final.append(result)
        else:
            final.append(token.rstrip())
    return final 


def get_tags_tokens_lowercase(tree):
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for item in tree:
        output_tags.append(item[1])
        output_tokens.append(item[0])
        output_lowercase.append(item[0].lower())
    return [output_tags, output_tokens, output_lowercase]    

def canArc(head, chil, stk, buf, tree):
    if tree[chil][-2] != head:
	return False
    for i in stk:
	if tree[i][-2] == chil:
	    return False
    for i in buf:
	if tree[i][-2] == chil:
	    return False
    return True
def get_actions(tree):
    output_actions = []
    stk = []
    buf = []
    for i in range(len(tree)):
	buf.append(i)
    while True:
	if len(stk) == 1 and len(buf) == 0:
	    break
	if len(stk) >= 2:
	    s0 = stk[-1]
	    s1 = stk[-2]
	    stk = stk[:-2]
	    if canArc(s1, s0, stk, buf, tree):
		output_actions.append("RIGHT-ARC("+tree[s0][-1]+")")
		stk.append(s1)
	    elif canArc(s0, s1, stk, buf, tree):
		output_actions.append("LEFT-ARC("+tree[s1][-1]+")")
		stk.append(s0)
	    elif len(buf) > 0:
		output_actions.append("SHIFT")
		stk.append(s1)
		stk.append(s0)
		stk.append(buf[0])
		buf = buf[1:]
	    else:
		break
	else:
	    if len(buf) > 0:
	    	output_actions.append("SHIFT")
		stk.append(buf[0])
		buf = buf[1:]
	    else:
		break
    output_actions.append("SHIFT")
    output_actions.append("LEFT-ARC(root)")
    if len(stk) != 1 or len(buf) != 0:
	return False, output_actions
    else:
	return True, output_actions

def readfile(filename):
    trees = []
    tree = []
    for line in open(filename,'r'):
	line = line.strip()
	if line == "":
		trees.append(tree)
		tree = []
	else:
		line = line.split()
		tree.append([line[1],line[3],int(line[6])-1,line[7]])
    if len(tree) != 0:
	trees.append(tree)
    return trees

def main():
    if len(sys.argv) != 3:
        raise NotImplementedError('Program only takes two arguments:  train file and dev file (for vocabulary mapping purposes)')
    trees = readfile(sys.argv[1])
    dev_trees = readfile(sys.argv[2])
    words_list = dependency_dict.get_dict(trees) 
    # get the oracle for the train file
    for tree in dev_trees:
        tags, tokens, lowercase = get_tags_tokens_lowercase(tree)
        assert len(tags) == len(tokens)
        assert len(tokens) == len(lowercase)
        unkified = unkify(tokens, words_list)    
        flag, output_actions = get_actions(tree)
	tags.append("ROOT")
        tokens.append("ROOT")
        lowercase.append("ROOT")
        unkified.append("ROOT")
	if flag:
	    print ' '.join(tags)
	    print ' '.join(tokens)
	    print ' '.join(lowercase)
	    print ' '.join(unkified)
            for action in output_actions:
                print action
            print ''
    
if __name__ == "__main__":
    main()
