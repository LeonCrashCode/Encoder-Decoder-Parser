
def get_dict(trees):
    words_dict = {} 
    for tree in trees:
	for item in tree:
	    if not(item[0] in words_dict):
            	words_dict[item[0]] = 1
            else:
            	words_dict[item[0]] += 1;
    words_list = []
    for item in words_dict:
        if words_dict[item] > 1:
            words_list.append(item) 
    return words_list 

#if __name__ == '__main__':
#    input_file = open('train.02-21', 'r')
#    lines = input_file.readlines()
#    words_list = get_dict(lines) 
#    #print 'number of words', len(words_list)
#    for word in words_list:
#        print word
#    input_file.close() 
