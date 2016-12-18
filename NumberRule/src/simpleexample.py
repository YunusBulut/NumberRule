'''
Created on 17.12.16

@author: Yunus Emrah Bulut
'''

from numberrulealgorithm import NumberRuleAlgorithm as nr

sentence = 'The GDP of Australia is about 400 billion'
keyword = 'GDP'
results = nr().run(keyword, sentence)

if len(results) > 0:
    print('relations: ')
    for r in results:
        print(r)
else:
    print('no relations found!')        
