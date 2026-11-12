a = [2,5,4]
index = [0,0,0]
i = 2

import time
overflow = False
last = len(index) - 1
while True:
    
    index[last] += 1
    if (index[last] == a[last]):
        overflow = True

    if (overflow):
        for i in range(last, 0 , -1):
            if (index[i] == a[i]):
                index[i] = 0
                index[i - 1] += 1
            else:
                break
        overflow = False
        if index[0] == a[0]:
            break
    print(index)

# def gen():
#     i = 2
#     overflow = False
#     tmp = index.copy()
#     while True:
    
#         if (i == 0 and index[i] == a[0]):
#             break
#         if (overflow):
#             i -= 1
#             index[i] += 1
#             overflow = False
#             continue

#         if (index[i] >= a[i]):
#             index[i] = 0
#             overflow = True
#             continue
            
#         tmp = index.copy()
#         i = len(index) - 1
#         index[i] += 1
#         break

#     yield tmp

# for _ in range(100):
#     print(gen().__next__())
