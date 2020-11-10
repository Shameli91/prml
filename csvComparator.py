import csv

with open('./main/predictions/predictions01.csv', 'r') as file:
    fileOne = file.readlines()

with open('./main/predictions/predictions02.csv', 'r') as file:
    fileTwo = file.readlines()

count = 0
for i in range(0, 4513):
    if fileOne[i] != fileTwo[i]:
        count += 1
        print('line1: ', fileOne[i], 'line2: ', fileTwo[i], '\n')
print('different lines: ', count)