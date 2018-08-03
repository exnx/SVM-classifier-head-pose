import csv
import sys
import os


file_name = 'predicted_labels3.csv'

# for reading
with open(file_name, "rt", newline='') as file:

    reader = csv.reader(file, delimiter=' ')

    total_count = 0

    attn_0 = 0
    attn_1 = 0
    attn_2 = 0
    attn_other = 0

    firstline = True

    for row in reader:

        if row:

            curr_house = row[0].strip()

            if curr_house == 'HH0617-rgb_2017_3_26_22_0_0' or curr_house == 'HH0490-rgb_2016_12_25_21_0_0':

                if firstline:
                    firstline = False
                    continue

                attention_label = row[9]

                if int(attention_label) == 0:
                    attn_0 += 1
                elif int(attention_label) == 1:
                    attn_1 += 1
                elif int(attention_label) == 2:
                    attn_2 += 1
                else:
                    attn_other += 1

                total_count += 1

    print('total count: ', total_count)


    print('attn 0 count: ', attn_0)
    print('attn 0 % ', attn_0/total_count)

    print('attn 1 count: ', attn_1)
    print('attn 1 % ', attn_1/total_count)

    print('attn 2 count: ', attn_2)
    print('attn 2 % ', attn_2/total_count)

    print('attn other count: ', attn_other)
    print('attn other % ', attn_other/total_count)
