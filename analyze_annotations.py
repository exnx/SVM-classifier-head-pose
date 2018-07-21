import csv
import sys
import os


file_name = 'Inderbir-attention-data.csv'

# for reading
with open(file_name, "rt", newline='') as file:

    reader = csv.reader(file, delimiter=',')
    total_count = 0
    nose_count = 0
    lear_count = 0
    rear_count = 0
    neck_count = 0
    leye_count = 0
    reye_count = 0

    error_one_count = 0
    error_two_count = 0

    attn_0 = 0
    attn_1 = 0
    attn_2 = 0
    attn_other = 0

    firstline = True

    for row in reader:

        if row:

            if firstline:
                firstline = False
                continue

            attention_label = row[18]

            if int(attention_label) == 0:
                attn_0 += 1
            elif int(attention_label) == 1:
                attn_1 += 1
            elif int(attention_label) == 2:
                attn_2 += 1
            else:
                attn_other += 1



            total_count += 1

            house = row[0]
            frame = row[1]
            face_found = row[2]
            attention = row[9]
            nose_x = row[14]
            nose_y = row[15]
            lear_y = row[13]
            rear_y = row[10]
            neck_x = row[16]
            leye_x = row[8]
            reye_x = row[6]

            if float(nose_y) > 0:
                nose_count += 1

            if float(lear_y) > 0:
                lear_count += 1

            if float(rear_y) > 0:
                rear_count += 1

            if float(neck_x) > 0:
                neck_count += 1

            if float(leye_x) > 0:
                leye_count += 1

            if float(reye_x) > 0:
                reye_count += 1

        # if attention == '1' and face_found == 'NA':
        #     error_one_count += 1
        #
        # if attention == '2' and face_found == 'NA':
        #     error_two_count += 1

    print('nose count ', nose_count)
    print('nose % ', nose_count/total_count)

    print('lear count ', lear_count)
    print('lear % ', lear_count/total_count)

    print('rear count ', rear_count)
    print('rear % ', rear_count/total_count)

    print('neck count ', neck_count)
    print('neck % ', neck_count/total_count)

    print('leye count ', leye_count)
    print('leye % ', leye_count/total_count)

    print('reye count ', reye_count)
    print('reye % ', reye_count/total_count)

    print('\n')

    print('attn 0 count: ', attn_0)
    print('attn 0 % ', attn_0/total_count)

    print('attn 1 count: ', attn_1)
    print('attn 1 % ', attn_1/total_count)

    print('attn 2 count: ', attn_2)
    print('attn 2 % ', attn_2/total_count)

    print('attn other count: ', attn_other)
    print('attn other % ', attn_other/total_count)
