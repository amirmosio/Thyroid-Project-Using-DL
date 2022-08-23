n = int(input())
array = ["" for i in range(100)]

for i in range(n):
    command = input().split()
    order = int(command[0])
    index = int(command[1])
    lenth = int(command[2])
    string = command[3]

    if order == 1:
        if index > len(array):
            array.append(string)
        else:
            array[index - 1] += string
    else:
        sum = 0
        merged_string = array[index - 1]
        for i in range(len(merged_string)):
            if i + lenth <= len(merged_string):
                if merged_string[i: i + lenth] == string:
                    sum += 1
        print(sum)
