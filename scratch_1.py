

def test_function(list1, list2, position):
    outlist1 = []
    outlist2 = []

    for outer_index, grouping in enumerate(list2):
        possible_combinators = []
        for number in grouping:
            if number >= position:
                possible_combinators.append(number)

        if len(list1) > 0:
            for number in possible_combinators:
                temp1 = list1[outer_index].copy()
                temp2 = list2[outer_index].copy()

                temp1.append(number)
                temp2.remove(number)

                outlist1.append(temp1)
                outlist2.append(temp2)
        else:
            for number in possible_combinators:
                temp1 = []
                temp2 = list2[outer_index].copy()

                temp1.append(number)
                temp2.remove(number)

                outlist1.append(temp1)
                outlist2.append(temp2)

    outpos = position -1

    return outlist1, outlist2, outpos


grand_total = 0
largest = 4
for size in range(largest, 0, -1):
    start_set = [[n for n in range(largest,0,-1)]]

    all_combos = []
    leftovers = start_set
    position = size

    while position > 0:
        all_combos, leftovers, position = test_function(all_combos,leftovers,position)

    grand_total +=len(all_combos)

print(grand_total)

