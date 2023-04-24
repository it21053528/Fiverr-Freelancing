n = int(input("Enter the number of elements in the list: "))
my_list = []
for i in range(n):
    item = int(input("Enter the item to be added to the list: "))
    my_list.append(item)

if len(my_list) > 1 and my_list[0] == my_list[-1]:
    print("The list", my_list, True)
else:
    print("The list", my_list, False)
    