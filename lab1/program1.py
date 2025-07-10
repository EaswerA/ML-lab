def count_pairs_with_sum(arr, target):
    seen = {}
    count = 0

    for num in arr:
        comp = target - num
        if seen.get(comp, 0) > 0:
            count += 1
            seen[comp] -= 1  
        else:
            seen[num] = seen.get(num, 0) + 1

    return count

arr = [2, 7, 4, 1, 3, 6]
target = 10
print("Number of pairs with sum 10:", count_pairs_with_sum(arr, target))
