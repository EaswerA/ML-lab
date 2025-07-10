def findrange(nums):
    if len(nums) < 3:
        return "Range determination not possible"
    
    maxval = max(nums)
    minval = min(nums)
    return maxval - minval

nums = [5, 3, 8, 1, 0, 4]
result = findrange(nums)
print("Range:", result)
