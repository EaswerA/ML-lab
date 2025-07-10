import random
import statistics

nums = [random.randint(1, 10) for i in range(25)]

mean = statistics.mean(nums)
median = statistics.median(nums)
mode = statistics.mode(nums)

print("Generated numbers:", nums)
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
