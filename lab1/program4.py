from collections import Counter

def highestoccurrchar(s):
    filtered = [ch.lower() for ch in s if ch.isalpha()]

    if not filtered:
        return "No alphabetic characters found"

    freq = Counter(filtered)

    char, count = freq.most_common(1)[0]
    return char, count

input_str = "hippopotamus"
char, count = highestoccurrchar(input_str)
print(f"The most frequent occurring character is '{char}' & occurrence count is {count}.")
