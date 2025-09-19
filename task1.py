def find_s_algorithm(data):
    hypothesis = ['Ø'] * (len(data[0]) - 1)
    for row in data:
        instance, label = row[:-1], row[-1]
        if label.lower() == 'yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] == 'Ø':
                    hypothesis[i] = instance[i]
                elif hypothesis[i] != instance[i]:
                    hypothesis[i] = '?'
    return hypothesis
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]
final_hypothesis = find_s_algorithm(training_data)
print("Final Hypothesis (Most Specific):", final_hypothesis)
