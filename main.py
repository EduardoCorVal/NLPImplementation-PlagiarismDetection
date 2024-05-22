from similarityCalculation import similarityCalculation

# Definir el umbral de que es plagio o no


if __name__ == '__main__':
    plagiarism = similarityCalculation()
    
    result = plagiarism.plagiarismDetection('test_input.txt')
    print(result)
