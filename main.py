from similarityCalculation import similarityCalculation

# Definir el umbral de que es plagio o no
TXT_FILES_PATH = 'DataBase'
UMBRAL = 0.8

if __name__ == '__main__':
    '''
    Este archivo es solo para inicializar la clase 'similarityCalculation'.
    Una vez inicializada, la función 'plagiarismDetection' se encarga del calculo.
    A partir de un documento, define si este es plagio o no.

    Si no es plagio regresa:
        Falso

    Si es plagio regresa:
        True
        Archivo original de donde se realizó el pagio
        Porcentaje de similitud
    '''
    file_to_analyse = 'test_input.txt'
    
    plagiarism = similarityCalculation(TXT_FILES_PATH, UMBRAL)
    
    result = plagiarism.plagiarismDetection(file_to_analyse)
    print(result)
