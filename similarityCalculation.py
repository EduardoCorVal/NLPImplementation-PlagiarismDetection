import os
import glob
import logging
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class similarityCalculation:
    """
    Clase para calcular la similitud entre documentos de texto y detectar plagio.
    
    Atributos:
    ----------
    lemmatizer : WordNetLemmatizer
        Objeto para lematizar palabras.
    files_path : str
        Ruta a la carpeta que contiene los archivos de texto.
    percentaje_simil : float
        Umbral de similitud para determinar el plagio.
    stop_words : set
        Conjunto de palabras vacías (stopwords) en inglés.
    """

    def __init__(self, txt_files_path: str, percentaje_simil: float) -> None:
        """
        Inicializa la clase similarityCalculation con la ruta de los archivos de texto
        y el umbral de similitud.
        
        Parámetros:
        -----------
        txt_files_path : str
            Ruta a la carpeta que contiene los archivos de texto.
        percentaje_simil : float
            Umbral de similitud para determinar el plagio.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.files_path = txt_files_path
        self.percentaje_simil = percentaje_simil
        self.stop_words = set(stopwords.words('english'))
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s\n%(message)s')
        
    def plagiarismDetection(self, input_file_path: str):
        """
        Detecta plagio comparando un archivo de entrada con los archivos en la base de datos.
        
        Parámetros:
        -----------
        input_file_path : str
            Ruta del archivo de entrada a verificar.
        
        Retorna:
        --------
        is_plagiarism : bool
            True si se detecta plagio, False en caso contrario.
        most_similar_file : str
            Nombre del archivo más similar en caso de plagio.
        similarity_score : float
            Puntaje de similitud del archivo más similar.
        """
        is_plagiarism = False
        files_and_content_processed = self.dataBaseProcessing()
        input_text = self._read_file(input_file_path)
        preprocessed_input_text = self._preprocess_text(input_text)
        
        most_similar_file, similarity_score = self.similarityComparison(preprocessed_input_text, files_and_content_processed)
        
        if similarity_score >= self.percentaje_simil:
            is_plagiarism = True
            return is_plagiarism, most_similar_file, similarity_score
        
        return is_plagiarism
    
    def similarityComparison(self, preprocessed_input_text: str, files_and_content: dict):
        """
        Compara el texto preprocesado de entrada con los textos en la base de datos.
        
        Parámetros:
        -----------
        preprocessed_input_text : str
            Texto de entrada preprocesado.
        files_and_content : dict
            Diccionario con los nombres de archivos y sus contenidos preprocesados.
        
        Retorna:
        --------
        most_similar_file : str
            Nombre del archivo más similar.
        similarity_score : float
            Puntaje de similitud del archivo más similar.
        """
        texts = [preprocessed_input_text] + list(files_and_content.values())
        vectorizer = CountVectorizer().fit_transform(texts)
        vectors = vectorizer.toarray()
        cosine_matrix = cosine_similarity(vectors)
        similarities = cosine_matrix[0][1:]  # Ignora la primera entrada que es el propio input_text

        most_similar_index = similarities.argmax()
        most_similar_file = list(files_and_content.keys())[most_similar_index]
        similarity_score = float(similarities[most_similar_index])

        logging.debug(f'Archivo más similar: {most_similar_file} con una similitud de {similarity_score}')
        
        return most_similar_file, similarity_score
    
    def dataBaseProcessing(self) -> dict:
        """
        Procesa los archivos de texto en la base de datos.
        
        Retorna:
        --------
        files_and_content_processed : dict
            Diccionario con los nombres de archivos y sus contenidos preprocesados.
        """
        files_and_content_processed = self._uploadDatabase(self.files_path)
        return files_and_content_processed
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocesa el texto realizando lematización, eliminación de signos de puntuación,
        conversión a minúsculas, eliminación de stopwords y números.
        
        Parámetros:
        -----------
        text : str
            Texto a preprocesar.
        
        Retorna:
        --------
        str
            Texto preprocesado.
        """
        text = text.lower()  # Convertir a minúsculas
        text = text.translate(str.maketrans('', '', string.punctuation))  # Eliminar signos de puntuación
        tokens = word_tokenize(text)  # Tokenizar el texto
        # Eliminar stopwords y lematizar
        processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        # Eliminar números
        processed_tokens = [token for token in processed_tokens if not token.isdigit()]
        return ' '.join(processed_tokens)
    
    def _uploadDatabase(self, filesPath: str) -> dict:
        """
        Sube los archivos de texto de la base de datos y los preprocesa.
        
        Parámetros:
        -----------
        filesPath : str
            Ruta a la carpeta que contiene los archivos de texto.
        
        Retorna:
        --------
        files_and_content : dict
            Diccionario con los nombres de archivos y sus contenidos preprocesados.
        """
        txt_files = glob.glob(os.path.join(filesPath, '*.txt'))
        files_and_content = {}
        for file in txt_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    file_name = os.path.basename(file)
                    files_and_content[file_name] = self._preprocess_text(content)
                    logging.debug(f'Read file: {file_name}')
            except Exception as e:
                logging.error(f'Error al leer el archivo {file}: {e}')
                
        return files_and_content
    
    def _read_file(self, path: str) -> str:
        """
        Lee el contenido de un archivo de texto.
        
        Parámetros:
        -----------
        path : str
            Ruta del archivo a leer.
        
        Retorna:
        --------
        str
            Contenido del archivo.
        """
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            logging.error(f'Error al leer el archivo: {e}')
            content = ''
        return content

