import os
import glob
import logging
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class similarityCalculation:
    """
    Clase para calcular la similitud entre textos y detectar plagio.

    Atributos:
        lemmatizer (WordNetLemmatizer): Instancia del lematizador de WordNet.
        files_path (str): Ruta a la carpeta que contiene los archivos .txt para la base de datos.
        percentaje_simil (float): Umbral de similitud para considerar un texto como plagio.
    """

    def __init__(self, txt_files_path: str, percentaje_simil: float) -> None:
        """
        Inicializa una instancia de la clase similarityCalculation.

        Args:
            txt_files_path (str): Ruta a la carpeta que contiene los archivos .txt para la base de datos.
            percentaje_simil (float): Umbral de similitud para considerar un texto como plagio.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.files_path = txt_files_path
        self.percentaje_simil = percentaje_simil
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    def plagiarismDetection(self, input_file_path: str):
        """
        Detecta plagio comparando un archivo de entrada con una base de datos de textos.

        Args:
            input_file_path (str): Ruta al archivo de entrada que se desea verificar.

        Returns:
            tuple: Un booleano indicando si se detectó plagio, el nombre del archivo más similar,
            y la puntuación de similitud.
        """
        is_plagiarism = False
        files_and_content_processed = self.dataBaseProcessing()
        input_text = self._read_file(input_file_path)
        lemmatized_input_text = self._lemmatize_text(input_text)

        most_similar_file, similarity_score = self.similarityComparison(lemmatized_input_text, files_and_content_processed)

        if similarity_score >= self.percentaje_simil:
            is_plagiarism = True
            return is_plagiarism, most_similar_file, similarity_score

        return is_plagiarism


    def similarityComparison(self, lemmatized_input_text: str, files_and_content: dict) -> tuple:
        """
        Compara un texto lematizado con una base de datos de textos y encuentra el más similar.

        Args:
            lemmatized_input_text (str): Texto de entrada lematizado.
            files_and_content (dict): Diccionario con nombres de archivos y sus contenidos lematizados.

        Returns:
            tuple: El nombre del archivo más similar y la puntuación de similitud.
        """
        texts = [lemmatized_input_text] + list(files_and_content.values())
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
        Procesa la base de datos de archivos .txt lematizándolos y almacenándolos en un diccionario.

        Returns:
            dict: Diccionario con nombres de archivos y sus contenidos lematizados.
        """
        files_and_content_processed = self._uploadDatabase(self.files_path)
        return files_and_content_processed


    def _lemmatize_text(self, text: str) -> str:
        """
        Lematiza un texto.

        Args:
            text (str): Texto a lematizar.

        Returns:
            str: Texto lematizado.
        """
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)


    def _uploadDatabase(self, filesPath: str) -> dict:
        """
        Sube y procesa la base de datos de archivos .txt desde una ruta especificada.

        Args:
            filesPath (str): Ruta a la carpeta que contiene los archivos .txt.

        Returns:
            dict: Diccionario con nombres de archivos y sus contenidos lematizados.
        """
        txt_files = glob.glob(os.path.join(filesPath, '*.txt'))
        files_and_content = {}
        for file in txt_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    file_name = os.path.basename(file)
                    files_and_content[file_name] = self._lemmatize_text(content)
                    logging.debug(f'Read file: {file_name}')
            except Exception as e:
                logging.error(f'Error al leer el archivo {file}: {e}')

        return files_and_content


    def _read_file(self, path: str) -> str:
        """
        Lee el contenido de un archivo.

        Args:
            path (str): Ruta al archivo que se desea leer.

        Returns:
            str: Contenido del archivo.
        """
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            logging.error(f'Error al leer el archivo: {e}')
            content = ''

        return content


    def _log_content(self, files_and_content: dict) -> None:
        """
        Registra el contenido de los archivos lematizados.

        Args:
            files_and_content (dict): Diccionario con nombres de archivos y sus contenidos lematizados.
        """
        for nombre, content in files_and_content.items():
            logging.info(f'Archivo: {nombre}\nContenido:\n{content}\n{"-"*40}')