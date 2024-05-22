import os
import glob
import logging
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class similarityCalculation():
    
    
    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.files_path = 'DataBase/'
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s\n%(message)s')
        
        
    def plagiarismDetection(self, input_file_path: str):
        is_plagiarism = False
        files_and_content_processed = self.dataBaseProcessing()
        input_text = self._read_file(input_file_path)
        lemmatized_input_text = self._lemmatize_text(input_text)
        
        most_similar_file, similarity_score = self.similarityComparison(lemmatized_input_text, files_and_content_processed)
        
        if similarity_score > 0.8:
            is_plagiarism = True
            return is_plagiarism, most_similar_file, similarity_score
        
        return is_plagiarism
        
        
    def similarityComparison(self, lemmatized_input_text: str, files_and_content: dict):
        texts = [lemmatized_input_text] + list(files_and_content.values())
        vectorizer = CountVectorizer().fit_transform(texts)
        vectors = vectorizer.toarray()
        cosine_matrix = cosine_similarity(vectors)
        similarities = cosine_matrix[0][1:]  # Ignora la primera entrada que es el propio input_text

        most_similar_index = similarities.argmax()
        most_similar_file = list(files_and_content.keys())[most_similar_index]
        similarity_score = similarities[most_similar_index]

        logging.debug(f'Archivo más similar: {most_similar_file} con una similitud de {similarity_score}')
        
        return most_similar_file, similarity_score
    
    
    def dataBaseProcessing(self) -> dict:
        files_and_content_processed = self._uploadDatabase(self.files_path)
        return files_and_content_processed
    
    
    def _lemmatize_text(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)
    
    
    def _uploadDatabase(self, filesPath: str) -> dict:
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
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            logging.error(f'Error al leer el archivo: {e}')
          
        return content
    
    
    def _log_content(self, files_and_content: dict) -> None:
        for nombre, content in files_and_content.items():
            logging.info(f'Archivo: {nombre}\nContenido:\n{content}\n{"-"*40}')
        
        