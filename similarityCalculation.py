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
    
    def __init__(self, txt_files_path: str, percentaje_simil: float) -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.files_path = txt_files_path
        self.percentaje_simil = percentaje_simil
        self.stop_words = set(stopwords.words('english'))
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s\n%(message)s')
        
    def plagiarismDetection(self, input_file_path: str):
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
        files_and_content_processed = self._uploadDatabase(self.files_path)
        return files_and_content_processed
    
    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        processed_tokens = [token for token in processed_tokens if not token.isdigit()]
        return ' '.join(processed_tokens)
    
    def _uploadDatabase(self, filesPath: str) -> dict:
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
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            logging.error(f'Error al leer el archivo: {e}')
            content = ''
        return content
    
    def _log_content(self, files_and_content: dict) -> None:
        for nombre, content in files_and_content.items():
            logging.info(f'Archivo: {nombre}\nContenido:\n{content}\n{"-"*40}')