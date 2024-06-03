import os
import glob
import logging
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch


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
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
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
        is_tp : bool
            True si 'TP' está en el nombre del archivo, False en caso contrario.
        """
        files_and_content_processed = self.dataBaseProcessing()
        input_text = self._read_file(input_file_path)
        preprocessed_input_text = self._preprocess_text(input_text)
        
        most_similar_file, similarity_score = self.similarityComparison(preprocessed_input_text, files_and_content_processed)
        
        is_plagiarism = similarity_score >= self.percentaje_simil
        is_tp = 'TP' in os.path.basename(input_file_path)

        if similarity_score >= self.percentaje_simil:
            is_plagiarism = True
            return is_plagiarism, most_similar_file, similarity_score, is_tp
        
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
        input_embedding = self._get_embedding(preprocessed_input_text)
        max_similarity = 0
        most_similar_file = None

        for file_name, content in files_and_content.items():
            content_embedding = self._get_embedding(content)
            similarity = self._cosine_similarity(input_embedding, content_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_file = file_name

        logging.debug(f'Archivo más similar: {most_similar_file} con una similitud de {max_similarity}')
        
        return most_similar_file, max_similarity
    
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
        
    def _get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Mean pooling

    def _cosine_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(tensor1, tensor2).item()
    
    def _log_content(self, files_and_content: dict) -> None:
        for nombre, content in files_and_content.items():
            logging.info(f'Archivo: {nombre}\nContenido:\n{content}\n{"-"*40}')

    
    def evaluate_directory(self, evaluation_dir: str):
        """
        Aplica la detección de plagio a todos los documentos en un directorio específico.
        
        Parámetros:
        -----------
        evaluation_dir : str
            Ruta al directorio que contiene los archivos de texto a evaluar.
        
        Retorna:
        --------
        auc : float
            Medida de desempeño basada en el conteo de True Positive, True Negative, False Positive y False Negative.
        """
        tp_count = 0
        tn_count = 0
        fp_count = 0
        fn_count = 0
        result = []
        result_negative = []

        evaluation_files = glob.glob(os.path.join(evaluation_dir, '*.txt'))

        for file in evaluation_files:
            is_plagiarism = self.plagiarismDetection(file)
            
            is_tp = 'TP' in os.path.basename(file)

            if is_tp:
                if is_plagiarism:
                    tp_count += 1
                else:
                    fn_count += 1
            else:
                if is_plagiarism:
                    fp_count += 1
                else:
                    tn_count += 1

            # Previo
            # if is_plagiarism:
            #     print(f'Procesando archivo: {os.path.basename(file)} - Plagio: {is_plagiarism[0]} - Archivo Original: {is_plagiarism[1]} - Porcentaje de Similitud: {is_plagiarism[2]} - True Positive: {is_tp}\n')
            # else:
            #     print(f'Procesando archivo: {os.path.basename(file)} - Plagio: {is_plagiarism} - True Positive: {is_tp}\n')
                
            # Modificación reporte
            if is_plagiarism:
                result.append(f'Archivo sospechoso: {os.path.basename(file)} - Plagio: {is_plagiarism[0]} - Documento plagiado: {is_plagiarism[1]} - Porcentaje de similitud: {is_plagiarism[2]}')
            else:
                result_negative.append(f'Procesando archivo: {os.path.basename(file)} - Plagio: {is_plagiarism}')
            
        type_plag = ['Parafraseo',
                     'Parafraseo',
                     'Cambio de tiempo',
                     'Parafraseo',
                     'Insertar o reemplazar frases',
                     'Cambio de voz',
                     'Insertar o reemplazar frases',
                     'Parafraseo',
                     'Cambio de voz']
            
        lista_tipos = [f"{a} {b}" for a, b in zip(result, type_plag)]
        
        lista_combinada = result + result_negative
        
        for element in lista_combinada:
            print(f'{element}\n')

        results = {
            'True Positive': tp_count,
            'True Negative': tn_count,
            'False Positive': fp_count,
            'False Negative': fn_count
        }

        if (tp_count + fn_count) == 0:
            tpr = 0
        else:
            tpr = tp_count / (tp_count + fn_count)
    
        if (fp_count + tn_count) == 0:
         fpr = 0
        else:
            fpr = fp_count / (fp_count + tn_count)
    
        auc = (1 + tpr - fpr)/2

        print (f'AUC: {auc}')

        return auc
